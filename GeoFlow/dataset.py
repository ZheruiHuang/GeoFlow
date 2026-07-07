import os
import os.path as osp
import heapq
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.decomposition import PCA
from tqdm import tqdm


def _euclid_edge_length(adj01, centroids):
    # adj01: (N,N) 0/1; centroids: (N,2)
    diff = centroids[:, None, :] - centroids[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)  # (N,N)
    return (dist * (adj01 > 0)).astype(np.float64)

def _all_pairs_dijkstra(adj01, weight):
    N = adj01.shape[0]
    neighbors = [np.nonzero(adj01[i])[0] for i in range(N)]
    dist_mat = np.full((N, N), np.inf, dtype=np.float64)
    for s in range(N):
        dist = np.full(N, np.inf, dtype=np.float64)
        dist[s] = 0.0
        pq = [(0.0, s)]
        visited = np.zeros(N, dtype=bool)
        while pq:
            d, u = heapq.heappop(pq)
            if visited[u]:
                continue
            visited[u] = True
            for v in neighbors[u]:
                w = weight[u, v]
                if w <= 0:
                    continue
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        dist_mat[s] = dist
    return dist_mat

def _exp_affinity_from_dist(D, tau, eps=1e-6):
    A = np.exp(-D / max(float(tau), eps))
    A[~np.isfinite(D)] = 0.0
    np.fill_diagonal(A, 1.0)
    return np.clip(A, 0.0, 1.0).astype(np.float32)

def _sym_norm(A):
    deg = A.sum(axis=1) + 1e-12
    inv_sqrt = (deg ** -0.5).astype(np.float32)
    A_norm = A * inv_sqrt[:, None]
    A_norm = A_norm * inv_sqrt[None, :]
    return A_norm.astype(np.float32)


class ODDataset(Dataset):
    def __init__(self, root_dir, filter=None):
        super().__init__()
        self.root_dir = root_dir
        self.data_dir = osp.join(root_dir, "data")
        self.areas = os.listdir(self.data_dir)
        if filter is not None:
            self.areas = [area for area in self.areas if area in filter]
        
        self.dataset = {}
        for area in self.areas:
            area_path = osp.join(self.data_dir, area)
            mask_path = osp.join(area_path, "mask.npy")
            self.dataset[area] = {
                "area": area,
                "adj": np.load(osp.join(area_path, "adj.npy")),
                "demos": np.load(osp.join(area_path, "demos.npy")),
                "pois": np.load(osp.join(area_path, "pois.npy")),
                "dis": np.load(osp.join(area_path, "dis.npy")),
                "od": np.load(osp.join(area_path, "od.npy")),
                "centroids": np.load(osp.join(area_path, "centroids.npy")),
                "mask": (np.load(mask_path) if osp.exists(mask_path) else None),
           }
        assert len(self.areas) == len(self.dataset)
        
        self.dataset = {k: self._preprocess(data) for k, data in tqdm(self.dataset.items())}
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        area = self.areas[idx]
        data = self.dataset[area]
        return data
    
    def _preprocess(self, data):
        feats = np.concatenate([data["demos"], data["pois"]], axis=-1)  # (N, F)
        max_vals = feats.max(axis=0)
        zero_cols = (max_vals == 0)
        safe_max = max_vals.copy()
        safe_max[zero_cols] = 1.0
        feats = feats / safe_max
        
        centroids = data["centroids"]  # (N, 2)
        centroids = centroids - centroids.mean(axis=0)
        pca = PCA(n_components=2)
        centroids = pca.fit_transform(centroids)
        max_abs = np.abs(centroids).max(axis=0)
        max_abs[max_abs == 0] = 1.0
        centroids = centroids / max_abs  # (N, 2)

        adj = data["adj"]  # (N, N)

        od = data["od"]  # (N, N)
        od_scaler = float(od.sum())
        od = od / od_scaler
        mask = data.get("mask")

        feats_t = torch.from_numpy(feats).to(torch.float32)
        centroids_t = torch.from_numpy(centroids).to(torch.float32)
        adj_t = torch.from_numpy(adj).to(torch.float32)
        od_t = torch.from_numpy(od).to(torch.float32)
        mask_t = None if mask is None else torch.from_numpy(mask).to(torch.float32)

        cents = data["centroids"].astype(np.float32)
        adj01 = data["adj"].astype(np.float32)
        L = _euclid_edge_length(adj01, cents)               # (N,N)
        D_ff = _all_pairs_dijkstra(adj01, L)                # (N,N)
        D_hop = _all_pairs_dijkstra(adj01, (adj01 > 0).astype(np.float64))
        pos_ff = D_ff[np.isfinite(D_ff) & (D_ff > 0)]
        tau_ff = float(np.percentile(pos_ff, 50)) if pos_ff.size > 0 else 1.0
        tau_hop = 2.0
        aff_ff = _exp_affinity_from_dist(D_ff, tau=tau_ff)  # (N,N) in [0,1]
        aff_hop = _exp_affinity_from_dist(D_hop, tau=tau_hop)
        A_ff_norm = _sym_norm(aff_ff)
        A_hop_norm = _sym_norm(aff_hop)

        D_geo = data["dis"].astype(np.float64)  # (N,N) straight-line distance
        pos_geo = D_geo[np.isfinite(D_geo) & (D_geo > 0)]
        tau_geo = float(np.percentile(pos_geo, 50)) if pos_geo.size > 0 else 1.0
        aff_geo = _exp_affinity_from_dist(D_geo, tau=tau_geo)
        A_geo_norm = _sym_norm(aff_geo)

        adj_pack = {
            "aff_ff": aff_ff.astype(np.float32),
            "aff_hop": aff_hop.astype(np.float32),
            "A_ff_norm": A_ff_norm.astype(np.float32),
            "A_hop_norm": A_hop_norm.astype(np.float32),
            "aff_geo": aff_geo.astype(np.float32),
            "A_geo_norm": A_geo_norm.astype(np.float32),
        }

        sample = {
            "area": data["area"],
            "feats": feats_t,
            "centroids": centroids_t,
            "adj": adj_t,
            "od": od_t,
            "mask": mask_t,
            "adj_pack": adj_pack,
            "od_scaler": od_scaler
        }
        return sample


def od_collate_fn(batch):
    feat_list = [item["feats"] for item in batch]
    od_list = [item["od"] for item in batch]
    mask_list = [item["mask"] for item in batch]
    adj_pack_list = [item["adj_pack"] for item in batch]
    centroid_list = [item["centroids"] for item in batch]
    area_list = [item["area"] for item in batch]
    od_scaler_list = [item["od_scaler"] for item in batch]

    if all(m is None for m in mask_list):
        mask_list = None

    return {
        "area_list": area_list,
        "feat_list": feat_list,
        "centroid_list": centroid_list,
        "adj_pack_list": adj_pack_list,
        "od_list": od_list,
        "mask_list": mask_list,
        "od_scaler_list": od_scaler_list
    }


def load_train_valid_test_areas(root_dir="."):
    dataset_dir = osp.join(root_dir, "dataset", "seed0")
    with open(osp.join(dataset_dir, "train_subf.json"), "r") as f:
        train_areas = json.load(f)
    with open(osp.join(dataset_dir, "valid_subf.json"), "r") as f:
        valid_areas = json.load(f)
    with open(osp.join(dataset_dir, "test_subf.json"), "r") as f:
        test_areas = json.load(f)
    return train_areas, valid_areas, test_areas
    
