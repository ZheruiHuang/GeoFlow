import argparse
import os
import json
import random
import shutil
import numpy as np
import geopandas as gpd
from tqdm import tqdm


REQUIRED_ARRAYS = ("adj.npy", "demos.npy", "pois.npy", "dis.npy", "od.npy")


def preprocess(raw_data_dir, asset_dir, out_root, output_json=None):
    data_dir = os.path.join(out_root, "data")
    dataset_dir = os.path.join(out_root, "dataset")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(dataset_dir, exist_ok=True)

    valid_subf = []
    for subf in tqdm(sorted(os.listdir(raw_data_dir)), position=0):
        raw_subf_path = os.path.join(raw_data_dir, subf)
        if not os.path.isdir(raw_subf_path):
            continue

        missing = [name for name in REQUIRED_ARRAYS
                   if not os.path.exists(os.path.join(raw_subf_path, name))]
        if missing:
            raise FileNotFoundError(f"{raw_subf_path} missing {missing}")

        adj_path = os.path.join(raw_subf_path, "adj.npy")
        demos_path = os.path.join(raw_subf_path, "demos.npy")
        pois_path = os.path.join(raw_subf_path, "pois.npy")
        dis_path = os.path.join(raw_subf_path, "dis.npy")
        od_path = os.path.join(raw_subf_path, "od.npy")

        adj = np.load(adj_path)
        demos = np.load(demos_path)
        pois = np.load(pois_path)
        dis = np.load(dis_path)
        od = np.load(od_path)

        n_regions = adj.shape[0]
        assert adj.shape[1] == n_regions, "Adjacency matrix must be square."
        assert demos.shape[0] == n_regions, "Demos must match number of regions."
        assert pois.shape[0] == n_regions, "POIs must match number of regions."
        assert dis.shape[0] == n_regions and dis.shape[1] == n_regions, "Distance matrix must be square."
        assert od.shape[0] == n_regions and od.shape[1] == n_regions, "OD matrix must be square."
        assert demos.shape[1] == 97, "Demos must have 97 features."
        assert pois.shape[1] == 34, "POIs must have 34 features."
        assert np.all(od >= 0), "OD matrix must have non-negative values."
        assert np.all(dis >= 0), "Distance matrix must have non-negative values."
        assert (dis.T == dis).all(), "Distance matrix must be symmetric."
        assert np.max([dis[i][i] for i in range(n_regions)]) < 1e-5, f"Diagonal of distance matrix must be zero.\n{dis}"
        assert (adj.T == adj).all(), "Adjacency matrix must be symmetric."

        gdf = gpd.read_file(os.path.join(asset_dir, subf, f"{subf}.shp"))
        assert gdf.shape[0] == n_regions, "GeoDataFrame must match number of regions."
        assert gdf.geometry.is_valid.all(), "All geometries must be valid."

        gdf.sort_values(by="GEOID", inplace=True)
        gdf.reset_index(drop=True, inplace=True)
        gdf.to_crs(gdf.estimate_utm_crs(), inplace=True)
        
        centroids = np.array([[geom.x, geom.y] for geom in gdf.geometry.centroid])
        diff = centroids[:, None, :] - centroids[None, :, :]
        geo_distance = np.sqrt(np.sum(diff * diff, axis=-1))

        if not np.allclose(geo_distance, dis, rtol=0.01, atol=1):
            continue

        out_subf_path = os.path.join(data_dir, subf)
        os.makedirs(out_subf_path, exist_ok=True)
        for name in REQUIRED_ARRAYS:
            shutil.copy2(os.path.join(raw_subf_path, name), os.path.join(out_subf_path, name))

        centroids_path = os.path.join(out_subf_path, "centroids.npy")
        np.save(centroids_path, centroids)
        valid_subf.append(subf)

    valid_subf = sorted(valid_subf)
    rng = random.Random(0)
    rng.shuffle(valid_subf)
    n = len(valid_subf)
    n_train = int(n * 0.7)
    n_valid = int(n * 0.15)
    split_dir = os.path.join(dataset_dir, "seed0")
    os.makedirs(split_dir, exist_ok=True)
    for name, areas in [
        ("train", valid_subf[:n_train]),
        ("valid", valid_subf[n_train:n_train + n_valid]),
        ("test", valid_subf[n_train + n_valid:]),
    ]:
        with open(os.path.join(split_dir, f"{name}_subf.json"), "w") as f:
            json.dump(areas, f, indent=2)
    print(f"Full raw data: {len(os.listdir(raw_data_dir))}")
    print(f"Valid data: {len(valid_subf)}")


if __name__ == "__main__":
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-data-dir", default=os.path.join(repo_root, "raw", "commuting", "data"))
    parser.add_argument("--asset-dir", default=os.path.join(repo_root, "raw", "commuting", "assets", "Boundaries_Regions_within_Areas"))
    parser.add_argument("--out-root", default=os.path.join(repo_root, "datasets", "commuting"))
    args = parser.parse_args()
    preprocess(args.raw_data_dir, args.asset_dir, args.out_root)
    print("Preprocessing completed.")
