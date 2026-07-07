"""Build the airtravel OD dataset."""
import argparse
import json
import os
import random

import numpy as np
import pandas as pd


def _load_dataframe(src_path):
    return pd.read_excel(src_path, sheet_name="Sheet 1")


def _cosine_distance(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na <= 0 or nb <= 0:
        return 1.0
    return 1.0 - float(a @ b) / (na * nb)


def _knn_adj(dist, k):
    n = dist.shape[0]
    k = min(k, n - 1)
    a = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        nbr = np.argsort(dist[i])[1 : k + 1]
        a[i, nbr] = 1
    a = np.maximum(a, a.T)
    np.fill_diagonal(a, 0)
    return a


def _cmds_coords(dist):
    """Classical MDS into 2-D from a square distance matrix (power iteration)."""
    n = int(dist.shape[0])
    if n == 0:
        return np.zeros((0, 2), dtype=np.float64)
    d2 = dist.astype(np.float64) ** 2
    rm, cm = d2.mean(axis=1), d2.mean(axis=0)
    tm = float(d2.mean())
    B = -0.5 * (d2 - rm[:, None] - cm[None, :] + tm)

    def power_iter(M, n_iter=200, tol=1e-8):
        rng = np.random.RandomState(0)
        v = rng.rand(n)
        v /= np.linalg.norm(v) or 1.0
        for _ in range(n_iter):
            w = M @ v
            nrm = np.linalg.norm(w)
            if nrm == 0:
                break
            w /= nrm
            if np.linalg.norm(w - v) < tol:
                v = w
                break
            v = w
        return float(v @ (M @ v)), v

    e1, v1 = power_iter(B)
    B2 = B - e1 * np.outer(v1, v1)
    e2, v2 = power_iter(B2)
    e1, e2 = max(e1, 0.0), max(e2, 0.0)
    return np.stack([v1 * np.sqrt(e1), v2 * np.sqrt(e2)], axis=1)


def _select_subset(items, rng, count=None, ratio=None, min_count=2):
    n = len(items)
    if count is None:
        count = max(min_count, int(round(n * (ratio if ratio is not None else 1.0))))
    count = min(max(count, min_count), n)
    if count >= n:
        return items
    idx = rng.choice(np.arange(n), size=count, replace=False)
    return [items[i] for i in idx]
def _build_area(sub, origin_nodes, dest_nodes, area_name, data_dir):
    nodes = origin_nodes + dest_nodes
    n = len(nodes)
    o_idx = {nd["code"]: i for i, nd in enumerate(origin_nodes)}
    d_idx = {nd["code"]: i + len(origin_nodes) for i, nd in enumerate(dest_nodes)}

    od = np.zeros((n, n), dtype=np.float64)
    grouped = (
        sub.groupby(["Origin_market_code", "Destination_code"])["Visitor_days"]
        .sum()
        .reset_index()
    )
    for o_code, d_code, days in grouped.itertuples(index=False):
        oi = o_idx.get(str(o_code))
        di = d_idx.get(str(d_code))
        if oi is not None and di is not None:
            od[oi, di] = float(days)

    out_flow = od.sum(axis=1)
    in_flow = od.sum(axis=0)
    out_deg = (od > 0).sum(axis=1).astype(np.float64)
    in_deg = (od > 0).sum(axis=0).astype(np.float64)
    total_flow = out_flow + in_flow
    area_total = float(total_flow.sum()) or 1.0
    safe_total = np.where(total_flow > 0, total_flow, 1.0)

    demos = np.stack([
        out_flow,                       # 0 out_flow
        in_flow,                        # 1 in_flow
        out_deg,                        # 2 out_degree
        in_deg,                         # 3 in_degree
        total_flow,                     # 4 total_flow
        out_flow / safe_total,          # 5 out_share
        in_flow / safe_total,           # 6 in_share
        total_flow / area_total,        # 7 total_share
    ], axis=1).astype(np.float64)

    is_origin = np.array([1.0 if nd["type"] == "origin" else 0.0 for nd in nodes],
                         dtype=np.float64)
    pois = np.stack([is_origin, 1.0 - is_origin], axis=1)

    node_vecs = [(od[i] if nodes[i]["type"] == "origin" else od[:, i]) for i in range(n)]
    dist = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            dist[i, j] = dist[j, i] = float(_cosine_distance(node_vecs[i], node_vecs[j]))

    adj = _knn_adj(dist, k=min(5, n - 1))
    centroids = _cmds_coords(dist.astype(np.float64))

    mask = np.zeros((n, n), dtype=np.float32)
    mask[: len(origin_nodes), len(origin_nodes) :] = 1.0  # origin->dest only

    ap = os.path.join(data_dir, area_name)
    os.makedirs(ap, exist_ok=True)
    np.save(os.path.join(ap, "adj.npy"), adj)
    np.save(os.path.join(ap, "centroids.npy"), centroids)
    np.save(os.path.join(ap, "demos.npy"), demos)
    np.save(os.path.join(ap, "pois.npy"), pois)
    np.save(os.path.join(ap, "dis.npy"), dist)
    np.save(os.path.join(ap, "od.npy"), od)
    np.save(os.path.join(ap, "mask.npy"), mask)
    return nodes
_RES_CODE = {"Monthly": "M", "Quarterly": "Q", "Seasonally": "S", "Yearly": "Y"}
_SEG_CODE = {"Domestic visitor": "DOM", "Total international visitor": "INT"}


def build(args):
    df = _load_dataframe(args.src)
    df["Date"] = pd.to_datetime(df["Date"])
    if args.temporal:
        df = df[df["Temporal_resolution"].isin(args.temporal.split(","))]
    if args.segment:
        df = df[df["Population_segment"].isin(args.segment.split(","))]

    origins = df[["Origin_market", "Origin_market_code"]].drop_duplicates(
        ).sort_values("Origin_market_code")
    dests = df[["Destination", "Destination_code"]].drop_duplicates(
        ).sort_values("Destination_code")
    origin_nodes_full = [{"type": "origin",
                          "code": str(r["Origin_market_code"]),
                          "name": str(r["Origin_market"])}
                         for _, r in origins.iterrows()]
    dest_nodes_full = [{"type": "destination",
                        "code": str(r["Destination_code"]),
                        "name": str(r["Destination"])}
                       for _, r in dests.iterrows()]

    data_dir = os.path.join(args.out_root, "data")
    dataset_dir = os.path.join(args.out_root, "dataset")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(dataset_dir, exist_ok=True)

    area_list, sizes = [], []
    rng = np.random.RandomState(args.seed)
    for (temporal, segment, date), sub in df.groupby(
            ["Temporal_resolution", "Population_segment", "Date"]):
        base_name = (f"{args.prefix}_{_RES_CODE.get(temporal, 'X')}"
                     f"_{_SEG_CODE.get(segment, 'X')}_{date.strftime('%Y%m%d')}")

        def _emit(area_name, o_nodes, d_nodes):
            nodes = _build_area(sub, o_nodes, d_nodes, area_name, data_dir)
            area_list.append(area_name)
            sizes.append(len(nodes))

        _emit(base_name, origin_nodes_full, dest_nodes_full)
        for s_idx in range(args.augment_subsamples):
            o_nodes = _select_subset(origin_nodes_full, rng,
                                     count=args.origin_sample_count,
                                     ratio=args.origin_sample_ratio, min_count=2)
            d_nodes = _select_subset(dest_nodes_full, rng,
                                     count=args.dest_sample_count,
                                     ratio=args.dest_sample_ratio, min_count=2)
            _emit(f"{base_name}_S{s_idx:02d}", o_nodes, d_nodes)

    # 70/15/15 splits for the requested seeds
    for seed in args.split_seeds:
        if args.split_shuffle_seed is None:
            shuffle_seed = 13 if seed == 0 else seed
        else:
            shuffle_seed = args.split_shuffle_seed
        rng2 = random.Random(shuffle_seed)
        a = area_list.copy()
        rng2.shuffle(a)
        n = len(a)
        n_tr, n_va = int(n * 0.70), int(n * 0.15)
        out = os.path.join(dataset_dir, f"seed{seed}")
        os.makedirs(out, exist_ok=True)
        for name, lst in [("train", a[:n_tr]),
                          ("valid", a[n_tr:n_tr + n_va]),
                          ("test", a[n_tr + n_va:])]:
            with open(os.path.join(out, f"{name}_subf.json"), "w") as f:
                json.dump(lst, f, indent=2)

    sizes = np.array(sizes)
    print(f"areas: {len(area_list)} | nodes min/mean/max = "
          f"{sizes.min()}/{sizes.mean():.1f}/{sizes.max()} | saved under {args.out_root}")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    repo_root = os.path.dirname(here)
    p.add_argument("--src", default=os.path.join(repo_root, "raw", "tourism",
                                                 "Flows_visitor_days_origin_fixed.xlsx"))
    p.add_argument("--out-root", default=os.path.join(repo_root, "datasets", "airtravel"),
                   help="Directory to populate with data/ and dataset/")
    p.add_argument("--temporal", default="Monthly,Quarterly,Seasonally,Yearly")
    p.add_argument("--segment",
                   default="Domestic visitor,Total international visitor")
    p.add_argument("--prefix", default="TOUR")
    p.add_argument("--seed", type=int, default=13,
                   help="RNG seed for area sub-sampling (NOT split seed)")
    p.add_argument("--augment-subsamples", type=int, default=21,
                   help="Per (resolution, segment, date) extra random subsamples; "
                        "default 21 creates the 1980-area dataset")
    p.add_argument("--origin-sample-ratio", type=float, default=0.7)
    p.add_argument("--dest-sample-ratio", type=float, default=0.7)
    p.add_argument("--origin-sample-count", type=int, default=None)
    p.add_argument("--dest-sample-count", type=int, default=None)
    p.add_argument("--split-seeds", type=int, nargs="+", default=[0])
    p.add_argument("--split-shuffle-seed", type=int, default=None,
                   help="RNG seed used to shuffle split lists; by default seed0 uses 13 to mirror the reference split")
    build(p.parse_args())


if __name__ == "__main__":
    main()
