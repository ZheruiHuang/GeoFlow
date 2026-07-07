"""Build the FAF freight dataset."""
import argparse
import json
import os
import random

import numpy as np

from faf_utils import (
    _aggregate_flows,
    _build_features,
    _build_neighbors,
    _build_pair_dist,
    _cmds_coords,
    _knn_adj,
    _read_band_midpoints,
    _sample_nodes,
)


def _sample_size(rng, lo, hi, alpha, beta):
    u = rng.beta(alpha, beta)
    k = int(round(lo + u * (hi - lo)))
    return max(lo, min(hi, k))


def rebuild_features(out_root, area_zones):
    data_dir = os.path.join(out_root, "data")

    zone_base = {}
    for area, zones in sorted(area_zones.items()):
        area_dir = os.path.join(data_dir, area)
        demos = np.load(os.path.join(area_dir, "demos.npy"))
        for idx, zone in enumerate(zones):
            if zone not in zone_base:
                zone_base[zone] = demos[idx, :11].astype(np.float64).copy()

    zone_ids = np.array(sorted(zone_base.keys()), dtype=np.int64)
    n_zones = len(zone_ids)
    base = np.stack([zone_base[int(zone)] for zone in zone_ids], axis=0)
    global_max = base.max(axis=0)
    global_max_safe = np.where(global_max == 0, 1.0, global_max)

    base_norm = base / global_max_safe
    log_norm = np.log1p(np.clip(base, 0, None)) / np.log1p(global_max_safe)
    rank_norm = (base.argsort(axis=0).argsort(axis=0).astype(np.float64) + 1) / n_zones

    out_ton, in_ton = base[:, 0], base[:, 1]
    out_val, in_val = base[:, 2], base[:, 3]
    out_tmi, in_tmi = base[:, 4], base[:, 5]
    out_partners, in_partners = base[:, 6], base[:, 7]
    sum_ton, sum_val, sum_tmi = base[:, 8], base[:, 9], base[:, 10]
    eps = 1e-6
    ratios = np.stack([
        np.tanh(np.log((out_ton + eps) / (in_ton + eps))),
        np.tanh(np.log((out_val + eps) / (in_val + eps))),
        np.tanh(np.log((sum_val + eps) / (sum_ton + eps))),
        np.tanh(np.log((sum_tmi + eps) / (sum_ton + eps))),
        np.tanh(np.log((out_partners + eps) / (in_partners + eps))),
    ], axis=1)

    def log_norm_safe(values, denom):
        return np.log1p(np.clip(values, 0, None)) / max(np.log1p(denom), 1.0)

    max_partner_out = (out_ton / np.clip(out_partners, 1, None)).max()
    max_partner_in = (in_ton / np.clip(in_partners, 1, None)).max()
    intensity = np.stack([
        log_norm_safe(out_ton / np.clip(out_partners, 1, None), max_partner_out),
        log_norm_safe(in_ton / np.clip(in_partners, 1, None), max_partner_in),
        log_norm_safe(out_partners + in_partners, n_zones),
    ], axis=1)

    freqs = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.float64)
    max_id = float(zone_ids.max())
    phases = (zone_ids[:, None].astype(np.float64) / max_id) * 2 * np.pi * freqs[None, :]
    sincos = np.concatenate([np.sin(phases), np.cos(phases)], axis=1)

    global_table = np.concatenate(
        [base_norm, log_norm, rank_norm, ratios, intensity, sincos], axis=1
    ).astype(np.float64)
    zone_to_idx = {int(zone): idx for idx, zone in enumerate(zone_ids)}

    all_means, all_mins, all_maxs = [], [], []
    for area in sorted(os.listdir(data_dir)):
        dis_path = os.path.join(data_dir, area, "dis.npy")
        if not os.path.exists(dis_path):
            continue
        dis = np.load(dis_path).astype(np.float64)
        if dis.shape[0] < 2:
            continue
        off = dis.copy()
        np.fill_diagonal(off, np.nan)
        all_means.append(np.nanmean(off, axis=1))
        all_mins.append(np.nanmin(off, axis=1))
        all_maxs.append(np.nanmax(off, axis=1))
    mean_scale = np.percentile(np.concatenate(all_means), 99) or 1.0
    min_scale = np.percentile(np.concatenate(all_mins), 99) or 1.0
    max_scale = np.percentile(np.concatenate(all_maxs), 99) or 1.0

    max_nodes = 132
    log_max_nodes = np.log1p(max_nodes)
    n_done = 0
    for area, zones in sorted(area_zones.items()):
        area_dir = os.path.join(data_dir, area)
        idx = np.array([zone_to_idx[zone] for zone in zones], dtype=np.int64)
        demos_new = global_table[idx].astype(np.float64)

        n = len(zones)
        sum_ton_area = sum_ton[idx]
        rank_area = (sum_ton_area.argsort().argsort().astype(np.float64) + 1) / n
        frac_area = sum_ton_area / max(sum_ton_area.sum(), 1e-9)
        dis = np.load(os.path.join(area_dir, "dis.npy")).astype(np.float64)
        if n >= 2:
            off = dis.copy()
            np.fill_diagonal(off, np.nan)
            dist_mean = np.nanmean(off, axis=1)
            dist_min = np.nanmin(off, axis=1)
            dist_max = np.nanmax(off, axis=1)
        else:
            dist_mean = np.zeros(n)
            dist_min = np.zeros(n)
            dist_max = np.zeros(n)

        pois_new = np.stack([
            np.full(n, np.log1p(n) / log_max_nodes),
            rank_area,
            np.clip(frac_area, 0, 1),
            np.clip(dist_mean / mean_scale, 0, 2),
            np.clip(dist_min / min_scale, 0, 2),
            np.clip(dist_max / max_scale, 0, 2),
        ], axis=1).astype(np.float64)

        np.save(os.path.join(area_dir, "demos.npy"), demos_new)
        np.save(os.path.join(area_dir, "pois.npy"), pois_new)
        n_done += 1

    print(
        f"Rebuilt FAF features for {n_done} areas: "
        f"demos={global_table.shape[1]}, pois=6",
        flush=True,
    )


def build_dataset(
    csv_path,
    metadata_path,
    out_root,
    year,
    num_areas,
    min_nodes,
    max_nodes,
    seed,
    prefix,
    beta_a,
    beta_b,
):
    band_mid = _read_band_midpoints(metadata_path)
    flows, out_totals, in_totals, out_partners, in_partners = _aggregate_flows(
        csv_path, year, band_mid
    )
    pair_dist, default_dist_miles = _build_pair_dist(flows)
    neighbors = _build_neighbors(flows)

    zones = sorted(set(list(out_totals.keys()) + list(in_totals.keys())))
    zone_weights = [
        float(out_totals[z][0] + in_totals[z][0]) + 1e-6 for z in zones
    ]

    data_dir = os.path.join(out_root, "data")
    dataset_dir = os.path.join(out_root, "dataset")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(dataset_dir, exist_ok=True)

    rng = np.random.RandomState(seed)
    area_list = []
    area_zones = {}

    for idx in range(num_areas):
        if idx and idx % 500 == 0:
            print(f"built FAF areas: {idx}/{num_areas}", flush=True)
        if idx == 0:
            k = min_nodes
        elif idx == 1:
            k = max_nodes
        else:
            k = _sample_size(rng, min_nodes, max_nodes, beta_a, beta_b)
        nodes = _sample_nodes(rng, zones, zone_weights, neighbors, k)
        if len(nodes) < min_nodes:
            extra = rng.choice(
                np.array([z for z in zones if z not in nodes], dtype=int),
                size=min_nodes - len(nodes),
                replace=False,
            ).tolist()
            nodes = nodes + extra
        nodes = nodes[:k]

        n = len(nodes)
        dis_miles = np.full((n, n), np.nan, dtype=np.float64)
        for i, o in enumerate(nodes):
            for j, d in enumerate(nodes):
                if o == d:
                    dis_miles[i, j] = 0.0
                    continue
                dist = pair_dist.get((o, d))
                if dist is not None:
                    dis_miles[i, j] = dist
        for i in range(n):
            for j in range(i + 1, n):
                if np.isnan(dis_miles[i, j]) and not np.isnan(dis_miles[j, i]):
                    dis_miles[i, j] = dis_miles[j, i]
                if np.isnan(dis_miles[j, i]) and not np.isnan(dis_miles[i, j]):
                    dis_miles[j, i] = dis_miles[i, j]
        dis_miles[np.isnan(dis_miles)] = default_dist_miles
        dis_miles = (dis_miles + dis_miles.T) / 2.0
        np.fill_diagonal(dis_miles, 0.0)
        dis = (dis_miles * 1609.34).astype(np.float32)

        centroids = _cmds_coords(dis.astype(np.float64))
        adj = _knn_adj(dis, k=min(3, n - 1))

        od = np.zeros((n, n), dtype=np.float64)
        for i, o in enumerate(nodes):
            for j, d in enumerate(nodes):
                if o == d:
                    continue
                flow = flows.get((o, d))
                if flow is None:
                    continue
                od[i, j] = flow[0] / 1000.0

        demos = np.zeros((n, 97), dtype=np.float64)
        pois = np.zeros((n, 34), dtype=np.float64)
        for i, z in enumerate(nodes):
            out_ton, out_val, out_tmi = out_totals[z]
            in_ton, in_val, in_tmi = in_totals[z]
            base = np.array(
                [
                    out_ton / 1000.0,
                    in_ton / 1000.0,
                    out_val / 1000.0,
                    in_val / 1000.0,
                    out_tmi / 1000.0,
                    in_tmi / 1000.0,
                    float(len(out_partners[z])),
                    float(len(in_partners[z])),
                    (out_ton + in_ton) / 1000.0,
                    (out_val + in_val) / 1000.0,
                    (out_tmi + in_tmi) / 1000.0,
                ],
                dtype=np.float64,
            )
            demos[i] = _build_features(base, 97)
            pois[i] = _build_features(base, 34)

        area_name = f"{prefix}{idx:05d}"
        area_path = os.path.join(data_dir, area_name)
        os.makedirs(area_path, exist_ok=True)
        np.save(os.path.join(area_path, "adj.npy"), adj.astype(np.int32))
        np.save(os.path.join(area_path, "demos.npy"), demos)
        np.save(os.path.join(area_path, "pois.npy"), pois)
        np.save(os.path.join(area_path, "dis.npy"), dis)
        np.save(os.path.join(area_path, "od.npy"), od)
        np.save(os.path.join(area_path, "centroids.npy"), centroids)

        area_list.append(area_name)
        area_zones[area_name] = [int(z) for z in nodes]

    return area_list, area_zones


def _write_splits(area_list, dataset_dir, seed=0):
    rng = random.Random(seed)
    areas = area_list.copy()
    rng.shuffle(areas)
    total_count = len(areas)
    n_train = int(total_count * 0.7)
    n_val = int(total_count * 0.15)
    train_areas = areas[:n_train]
    val_areas = areas[n_train : n_train + n_val]
    test_areas = areas[n_train + n_val :]

    _write_split_lists(dataset_dir, seed, train_areas, val_areas, test_areas)


def _write_split_lists(dataset_dir, seed, train_areas, val_areas, test_areas):
    seed_dir = os.path.join(dataset_dir, f"seed{seed}")
    os.makedirs(seed_dir, exist_ok=True)
    with open(os.path.join(seed_dir, "train_subf.json"), "w") as handle:
        json.dump(train_areas, handle, indent=2)
    with open(os.path.join(seed_dir, "valid_subf.json"), "w") as handle:
        json.dump(val_areas, handle, indent=2)
    with open(os.path.join(seed_dir, "test_subf.json"), "w") as handle:
        json.dump(test_areas, handle, indent=2)


def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--faf-csv",
        default=os.path.join(repo_root, "raw", "FAF", "FAF5.7.1_2018-2024.zip"),
    )
    parser.add_argument(
        "--metadata",
        default=os.path.join(repo_root, "raw", "FAF", "FAF5_metadata.xlsx"),
    )
    parser.add_argument("--out-root", default=os.path.join(repo_root, "datasets", "faf"))
    parser.add_argument("--year", type=int, default=2022)
    parser.add_argument("--num-areas", type=int, default=5000)
    parser.add_argument("--min-nodes", type=int, default=10)
    parser.add_argument("--max-nodes", type=int, default=132)
    parser.add_argument("--beta-a", type=float, default=2.0)
    parser.add_argument("--beta-b", type=float, default=3.21)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--prefix", default="FAF_R")
    parser.add_argument("--split-seed", type=int, default=0)
    args = parser.parse_args()

    area_list, area_zones = build_dataset(
        csv_path=args.faf_csv,
        metadata_path=args.metadata,
        out_root=args.out_root,
        year=args.year,
        num_areas=args.num_areas,
        min_nodes=args.min_nodes,
        max_nodes=args.max_nodes,
        seed=args.seed,
        prefix=args.prefix,
        beta_a=args.beta_a,
        beta_b=args.beta_b,
    )

    dataset_dir = os.path.join(args.out_root, "dataset")
    rebuild_features(args.out_root, area_zones)
    _write_splits(area_list, dataset_dir, seed=args.split_seed)

    sizes = np.array([len(zones) for zones in area_zones.values()])
    print(
        f"areas: {len(area_list)} | zones min/mean/max = "
        f"{sizes.min()}/{sizes.mean():.1f}/{sizes.max()} | saved under {args.out_root}"
    )


if __name__ == "__main__":
    main()
