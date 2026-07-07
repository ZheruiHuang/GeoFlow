import zipfile
from collections import defaultdict

import numpy as np
import pandas as pd


def _parse_band_midpoint(desc):
    text = str(desc).replace(",", "").strip()
    if not text or text == "nan":
        return None
    low = text.lower()
    if low.startswith("below"):
        hi = float(text.split()[-1])
        return hi / 2.0
    if low.startswith("over"):
        lo = float(text.split()[-1])
        return lo + 250.0
    if "-" in text:
        parts = [p.strip() for p in text.split("-")]
        lo = float(parts[0])
        hi = float(parts[1])
        return (lo + hi) / 2.0
    return None


def _read_band_midpoints(metadata_path):
    df = pd.read_excel(metadata_path, sheet_name="Distance Band")
    df = df.dropna(subset=["Numeric Label"])
    mapping = {}
    for _, row in df.iterrows():
        label = int(row["Numeric Label"])
        midpoint = _parse_band_midpoint(row.get("Description", ""))
        if midpoint is not None:
            mapping[label] = float(midpoint)
    return mapping


def _iter_csv_chunks(csv_path, usecols, chunksize):
    if zipfile.is_zipfile(csv_path):
        with zipfile.ZipFile(csv_path) as archive:
            csv_members = [name for name in archive.namelist()
                           if name.lower().endswith(".csv")]
            if len(csv_members) != 1:
                raise ValueError(
                    f"Expected exactly one CSV in {csv_path}, found {csv_members}"
                )
            with archive.open(csv_members[0]) as handle:
                yield from pd.read_csv(handle, usecols=usecols, chunksize=chunksize)
        return
    yield from pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize)


def _aggregate_flows(csv_path, year, band_mid_map):
    ton_col = f"tons_{year}"
    val_col = f"value_{year}"
    tmi_col = f"tmiles_{year}"
    usecols = [
        "dms_orig",
        "dms_dest",
        "trade_type",
        "dist_band",
        ton_col,
        val_col,
        tmi_col,
    ]

    flows = defaultdict(lambda: np.zeros(5, dtype=np.float64))
    out_totals = defaultdict(lambda: np.zeros(3, dtype=np.float64))
    in_totals = defaultdict(lambda: np.zeros(3, dtype=np.float64))
    out_partners = defaultdict(set)
    in_partners = defaultdict(set)

    for chunk in _iter_csv_chunks(csv_path, usecols=usecols, chunksize=200000):
        chunk = chunk[chunk["trade_type"] == 1]
        chunk = chunk.dropna(
            subset=["dms_orig", "dms_dest", "dist_band", ton_col, val_col, tmi_col]
        )
        if chunk.empty:
            continue
        chunk["dms_orig"] = chunk["dms_orig"].astype(int)
        chunk["dms_dest"] = chunk["dms_dest"].astype(int)
        chunk["dist_band"] = chunk["dist_band"].astype(int)
        grouped = (
            chunk.groupby(["dms_orig", "dms_dest", "dist_band"])[
                [ton_col, val_col, tmi_col]
            ]
            .sum()
            .reset_index()
        )
        for row in grouped.itertuples(index=False):
            o = int(row[0])
            d = int(row[1])
            band = int(row[2])
            ton = float(row[3])
            val = float(row[4])
            tmi = float(row[5])
            band_mid = band_mid_map.get(band)
            if band_mid is None:
                continue
            weight = ton if ton > 0 else 1.0
            flows[(o, d)][0] += ton
            flows[(o, d)][1] += val
            flows[(o, d)][2] += tmi
            flows[(o, d)][3] += band_mid * weight
            flows[(o, d)][4] += weight
            out_totals[o] += np.array([ton, val, tmi], dtype=np.float64)
            in_totals[d] += np.array([ton, val, tmi], dtype=np.float64)
            out_partners[o].add(d)
            in_partners[d].add(o)

    return flows, out_totals, in_totals, out_partners, in_partners


def _build_features(base, out_dim):
    base = np.asarray(base, dtype=np.float64)
    feats = [
        base,
        np.log1p(base),
        np.sqrt(base),
    ]
    maxv = base.max() if base.max() > 0 else 1.0
    sumv = base.sum() if base.sum() > 0 else 1.0
    feats.append(base / maxv)
    feats.append(base / sumv)
    flat = np.concatenate(feats)
    if flat.size < out_dim:
        rep = np.tile(base, int(np.ceil((out_dim - flat.size) / base.size)))
        flat = np.concatenate([flat, rep[: out_dim - flat.size]])
    else:
        flat = flat[:out_dim]
    return flat


def _knn_adj(dist, k):
    n = dist.shape[0]
    k = min(k, n - 1)
    adj = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        order = np.argsort(dist[i])
        neighbors = order[1 : k + 1]
        adj[i, neighbors] = 1
    adj = np.maximum(adj, adj.T)
    np.fill_diagonal(adj, 0)
    return adj


def _cmds_coords(dist):
    n = int(dist.shape[0])
    if n == 0:
        return np.zeros((0, 2), dtype=np.float64)

    d2 = dist.astype(np.float64) ** 2
    row_mean = d2.mean(axis=1, keepdims=True)
    col_mean = d2.mean(axis=0, keepdims=True)
    total_mean = float(d2.mean())
    b = -0.5 * (d2 - row_mean - col_mean + total_mean)

    def power_iter(mat, n_iter=200, tol=1e-8):
        rng = np.random.RandomState(0)
        v = rng.rand(n).astype(np.float64)
        v_norm = float(np.linalg.norm(v))
        if v_norm == 0.0:
            v = np.zeros(n, dtype=np.float64)
            v[0] = 1.0
            v_norm = 1.0
        v = v / v_norm
        for _ in range(n_iter):
            w = mat @ v
            w_norm = float(np.linalg.norm(w))
            if w_norm == 0.0:
                break
            w = w / w_norm
            diff = float(np.linalg.norm(w - v))
            v = w
            if diff < tol:
                break
        eig = float(v @ (mat @ v))
        return eig, v

    eig1, v1 = power_iter(b)
    b2 = b - eig1 * np.outer(v1, v1)
    eig2, v2 = power_iter(b2)

    eig1 = max(float(eig1), 0.0)
    eig2 = max(float(eig2), 0.0)
    coords = np.zeros((n, 2), dtype=np.float64)
    coords[:, 0] = v1 * np.sqrt(eig1)
    coords[:, 1] = v2 * np.sqrt(eig2)
    return coords


def _build_pair_dist(flows):
    pair_dist = {}
    values = []
    for key, val in flows.items():
        if val[4] <= 0:
            continue
        d = float(val[3] / val[4])
        pair_dist[key] = d
        if d > 0:
            values.append(d)
    values.sort()
    if values:
        mid = len(values) // 2
        if len(values) % 2 == 1:
            median = values[mid]
        else:
            median = 0.5 * (values[mid - 1] + values[mid])
    else:
        median = 1500.0
    return pair_dist, float(median)


def _build_neighbors(flows):
    neighbors = defaultdict(list)
    for (o, d), val in flows.items():
        ton = float(val[0])
        if ton <= 0:
            continue
        neighbors[o].append((d, ton))
    for o, items in neighbors.items():
        items.sort(key=lambda kv: kv[1], reverse=True)
    return neighbors


def _sample_nodes(rng, zones, zone_weights, neighbors, k):
    nodes = []
    zones_arr = np.array(zones, dtype=int)
    weights = np.array(zone_weights, dtype=np.float64)
    weights = weights / weights.sum()
    start = int(rng.choice(zones_arr, p=weights))
    nodes.append(start)
    attempts = 0
    while len(nodes) < k and attempts < k * 20:
        attempts += 1
        base = int(rng.choice(nodes))
        cand = [d for d, _ in neighbors.get(base, []) if d not in nodes]
        if cand:
            cand_weights = np.array(
                [w for d, w in neighbors[base] if d not in nodes], dtype=np.float64
            )
            cand_weights = cand_weights / cand_weights.sum()
            pick = int(rng.choice(np.array(cand, dtype=int), p=cand_weights))
            nodes.append(pick)
            continue
        remaining = [z for z in zones if z not in nodes]
        if not remaining:
            break
        nodes.append(int(rng.choice(np.array(remaining, dtype=int))))
    return nodes
