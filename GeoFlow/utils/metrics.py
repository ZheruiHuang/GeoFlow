import numpy as np
import torch
from collections import defaultdict


EPS = 1e-8

def _to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def cal_od_metrics(a, b, mask=None):
    """
    b is ground truth.
    """
    a = _to_numpy(a).astype(np.float64, copy=False)
    b = _to_numpy(b).astype(np.float64, copy=False)
    a = np.clip(a, 0.0, None)
    b = np.clip(b, 0.0, None)

    mask_bool = None
    if mask is not None:
        mask_np = _to_numpy(mask)
        mask_bool = (mask_np > 0)
        a_masked = a * mask_bool
        b_masked = b * mask_bool
        a_vals = a[mask_bool]
        b_vals = b[mask_bool]
    else:
        a_masked = a
        b_masked = b
        a_vals = a
        b_vals = b

    metrics = {
        "num_regions": num_regions(a, b),
        "RMSE": RMSE(a_vals, b_vals),
        "NRMSE": NRMSE(a_vals, b_vals),
        "MAE": MAE(a_vals, b_vals),
        "MAPE": MAPE(a_vals, b_vals),
        "SMAPE": SMAPE(a_vals, b_vals),
        "CPC": CPC(a_masked, b_masked),
        "NegCPC": -CPC(a_masked, b_masked),
        "accuracy": accuracy(a_vals, b_vals),
        "matrix_COS_similarity": matrix_COS_similarity(a_masked, b_masked),
        "JSD_inflow": JSD_inflow(a_masked, b_masked),
        "JSD_outflow": JSD_outflow(a_masked, b_masked),
        "JSD_ODflow": JSD_ODflow(a_masked, b_masked)
    }
    return metrics

def RMSE(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))

def NRMSE(a, b):
    denom = np.std(b)
    denom = max(denom, EPS)
    return float(RMSE(a, b) / denom)

def MAE(a, b):
    return float(np.mean(np.abs(a - b)))

def MAPE(a, b):
    denom = np.maximum(np.abs(b), EPS)
    return float(np.mean(np.abs(a - b) / denom))

def MSE(a, b):
    return float(np.mean((a - b) ** 2))

def SMAPE(a, b):
    denom = (np.abs(a) + np.abs(b)) / 2.0 + EPS
    return float(np.mean(np.abs(a - b) / denom))

def CPC(a, b):
    """
    2 * sum(min(a,b)) / (sum(a)+sum(b))
    """
    denom = a.sum() + b.sum()
    if denom <= EPS:
        return 0.0
    return float(2.0 * np.minimum(a, b).sum() / denom)

def accuracy(a, b):
    a_bin = (a > 0).astype(np.uint8)
    b_bin = (b > 0).astype(np.uint8)
    return float((a_bin == b_bin).sum() / a_bin.size)

def matrix_COS_similarity(a, b):
    # row-wise
    a_row_norm = np.sqrt((a ** 2).sum(0)) + EPS
    b_row_norm = np.sqrt((b ** 2).sum(0)) + EPS
    row_sim = (a * b).sum(0) / (a_row_norm * b_row_norm)

    # col-wise
    a_col_norm = np.sqrt((a ** 2).sum(1)) + EPS
    b_col_norm = np.sqrt((b ** 2).sum(1)) + EPS
    col_sim = (a * b).sum(1) / (a_col_norm * b_col_norm)

    return float((row_sim.mean() + col_sim.mean()) / 2.0)

def values_to_bucket(values):
    max_ = float(values.max())
    edges = [0.0, 1.0]
    x = 1.0
    while edges[-1] < max_:
        x *= 2.0
        edges.append(x)
    hist, _ = np.histogram(values, bins=edges)
    return edges, hist.tolist()

def JS_divergence(p, q):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p + EPS
    q = q + EPS
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * (np.log(p) - np.log(m)))
    kl_qm = np.sum(q * (np.log(q) - np.log(m)))
    jsd_nat = 0.5 * (kl_pm + kl_qm)
    return float(jsd_nat / np.log(2))

def JSD_in(a, b):
    a_in, b_in = a.sum(0), b.sum(0)
    sections, b_dist = values_to_bucket(b_in)
    a_hist, _ = np.histogram(a_in, bins=sections)
    return JS_divergence(a_hist, b_dist)

def JSD_out(a, b):
    a_out, b_out = a.sum(1), b.sum(1)
    sections, b_dist = values_to_bucket(b_out)
    a_hist, _ = np.histogram(a_out, bins=sections)
    return JS_divergence(a_hist, b_dist)

def JSD_indegree(a, b):
    return JSD_in(a, b)

def JSD_outdegree(a, b):
    return JSD_out(a, b)

def JSD_inflow(a, b):
    return JSD_in(a, b)

def JSD_outflow(a, b):
    return JSD_out(a, b)

def JSD_ODflow(a, b):
    a = a.reshape(-1)
    b = b.reshape(-1)
    sections, b_dist = values_to_bucket(b)
    a_hist, _ = np.histogram(a, bins=sections)
    return JS_divergence(a_hist, b_dist)

def false_negative_rate(a, b):
    a_bin = (a > 0)
    b_bin = (b > 0)
    denom = b_bin.sum()
    return float(((~a_bin) & b_bin).sum() / max(denom, 1))

def false_positive_rate(a, b):
    a_bin = (a > 0)
    b_bin = (b > 0)
    denom = (~b_bin).sum()
    return float((a_bin & (~b_bin)).sum() / max(denom, 1)) if denom > 0 else np.nan

def nonzero_flow_fraction(a, b):
    a_bin = (a > 0).sum() / a.size
    b_bin = (b > 0).sum() / b.size
    return float(np.abs(a_bin - b_bin) / max(b_bin, EPS))

def num_regions(a, b):
    return int(b.shape[0])

def average_listed_metrics(listed_metrics):
    sums = defaultdict(float)
    for d in listed_metrics:
        for k, v in d.items():
            sums[k] += float(v)
    n = max(len(listed_metrics), 1)
    return {k: v / n for k, v in sums.items()}
