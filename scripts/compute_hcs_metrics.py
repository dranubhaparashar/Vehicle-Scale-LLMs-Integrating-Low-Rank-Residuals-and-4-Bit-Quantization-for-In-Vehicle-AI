#!/usr/bin/env python3
# compute_hcs_metrics.py
# Aggregates HCS annotations and reports mean±std, 95% CI, and Fleiss' kappa per dimension.

import argparse, pandas as pd, numpy as np, json, math
from collections import defaultdict

DIMENSIONS = ["coherence","relevance","instruction_following","safety_compliance","fluency"]
WEIGHTS = {"coherence":0.40,"relevance":0.20,"instruction_following":0.20,"safety_compliance":0.10,"fluency":0.10}
CATEGORIES = [1,2,3,4,5]

def fleiss_kappa(counts: np.ndarray) -> float:
    """
    counts: matrix of shape (N_items, N_categories) with integer tallies per category per item.
    """
    if counts.size == 0:
        return float("nan")
    N, k = counts.shape
    n = counts.sum(axis=1)
    if not np.all(n == n[0]):
        # items must have same number of ratings for standard Fleiss' kappa
        # approximate fix: downsample to min n across items
        n0 = int(n.min())
        new_counts = []
        rng = np.random.default_rng(0)
        for i in range(N):
            reps = []
            for j, c in enumerate(counts[i]):
                reps += [j]*int(c)
            if len(reps) < n0:
                continue
            sel = rng.choice(reps, size=n0, replace=False)
            row = np.bincount(sel, minlength=k)
            new_counts.append(row)
        counts = np.vstack(new_counts) if new_counts else counts
        if counts.size == 0:
            return float("nan")
        N, k = counts.shape
        n = counts.sum(axis=1)
    n = n[0]
    P_i = ((counts*(counts-1)).sum(axis=1)) / (n*(n-1))
    p_j = counts.sum(axis=0) / (N*n)
    P_bar = P_i.mean()
    P_e = (p_j**2).sum()
    if P_e == 1.0:
        return 1.0
    return float((P_bar - P_e) / (1 - P_e))

def ci95(mean, std, n):
    if n <= 1:
        return (mean, mean)
    se = std / math.sqrt(n)
    z = 1.96
    return (mean - z*se, mean + z*se)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="annotations CSV")
    ap.add_argument("--out", default="hcs_report.json")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    for d in DIMENSIONS:
        if d not in df.columns:
            raise ValueError(f"Missing column: {d}")
    df["hcs"] = sum(WEIGHTS[d]*df[d] for d in DIMENSIONS)

    item_stats = {}
    for col in DIMENSIONS + ["hcs"]:
        grp = df.groupby("item_id")[col]
        item_stats[col] = {
            "mean_of_item_means": float(grp.mean().mean()),
            "std_of_item_means": float(grp.mean().std(ddof=1)),
            "mean_of_item_stds": float(grp.std(ddof=1).mean()),
        }

    global_stats = {}
    for col in DIMENSIONS + ["hcs"]:
        mean = df[col].mean()
        std = df[col].std(ddof=1)
        lo, hi = ci95(mean, std, len(df))
        global_stats[col] = {"mean": float(mean), "std": float(std), "ci95": [float(lo), float(hi)]}

    kappas = {}
    for dim in DIMENSIONS:
        mat = []
        for item_id, g in df.groupby("item_id"):
            counts = [(g[dim] == c).sum() for c in CATEGORIES]
            mat.append(counts)
        mat = np.asarray(mat, dtype=float)
        kappa = fleiss_kappa(mat)
        kappas[dim] = float(kappa)

    report = {
        "weights": WEIGHTS,
        "global": global_stats,
        "per_item_summary": item_stats,
        "fleiss_kappa": kappas,
        "n_items": int(df["item_id"].nunique()),
        "n_rows": int(len(df)),
        "n_annotators": int(df["annotator_id"].nunique()),
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
