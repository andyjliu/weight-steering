"""Recompute the weight-steering correlation against a different ground-truth
matrix, reusing the already-saved similarity_matrix.npy (no task-vector rebuild).

Mirrors the correlation/off_diagonal/pair_predictions logic in cs_task_vectors.py.

    python cs_recompute_correlation.py \
        --ground_truth output/ground_truth/dpo_generalization_likert_normalized.npy \
        --ground_truth_values output/ground_truth/dpo_generalization_likert_values.json \
        --output_dir output/ws_results_likert_normalized
"""
import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

CONSTRAINTS = [
    "no_affirm_delusions", "no_exploit_psych_weaknesses", "no_undermine_relationships",
    "no_lie_to_hide", "no_false_impressions", "no_white_lies",
    "no_discriminate", "no_stereotypes", "no_identity_attacks",
]


def off_diagonal(matrix, symmetric):
    n = matrix.shape[0]
    idx = (list(combinations(range(n), 2)) if symmetric
           else [(i, j) for i in range(n) for j in range(n) if i != j])
    return np.array([matrix[i, j] for i, j in idx]), idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--similarity", default="output/ws_results/similarity_matrix.npy")
    ap.add_argument("--ground_truth",
                    default="output/ground_truth/dpo_generalization_likert_normalized.npy")
    ap.add_argument("--ground_truth_values",
                    default="output/ground_truth/dpo_generalization_likert_values.json")
    ap.add_argument("--values", nargs="+", default=CONSTRAINTS)
    ap.add_argument("--symmetric", action="store_true")
    ap.add_argument("--output_dir", default="output/ws_results_likert_normalized")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    values = args.values
    n = len(values)

    sim = np.load(args.similarity)
    if sim.shape != (n, n):
        raise ValueError(f"similarity matrix {sim.shape} != ({n},{n})")
    np.save(out_dir / "similarity_matrix.npy", sim)

    gt = np.load(args.ground_truth)
    gt_values = json.loads(Path(args.ground_truth_values).read_text())
    gt_idx = {v: k for k, v in enumerate(gt_values)}
    missing = [v for v in values if v not in gt_idx]
    if missing:
        raise ValueError(f"Values missing from ground truth: {missing}")
    reorder = [gt_idx[v] for v in values]
    gt = gt[np.ix_(reorder, reorder)]

    results = {}
    for sym in (False, True):
        sim_flat, idx = off_diagonal(sim, sym)
        gt_flat, _ = off_diagonal(gt, sym)
        # Drop pairs where the GT is NaN (normalized matrices can have NaNs).
        ok = ~(np.isnan(sim_flat) | np.isnan(gt_flat))
        rho, pval = spearmanr(sim_flat[ok], gt_flat[ok])
        key = "symmetric" if sym else "all_offdiag"
        results[key] = {"spearman_rho": float(rho), "p_value": float(pval),
                        "n_pairs": int(ok.sum())}
        print(f"{key}: Spearman rho={rho:.4f}, p={pval:.4f} (n={int(ok.sum())})")

    use = "symmetric" if args.symmetric else "all_offdiag"
    summary = {"values": values, "ground_truth": args.ground_truth,
               "reported": use, **results}
    (out_dir / "correlation.json").write_text(json.dumps(summary, indent=2))

    with open(out_dir / "pair_predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["value_i", "value_j", "predicted_sim", "gt_i_to_j", "gt_j_to_i"])
        for i, j in combinations(range(n), 2):
            w.writerow([values[i], values[j], float(sim[i, j]),
                        float(gt[i, j]), float(gt[j, i])])
    print(f"Saved results to {out_dir}")


if __name__ == "__main__":
    main()
