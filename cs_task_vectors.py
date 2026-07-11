"""Stage 3 (0609 weight steering): weight task vectors + similarity vs DPO generalization.

For each constraint we have two LoRA adapters (pos / neg) trained on the persona
completions. The weight steering vector is the contrastive task vector

    t = dW_pos - dW_neg ,    dW = (lora_alpha / r) * B @ A   (per target module)

Because both adapters sit on the same base and touch the same modules, the
non-target parameters cancel exactly, so we reconstruct the full-shape task
vector directly from the adapters -- no 250 GB of merged 7B models needed. We
wrap it in the weight-steering codebase's `TaskVector` and use its own
`cosine_similarity` to build the 9x9 matrix, then Spearman-correlate the
off-diagonal against the DPO generalization ground truth (mirrors
src/value_vectors/compute_similarity.py).

Usage (from repo root):
    python weight-steering/cs_task_vectors.py \
        --adapters_dir /data/tir/.../weight_steering \
        --adapter_tmpl 0609_olmo_{value}_{pol}_seed42 \
        --ground_truth output/ground_truth/dpo_generalization.npy \
        --ground_truth_values output/ground_truth/dpo_generalization_values.json \
        --output_dir output/ws_results
"""

import argparse
import json
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from scipy.stats import spearmanr

from task_vectors import TaskVector  # weight-steering codebase

CONSTRAINTS = [
    "no_affirm_delusions",
    "no_exploit_psych_weaknesses",
    "no_undermine_relationships",
    "no_lie_to_hide",
    "no_false_impressions",
    "no_white_lies",
    "no_discriminate",
    "no_stereotypes",
    "no_identity_attacks",
]

_PREFIX = "base_model.model."
_LORA_A = re.compile(r"^(.*?)\.lora_A(?:\.default)?\.weight$")
_LORA_B = re.compile(r"^(.*?)\.lora_B(?:\.default)?\.weight$")
# NOTE: in the peft/axolotl version that trained the 0609 OLMo-2 adapters,
# modules_to_save are NOT saved with a ".modules_to_save(.default).weight" infix
# (the original _MTS regex below assumed they were and silently matched nothing,
# dropping embed_tokens/lm_head from every task vector -- README risk #4). They
# are saved as plain full weights under their normal module path:
#     base_model.model.lm_head.weight
#     base_model.model.model.embed_tokens.weight
# So we identify them as any non-LoRA ".weight" key (see _load_adapter_deltas).
_MTS = re.compile(r"^(.*?)\.modules_to_save(?:\.default)?\.weight$")  # legacy, unused

# Base full-module weights needed to diff modules_to_save (embed_tokens, lm_head).
# Loaded once, lazily, the first time a modules_to_save key is seen.
_BASE_CACHE = {}


def _strip_prefix(key):
    return key[len(_PREFIX):] if key.startswith(_PREFIX) else key


def _get_base_weights(base_model_name, needed_keys, device):
    """Load base weights for the given state-dict keys (cached)."""
    missing = [k for k in needed_keys if k not in _BASE_CACHE]
    if missing:
        from transformers import AutoModelForCausalLM
        print(f"  Loading base {base_model_name} for modules_to_save diff "
              f"({len(missing)} module(s))...", flush=True)
        base = AutoModelForCausalLM.from_pretrained(
            base_model_name, torch_dtype=torch.float32
        )
        bsd = base.state_dict()
        for k in missing:
            if k not in bsd:
                raise KeyError(f"Base key {k} not in {base_model_name} state_dict")
            _BASE_CACHE[k] = bsd[k].float().cpu()
        del base, bsd
    return {k: _BASE_CACHE[k] for k in needed_keys}


def _load_adapter_deltas(adapter_dir, base_model_name, device, lora_only=False):
    """Return {module_key: dW} for one adapter.

    LoRA modules:        dW = (alpha / r) * B @ A   (no base needed)
    modules_to_save:     dW = trained_full - base_full   (embed_tokens, lm_head)

    If lora_only, the modules_to_save (embed_tokens/lm_head) deltas are dropped
    entirely -- isolates the transformer-weight mechanism from the embedding/
    token-frequency component, and skips the base-model load.
    """
    cfg = json.loads((Path(adapter_dir) / "adapter_config.json").read_text())
    scaling = cfg["lora_alpha"] / cfg["r"]

    sd_path = Path(adapter_dir) / "adapter_model.safetensors"
    sd = load_file(str(sd_path)) if sd_path.exists() else torch.load(
        Path(adapter_dir) / "adapter_model.bin", map_location="cpu"
    )

    deltas = {}
    # LoRA A/B -> dW = scaling * B @ A
    for key, A in sd.items():
        m = _LORA_A.match(key)
        if not m:
            continue
        module = _strip_prefix(m.group(1))
        B = sd[key.replace("lora_A", "lora_B")]
        dW = scaling * (B.to(device).float() @ A.to(device).float())
        deltas[module] = dW.cpu()

    # modules_to_save (embed_tokens, lm_head): saved as plain full weights under
    # their normal module path (no ".modules_to_save" infix in this peft version).
    # Any non-LoRA ".weight" key is one of these. dW = trained_full - base_full.
    mts = {}  # module_key -> (trained_tensor, base_state_dict_key)
    for key, W in sd.items():
        if lora_only or _LORA_A.match(key) or _LORA_B.match(key) or not key.endswith(".weight"):
            continue
        rel = _strip_prefix(key)            # e.g. "model.embed_tokens.weight" or "lm_head.weight"
        module = rel[: -len(".weight")]     # "model.embed_tokens" / "lm_head"
        mts[module] = (W, rel)              # base state_dict key == rel
    if mts:
        base = _get_base_weights(base_model_name, [bk for _, bk in mts.values()], device)
        for module, (W, base_key) in mts.items():
            deltas[module] = (W.float() - base[base_key]).cpu()

    if not deltas:
        raise ValueError(f"No LoRA or modules_to_save weights found in {sd_path}")
    return deltas


def build_task_vector(adapters_dir, tmpl, value, base_model_name, device, lora_only=False):
    """Contrastive weight task vector t = dW_pos - dW_neg, as a TaskVector."""
    pos_dir = Path(adapters_dir) / tmpl.format(value=value, pol="pos")
    neg_dir = Path(adapters_dir) / tmpl.format(value=value, pol="neg")
    pos = _load_adapter_deltas(pos_dir, base_model_name, device, lora_only)
    neg = _load_adapter_deltas(neg_dir, base_model_name, device, lora_only)
    if set(pos) != set(neg):
        raise ValueError(f"pos/neg module mismatch for {value}: "
                         f"{set(pos) ^ set(neg)}")
    vector = {k: pos[k] - neg[k] for k in pos}
    return TaskVector(vector=vector)


def _log_norm_breakdown(vector, label):
    """Report what share of the task-vector squared-norm comes from the embed/lm_head
    full-weight deltas vs the LoRA linear-block deltas (the modules_to_save norm can
    dominate the global cosine -- worth seeing once)."""
    emb_sq = lora_sq = 0.0
    for k, v in vector.items():
        s = float((v.float() ** 2).sum())
        if "embed_tokens" in k or "lm_head" in k:
            emb_sq += s
        else:
            lora_sq += s
    tot = emb_sq + lora_sq
    if tot > 0:
        print(f"  [norm breakdown @ {label}] LoRA blocks: {100*lora_sq/tot:.1f}%  |  "
              f"embed+lm_head: {100*emb_sq/tot:.1f}%  of task-vector sq-norm", flush=True)


def off_diagonal(matrix, symmetric):
    n = matrix.shape[0]
    idx = (list(combinations(range(n), 2)) if symmetric
           else [(i, j) for i in range(n) for j in range(n) if i != j])
    return np.array([matrix[i, j] for i, j in idx]), idx


def main():
    ap = argparse.ArgumentParser(description="Weight task vectors + DPO-generalization correlation")
    ap.add_argument("--adapters_dir", required=True, help="Dir containing the per-task adapter dirs")
    ap.add_argument("--adapter_tmpl", default="0609_olmo_{value}_{pol}_seed42",
                    help="Adapter dir name template with {value} and {pol}")
    ap.add_argument("--base_model", default="allenai/OLMo-2-1124-7B-SFT",
                    help="Base model (for diffing modules_to_save: embed_tokens, lm_head)")
    ap.add_argument("--ground_truth", default=None,
                    help="Optional square GT matrix (.npy) for an in-script "
                         "sanity correlation. Omit to just build + save the "
                         "cosine matrix (correlation belongs downstream).")
    ap.add_argument("--ground_truth_values", default=None)
    ap.add_argument("--values", nargs="+", default=CONSTRAINTS)
    ap.add_argument("--symmetric", action="store_true",
                    help="Use upper triangle only (default: all off-diagonal, matching the asymmetric DPO matrix)")
    ap.add_argument("--output_dir", default="output/ws_results")
    ap.add_argument("--lora_only", action="store_true",
                    help="Drop embed_tokens/lm_head (modules_to_save) deltas; cosine over "
                         "LoRA transformer-block deltas only. Isolates weight mechanism from "
                         "the embedding/token-frequency component (and skips the base load).")
    ap.add_argument("--cache_dir", default=None,
                    help="Dir of per-principle task-vector caches ({value}.pt). With "
                         "--build_index, build+save one vector here and exit (parallel array "
                         "build). With --from_cache, load all vectors from here instead of "
                         "rebuilding inside the n^2 cosine loop.")
    ap.add_argument("--build_index", type=int, default=None,
                    help="Array-build mode: build ONLY values[build_index], save to "
                         "--cache_dir/{value}.pt, and exit. One adapter pair per SLURM task.")
    ap.add_argument("--from_cache", action="store_true",
                    help="Gather mode: load each task vector once from --cache_dir and compute "
                         "the cosine matrix in-memory (no n^2 rebuild).")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    values = args.values
    n = len(values)

    # ---- Array-build mode: build ONE task vector and save it, then exit. ----
    # The n^2 cosine loop below otherwise rebuilds each dense (embed+lm_head) task
    # vector ~n times; instead a SLURM array builds the n vectors in parallel (one
    # adapter pair per task) and --from_cache gathers them cheaply.
    if args.build_index is not None:
        assert args.cache_dir, "--build_index requires --cache_dir"
        from safetensors.torch import save_file
        v = values[args.build_index]
        cache_dir = Path(args.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        tv = build_task_vector(args.adapters_dir, args.adapter_tmpl, v,
                               args.base_model, device, args.lora_only)
        _log_norm_breakdown(tv.vector, v)
        # Save as a safetensors file (CPU, contiguous) keyed by module. safetensors
        # supports lazy per-key loading, so the gather can stream ONE module across
        # all n vectors instead of holding 20 x ~25GB dense vectors in RAM at once.
        out = cache_dir / f"{v}.safetensors"
        save_file({k: t.cpu().contiguous().float() for k, t in tv.vector.items()},
                  str(out))
        print(f"Built + saved task vector for {v} -> {out}", flush=True)
        return

    # n x n cosine matrix.
    sim = np.eye(n)
    if args.from_cache:
        # Gather mode (shard-streaming): the dense task vectors are ~25GB each, so we
        # never hold them all. Instead, for each module key we load that one tensor
        # from all n cached safetensors files and accumulate the Gram matrix
        # G[i,j] += <t_i[m], t_j[m]> and norm_sq[i] += <t_i[m], t_i[m]>. cos =
        # G[i,j]/sqrt(norm_i*norm_j). Summing over modules == TaskVector.cosine_similarity,
        # just reordered, so the values are identical. Peak RAM ~= n * largest module
        # (embed/lm_head ~1.6GB -> ~33GB), not n * full-vector.
        from safetensors import safe_open
        assert args.cache_dir, "--from_cache requires --cache_dir"
        cache_dir = Path(args.cache_dir)
        files = {v: cache_dir / f"{v}.safetensors" for v in values}
        missing = [v for v in values if not files[v].exists()]
        if missing:
            raise FileNotFoundError(f"Missing cached task vectors in {cache_dir}: {missing}")
        handles = [safe_open(str(files[v]), framework="pt", device="cpu") for v in values]
        keys = list(handles[0].keys())
        for h, v in zip(handles[1:], values[1:]):
            if set(h.keys()) != set(keys):
                raise ValueError(f"Module-key mismatch between {values[0]} and {v}")
        G = np.zeros((n, n), dtype=np.float64)
        norm_sq = np.zeros(n, dtype=np.float64)
        emb_sq = lora_sq = 0.0  # for the row-0 norm-breakdown log
        for ki, key in enumerate(keys):
            dim = handles[0].get_slice(key).get_shape()
            flat = int(np.prod(dim))
            X = torch.empty((n, flat), dtype=torch.float32)
            for i, h in enumerate(handles):
                X[i] = h.get_tensor(key).reshape(-1)
            Gm = (X @ X.T).numpy()            # (n, n) Gram contribution for this module
            G += Gm
            norm_sq += np.diag(Gm)
            s0 = float(Gm[0, 0])
            if "embed_tokens" in key or "lm_head" in key:
                emb_sq += s0
            else:
                lora_sq += s0
            del X, Gm
            if (ki + 1) % 50 == 0 or ki + 1 == len(keys):
                print(f"  module {ki+1}/{len(keys)} accumulated", flush=True)
        tot = emb_sq + lora_sq
        if tot > 0:
            print(f"  [norm breakdown @ {values[0]}] LoRA blocks: {100*lora_sq/tot:.1f}%  |  "
                  f"embed+lm_head: {100*emb_sq/tot:.1f}%  of task-vector sq-norm", flush=True)
        for i in range(n):
            for j in range(i + 1, n):
                denom = (norm_sq[i] * norm_sq[j]) ** 0.5
                c = G[i, j] / denom if denom > 0 else 0.0
                sim[i, j] = sim[j, i] = c
        print(f"  cosine matrix complete ({n}x{n})", flush=True)
    else:
        # Inline build (original): build vec_i once (held), rebuild vec_j in the inner
        # loop to bound memory to ~2 vectors.
        for i in range(n):
            vi = build_task_vector(args.adapters_dir, args.adapter_tmpl, values[i],
                                   args.base_model, device, args.lora_only)
            if i == 0:
                _log_norm_breakdown(vi.vector, values[i])
            for j in range(i + 1, n):
                vj = build_task_vector(args.adapters_dir, args.adapter_tmpl, values[j],
                                       args.base_model, device, args.lora_only)
                c = vi.cosine_similarity(vj)
                sim[i, j] = sim[j, i] = c
                del vj
            del vi
            print(f"  cosine row {i+1}/{n} ({values[i]}) done", flush=True)

    np.save(out_dir / "similarity_matrix.npy", sim)
    # Save the name ordering so a downstream rectangular correlation (e.g. the
    # 13x21 AD-DPO grid in notebooks/0618/ad_training/src/predict_dpo.py) can reindex
    # this cosine matrix by principle without re-deriving the order.
    (out_dir / "similarity_values.json").write_text(json.dumps(values, indent=2))

    # Optional in-script correlation. Task-vector generation must not require
    # a GT matrix -- downstream correlation tooling reads similarity_matrix.npy
    # + similarity_values.json instead.
    if not args.ground_truth or not args.ground_truth_values:
        print(f"\nNo --ground_truth given; saved cosine matrix + values to {out_dir}")
        return

    # Ground truth, reordered to `values`. This branch is for a SQUARE GT whose
    # values file is a flat name list (the interim prompt-steering sanity target).
    # The real 13x21 AD-DPO target is rectangular with a {"rows":..,"cols":..} values
    # file -- that correlation is done by predict_dpo.py against similarity_matrix.npy,
    # so skip here rather than crash.
    gt = np.load(args.ground_truth)
    gt_values = json.loads(Path(args.ground_truth_values).read_text())
    if not isinstance(gt_values, list) or gt.ndim != 2 or gt.shape[0] != gt.shape[1]:
        print(f"\nGround truth at {args.ground_truth} is not a square/flat-values matrix "
              f"(shape={gt.shape}); skipping the in-script correlation. The rectangular "
              f"correlation is done by notebooks/0618/ad_training/src/predict_dpo.py using "
              f"{out_dir}/similarity_matrix.npy + similarity_values.json.")
        print(f"Saved cosine matrix + values to {out_dir}")
        return
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
        rho, pval = spearmanr(sim_flat, gt_flat)
        key = "symmetric" if sym else "all_offdiag"
        results[key] = {"spearman_rho": float(rho), "p_value": float(pval), "n_pairs": len(idx)}
        print(f"{key}: Spearman rho={rho:.4f}, p={pval:.4f} (n={len(idx)})")

    use = "symmetric" if args.symmetric else "all_offdiag"
    summary = {"values": values, "ground_truth": args.ground_truth,
               "reported": use, **results}
    (out_dir / "correlation.json").write_text(json.dumps(summary, indent=2))

    # Per-pair predictions for error analysis (upper triangle).
    import csv
    with open(out_dir / "pair_predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["value_i", "value_j", "predicted_sim", "gt_i_to_j", "gt_j_to_i"])
        for i, j in combinations(range(n), 2):
            w.writerow([values[i], values[j], float(sim[i, j]),
                        float(gt[i, j]), float(gt[j, i])])
    print(f"Saved results to {out_dir}")


if __name__ == "__main__":
    main()
