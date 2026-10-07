"""FAST factored cosine for AD weight-steering (deadline path).

The contrastive task vector for a module is t = dW_pos - dW_neg.
  LoRA modules:    dW = s * B @ A  (s = alpha/r).  So
                   t = s*(Bp@Ap - Bn@An) = B' @ A'  with
                   B' = s*[Bp | -Bn]  (out x 2r),  A' = [Ap ; An]  (2r x in).
                   <t_i, t_j>_F = trace((B'_i A'_i)^T B'_j A'_j)
                               = trace((B'_i^T B'_j)(A'_j A'_i^T))   -- all 2r x 2r, cheap.
  embed/lm_head:   t = (Wpos - base) - (Wneg - base) = Wpos - Wneg  (base cancels!),
                   so just the dense diff; dotted densely (only 2 such modules).

Reads only the adapters (~70GB, mostly embed/lm_head), never the ~584GB dense
reconstruction. Produces the SAME cosine matrix as cs_task_vectors.py.

  python weight-steering/cs_factored_similarity.py --adapters_dir ... \
      --output_dir output/ws_ad_results --values helpful ... [--validate_cache DIR]
"""
import argparse, json, re
from pathlib import Path
import numpy as np, torch
from safetensors import safe_open

PREFIX = "base_model.model."
LORA_A = re.compile(r"^(.*?)\.lora_A(?:\.default)?\.weight$")
LORA_B = re.compile(r"^(.*?)\.lora_B(?:\.default)?\.weight$")


def _strip(k):
    return k[len(PREFIX):] if k.startswith(PREFIX) else k


def load_factors(adapter_dir):
    """Return (scaling, {lora_module: (A,B)}, {mts_module: full_weight}) for one adapter."""
    cfg = json.loads((Path(adapter_dir) / "adapter_config.json").read_text())
    s = cfg["lora_alpha"] / cfg["r"]
    f = safe_open(str(Path(adapter_dir) / "adapter_model.safetensors"),
                  framework="pt", device="cpu")
    keys = list(f.keys())
    lora, mts = {}, {}
    for k in keys:
        m = LORA_A.match(k)
        if m:
            mod = _strip(m.group(1))
            A = f.get_tensor(k).float()
            B = f.get_tensor(k.replace("lora_A", "lora_B")).float()
            lora[mod] = (A, B)
    for k in keys:
        if LORA_A.match(k) or LORA_B.match(k) or not k.endswith(".weight"):
            continue
        mts[_strip(k)[: -len(".weight")]] = f.get_tensor(k).float()
    return s, lora, mts


def build_vec(adapters_dir, tmpl, value):
    """Factored task vector: {lora_mod: (Bp, Ap)} where Bp=s[Bpos|-Bneg], Ap=[Apos;Aneg];
    plus {mts_mod: dense diff}."""
    sp, lp, mp = load_factors(Path(adapters_dir) / tmpl.format(value=value, pol="pos"))
    sn, ln, mn = load_factors(Path(adapters_dir) / tmpl.format(value=value, pol="neg"))
    assert abs(sp - sn) < 1e-9 and set(lp) == set(ln) and set(mp) == set(mn)
    lora = {}
    for mod in lp:
        Ap, Bp = lp[mod]; An, Bn = ln[mod]
        Bcat = sp * torch.cat([Bp, -Bn], dim=1)      # (out, 2r)
        Acat = torch.cat([Ap, An], dim=0)            # (2r, in)
        lora[mod] = (Bcat, Acat)
    mts = {mod: (mp[mod] - mn[mod]) for mod in mp}    # base cancels
    return lora, mts


def dot_and_norms(vi, vj):
    """(<t_i,t_j>, ||t_i||^2, ||t_j||^2) summed over all modules."""
    li, mi = vi; lj, mj = vj
    dot = ni = nj = 0.0
    for mod in li:
        Bi, Ai = li[mod]; Bj, Aj = lj[mod]
        P = Bi.T @ Bj                # (2r,2r)
        Q = Aj @ Ai.T                # (2r,2r)
        dot += float((P * Q.T).sum())
        Pii = Bi.T @ Bi; Qii = Ai @ Ai.T; ni += float((Pii * Qii.T).sum())
        Pjj = Bj.T @ Bj; Qjj = Aj @ Aj.T; nj += float((Pjj * Qjj.T).sum())
    for mod in mi:
        di, dj = mi[mod], mj[mod]
        dot += float((di * dj).sum())
        ni += float((di * di).sum()); nj += float((dj * dj).sum())
    return dot, ni, nj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters_dir", required=True)
    ap.add_argument("--adapter_tmpl", default="0618_ws_ad_{value}_{pol}_seed42")
    ap.add_argument("--values", nargs="+", required=True)
    ap.add_argument("--output_dir", default="output/ws_ad_results")
    ap.add_argument("--validate_cache", default=None,
                    help="Dir of dense {value}.safetensors; if set, check factored cosine "
                         "matches the dense cache for the first 2 values before the full run.")
    args = ap.parse_args()
    values = args.values
    n = len(values)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    print(f"Loading {n} factored task vectors...", flush=True)
    vecs = []
    for v in values:
        vecs.append(build_vec(args.adapters_dir, args.adapter_tmpl, v))
        print(f"  loaded {v}", flush=True)

    # ---- optional correctness check vs the dense cache ----
    if args.validate_cache:
        cd = Path(args.validate_cache)
        def dense_cos(a, b):
            ha = safe_open(str(cd / f"{a}.safetensors"), framework="pt", device="cpu")
            hb = safe_open(str(cd / f"{b}.safetensors"), framework="pt", device="cpu")
            dot = na = nb = 0.0
            for k in ha.keys():
                ta = ha.get_tensor(k).float().reshape(-1)
                tb = hb.get_tensor(k).float().reshape(-1)
                dot += float(ta @ tb); na += float(ta @ ta); nb += float(tb @ tb)
            return dot / (na ** 0.5 * nb ** 0.5)
        for (a, b) in [(0, 1), (0, n // 2)]:
            d, ni, nj = dot_and_norms(vecs[a], vecs[b])
            fac = d / (ni ** 0.5 * nj ** 0.5)
            den = dense_cos(values[a], values[b])
            print(f"VALIDATE {values[a]} vs {values[b]}: factored={fac:.6f} dense={den:.6f} "
                  f"diff={abs(fac-den):.2e}", flush=True)
            assert abs(fac - den) < 1e-3, "FACTORED MISMATCH vs dense cache"
        print("FACTORED MATCHES DENSE CACHE ✓", flush=True)

    sim = np.eye(n)
    norms = [None] * n
    for i in range(n):
        di, ni, _ = dot_and_norms(vecs[i], vecs[i]); norms[i] = ni
    for i in range(n):
        for j in range(i + 1, n):
            d, _, _ = dot_and_norms(vecs[i], vecs[j])
            sim[i, j] = sim[j, i] = d / (norms[i] ** 0.5 * norms[j] ** 0.5)
        print(f"  row {i+1}/{n} done", flush=True)

    np.save(out / "similarity_matrix.npy", sim)
    (out / "similarity_values.json").write_text(json.dumps(values, indent=2))
    print(f"Saved {n}x{n} cosine matrix -> {out}/similarity_matrix.npy", flush=True)


if __name__ == "__main__":
    main()
