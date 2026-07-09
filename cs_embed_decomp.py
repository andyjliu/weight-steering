"""One-off: decompose the CS weight task vector's embedding component into
embed_tokens-only vs lm_head-only, and compare to the sentence-embedding baseline.

The task vector is t = dW_pos - dW_neg. For the modules_to_save full weights,
dW = W_trained - W_base, so the base CANCELS: t = W_pos - W_neg. Hence we can
build embed/lm_head task vectors straight from the two adapters (no base load).

All correlations vs dpo_resplit_avg_likert_normalized (off-diagonal Spearman).
"""
import os, json, numpy as np, torch
from safetensors import safe_open
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity

AD = "/data/tir/projects/tir3/users/andyliu/conflictscope-finetune/weight_steering"
TMPL = "0610_olmo_cs_{value}_{pol}_seed42"
EMBED = "base_model.model.model.embed_tokens.weight"
LMH = "base_model.model.lm_head.weight"
CONSTRAINTS = ["no_affirm_delusions", "no_exploit_psych_weaknesses", "no_undermine_relationships",
               "no_lie_to_hide", "no_false_impressions", "no_white_lies",
               "no_discriminate", "no_stereotypes", "no_identity_attacks"]
GT_DIR = "output/ground_truth"

gt = np.load(os.path.join(GT_DIR, "dpo_resplit_avg_likert_normalized.npy"))
assert json.load(open(os.path.join(GT_DIR, "dpo_resplit_avg_likert_values.json"))) == CONSTRAINTS
mask = ~np.eye(9, dtype=bool)
g = gt[mask]


def load(value, pol, key):
    p = os.path.join(AD, TMPL.format(value=value, pol=pol), "adapter_model.safetensors")
    with safe_open(p, "pt") as f:
        return f.get_tensor(key).float()


# task vectors t = W_pos - W_neg, stored fp16 to bound memory
tv_embed, tv_lmh = {}, {}
for c in CONSTRAINTS:
    tv_embed[c] = (load(c, "pos", EMBED) - load(c, "neg", EMBED)).flatten().half().numpy()
    tv_lmh[c] = (load(c, "pos", LMH) - load(c, "neg", LMH)).flatten().half().numpy()
    print("built", c, flush=True)


def corr(vecs, label):
    M = np.stack([vecs[c].astype(np.float32) for c in CONSTRAINTS])
    S = cosine_similarity(M)
    r = stats.spearmanr(S[mask], g)
    print(f"{label:<22} rho={r.correlation:+.4f}  p={r.pvalue:.4f}")
    return S


corr(tv_embed, "embed_tokens-only")
corr(tv_lmh, "lm_head-only")
corr({c: np.concatenate([tv_embed[c], tv_lmh[c]]) for c in CONSTRAINTS}, "embed+lm_head")

ne = np.mean([np.sum(tv_embed[c].astype(np.float64) ** 2) for c in CONSTRAINTS])
nl = np.mean([np.sum(tv_lmh[c].astype(np.float64) ** 2) for c in CONSTRAINTS])
print(f"\nmean sq-norm  embed={ne:.1f}  lm_head={nl:.1f}  (embed share {100*ne/(ne+nl):.1f}%)\n")

for method in ("default", "conflictscope"):
    d = f"persona_vectors/persona_vectors/all-mpnet-base-v2/{method}"
    try:
        vecs = {c: torch.load(os.path.join(d, f"{c}_response_avg_diff.pt"),
                              weights_only=False).float().numpy() for c in CONSTRAINTS}
        corr(vecs, f"sen-embd {method}")
    except Exception as e:
        print(f"sen-embd {method}: {e}")
