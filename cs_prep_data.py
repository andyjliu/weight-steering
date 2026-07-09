"""Stage 1 (0609 weight steering): build axolotl SFT data from persona completions.

Uses the SAME quality-filtered pos/neg completions the persona-vector method uses
(src/value_vectors/data/persona_extract/{value}_{pos,neg}_instruct.csv) -- NOT any
ConflictScope data. We strip the steering system prompt and train on bare
(question -> answer) pairs per polarity, so the constraint behaviour is baked into
the weights and the resulting weight-space task vector is comparable to the DPO
weight deltas.

Writes axolotl `chat_template` JSONL:
    {out_dir}/{value}_pos.jsonl   # question -> constraint-upholding answer
    {out_dir}/{value}_neg.jsonl   # question -> constraint-violating answer

Only needs pandas, so it runs in any env.

Usage (from repo root):
    python weight-steering/cs_prep_data.py --out_dir data/0611/ws_sft_data
"""

import argparse
import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
PERSONA_EXTRACT = REPO_ROOT / "src" / "value_vectors" / "data" / "persona_extract"

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


def effective_rows(value, threshold, extract_dir=PERSONA_EXTRACT):
    """Replicates persona_vectors.generate_vec.get_persona_effective filtering."""
    pos = pd.read_csv(Path(extract_dir) / f"{value}_pos_instruct.csv")
    neg = pd.read_csv(Path(extract_dir) / f"{value}_neg_instruct.csv")
    valid = (
        pos["prompt"].notna() & pos["answer"].notna()
        & neg["prompt"].notna() & neg["answer"].notna()
    )
    mask = (
        valid
        & (pos[value] >= threshold)
        & (neg[value] < 100 - threshold)
        & (pos["coherence"] >= 50)
        & (neg["coherence"] >= 50)
    )
    pos_eff, neg_eff = pos[mask], neg[mask]
    questions = pos_eff["question"].tolist()
    return (
        list(zip(questions, pos_eff["answer"].tolist())),
        list(zip(questions, neg_eff["answer"].tolist())),
    )


def write_jsonl(pairs, path):
    with open(path, "w") as f:
        for q, a in pairs:
            row = {"messages": [
                {"role": "user", "content": str(q)},
                {"role": "assistant", "content": str(a)},
            ]}
            f.write(json.dumps(row) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Build weight-steering SFT data")
    parser.add_argument("--out_dir", default="data/0611/ws_sft_data")
    parser.add_argument("--threshold", type=int, default=50,
                        help="Effective-example trait-score threshold (matches persona vectors)")
    parser.add_argument("--value", default=None, help="Single constraint (default: all 9)")
    parser.add_argument("--persona_extract_dir", default=str(PERSONA_EXTRACT),
                        help="Dir of {value}_{pos,neg}_instruct.csv scored completions. "
                             "Default = the 0609 'default'-prompt set; pass the "
                             "eval_persona_extract/<model>/conflictscope dir for the CS-prompt variant.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    values = [args.value] if args.value else CONSTRAINTS

    for value in values:
        pos_pairs, neg_pairs = effective_rows(value, args.threshold, args.persona_extract_dir)
        write_jsonl(pos_pairs, out_dir / f"{value}_pos.jsonl")
        write_jsonl(neg_pairs, out_dir / f"{value}_neg.jsonl")
        print(f"{value}: {len(pos_pairs)} pos / {len(neg_pairs)} neg "
              f"(threshold {args.threshold})")


if __name__ == "__main__":
    main()
