#!/usr/bin/env python3
"""Assemble the Opus-NIST batched rerun outputs and score vs human."""
import json, glob, sys
from collections import Counter
from pathlib import Path

M = {"Full Support": "FS", "Partial Support": "PS", "No Support": "NS"}
LABS = ("NS", "PS", "FS")
RANK = {"NS": 0, "PS": 1, "FS": 2}

OUT_DIR = "/tmp/nist_full_out"
PAIRS = "citation-support-agents/analysis-lingwei/opus48-nist-rerun/pairs.jsonl"
DEST = Path("citation-support-agents/analysis-lingwei/opus48-nist-rerun")

# 1) collect verdicts from all batch output files
verds = {}
bad_files = []
for fp in sorted(glob.glob(f"{OUT_DIR}/*.json")):
    try:
        d = json.load(open(fp))
        for r in d["results"]:
            v = M.get(r["verdict"])
            if v:
                verds[r["id"]] = v
    except Exception as e:
        bad_files.append((fp, str(e)))

pairs = [json.loads(l) for l in open(PAIRS)]
human = {p["key"]: p["human_label"] for p in pairs}
missing = [k for k in human if k not in verds]
extra = [k for k in verds if k not in human]

print(f"judged: {len(verds)} / {len(human)} pairs")
print(f"missing: {len(missing)} | extra(unknown id): {len(extra)} | bad files: {len(bad_files)}")
if bad_files:
    print("  bad files:", bad_files[:5])
if missing:
    # which batches are incomplete
    miss_batches = Counter()
    idx = {p["key"]: i for i, p in enumerate(pairs)}
    for k in missing:
        miss_batches[idx[k] // 20] += 1
    print("  incomplete batches (batch_idx: missing_count):", dict(sorted(miss_batches.items())))
    print("  -> re-run these batches before final scoring.")

print("Opus-NIST verdict dist:", dict(Counter(verds[k] for k in human if k in verds)))

# 2) confusion matrix vs human (only judged pairs)
conf = {h: {o: 0 for o in LABS} for h in LABS}
agree = n = 0
hstrict = ostrict = 0
for k, h in human.items():
    if k not in verds:
        continue
    o = verds[k]; n += 1; conf[h][o] += 1
    if h == o: agree += 1
    elif RANK[h] < RANK[o]: hstrict += 1
    else: ostrict += 1

# Cohen's kappa (3-class)
po = agree / n
row = {h: sum(conf[h].values()) for h in LABS}
col = {o: sum(conf[h][o] for h in LABS) for o in LABS}
pe = sum((row[c] / n) * (col[c] / n) for c in LABS)
kappa = (po - pe) / (1 - pe) if pe != 1 else 0.0

print(f"\n=== Opus-4.8 (NIST prompt) vs Human  | N={n} ===")
print(f"agreement = {agree}/{n} = {po:.1%}   Cohen's kappa = {kappa:.3f}")
print(f"Opus more generous than human: {hstrict} | Opus stricter: {ostrict}")
print("confusion (rows=human, cols=Opus-NIST):")
print("            Opus-NS  Opus-PS  Opus-FS")
for h in LABS:
    print(f"  human-{h}:   {conf[h]['NS']:>6} {conf[h]['PS']:>8} {conf[h]['FS']:>8}")
print("\nReference (existing analysis, same 22-topic human set):")
print("  Opus-4.8 (simple prompt) 58.9% kappa 0.377 | GPT-4o 56.1% 0.339 | GPT-5.5 50.8% 0.273")

# 3) persist per-pair verdicts for the record
if not missing:
    with (DEST / "opus48_nist_verdicts.jsonl").open("w") as f:
        for p in pairs:
            f.write(json.dumps({"key": p["key"], "topic_id": p["topic_id"], "run_id": p["run_id"],
                                "sentence_index": p["sentence_index"], "citation_slot": p["citation_slot"],
                                "human_label": p["human_label"], "opus_nist_label": verds[p["key"]]}, ensure_ascii=False) + "\n")
    print(f"\nwrote {DEST}/opus48_nist_verdicts.jsonl ({len(pairs)} rows)")
else:
    print("\n(skipping final verdict file until missing batches are filled)")
