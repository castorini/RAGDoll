#!/usr/bin/env python3
"""Compute still-missing pairs from accumulated outputs and build a throttled retry workflow."""
import json, glob, sys, shutil
from pathlib import Path
sys.path.insert(0,"/Users/lingweigu/Research/PiTREC-support-eval/src")
from pi_trec.support.prompts import render_support_prompt

M={"Full Support","Partial Support","No Support"}
OUT="/tmp/nist_full_out"
pairs=[json.loads(l) for l in open("/Users/lingweigu/Research/citation-support-agents/analysis-lingwei/opus48-nist-rerun/pairs.jsonl")]
by_id={p["key"]:p for p in pairs}

judged={}
for fp in glob.glob(f"{OUT}/*.json"):
    try:
        d=json.load(open(fp))
        for r in d.get("results",[]):
            if r.get("verdict") in M and r.get("id") in by_id:
                judged[r["id"]]=r["verdict"]
    except Exception:
        pass

missing=[p for p in pairs if p["key"] not in judged]
print(f"judged={len(judged)} / {len(pairs)} | missing={len(missing)}")

if not missing:
    print("ALL DONE — no retry needed.")
    sys.exit(0)

# re-batch missing (bigger batches => fewer agents => gentler on rate limit)
RB=Path("/tmp/nist_retry");
if RB.exists(): shutil.rmtree(RB)
RB.mkdir(parents=True)
B=50
specs=[]
for bi in range(0,len(missing),B):
    chunk=missing[bi:bi+B]
    gidx=pairs.index(chunk[0])
    inp=str(RB/f"r{gidx:05d}.jsonl"); outp=f"{OUT}/r{gidx:05d}.json"
    with open(inp,"w") as f:
        for p in chunk:
            f.write(json.dumps({"id":p["key"],"prompt":render_support_prompt(statement=p["statement"],citation=p["citation"],sentence_context=p["sentence_context"])},ensure_ascii=False)+"\n")
    specs.append({"in":inp,"out":outp})

INSTR=("You are completing a batch of INDEPENDENT TREC RAG support-assessment tasks.\n"
"1. Use the Read tool to read the JSONL file at {IN}. Each line is a JSON object with 'id' and 'prompt'.\n"
"2. Each 'prompt' is a complete, self-contained support-assessment task: judge whether the Cited Passage supports the Sentence and decide exactly one verdict from: Full Support, Partial Support, No Support. Judge EACH line INDEPENDENTLY, using ONLY that line's own prompt content.\n"
"3. Use the Write tool to write a JSON file to {OUT} with EXACTLY this shape: {\"results\":[{\"id\":\"<id copied from input>\",\"verdict\":\"<Full Support|Partial Support|No Support>\"}]} with one entry for EVERY input line, ids copied verbatim.\n"
"4. After writing the file, respond with the integer count of results you wrote.")

script='''export const meta = {
  name: 'opus-nist-retry',
  description: 'Throttled retry of missing NIST support batches (Opus 4.8 medium), sequential chunks to respect rate limits',
  phases: [{ title: 'Judge' }],
}
const SPECS = '''+json.dumps(specs)+''';
const INSTR = '''+json.dumps(INSTR)+''';
const RET = { type:'object', additionalProperties:false, required:['count'], properties:{ count:{ type:'number' } } };
function instr(s){ return INSTR.split('{IN}').join(s.in).split('{OUT}').join(s.out); }
phase('Judge');
const CHUNK = 5;            // ~5 concurrent at a time => gentle on the server rate limit
let ok = 0;
for (let i = 0; i < SPECS.length; i += CHUNK) {
  const grp = SPECS.slice(i, i + CHUNK);
  const r = await parallel(grp.map((s, j) => () =>
    agent(instr(s), { model:'opus', effort:'medium', agentType:'general-purpose', label:'retry'+(i+j), phase:'Judge', schema:RET })
      .then(x => (x && x.count > 0) ? 1 : 0)
      .catch(() => 0)
  ));
  ok += r.reduce((a,b)=>a+b,0);
  log('chunk '+(Math.floor(i/CHUNK)+1)+'/'+Math.ceil(SPECS.length/CHUNK)+' cumulative_ok_batches='+ok);
}
return { ok_batches: ok, total_batches: SPECS.length };
'''
open("/tmp/nist_retry.js","w").write(script)
print(f"retry batches={len(specs)} (size {B}); chunked 5-concurrent; wrote /tmp/nist_retry.js")
