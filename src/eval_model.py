import argparse
import json
import time
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.common import MODEL_ID, build_prompt, load_model, parse_calls, parse_calls_lenient, score

ap = argparse.ArgumentParser()
ap.add_argument("--adapter", default=None)
ap.add_argument("--name", default="base")
ap.add_argument("--hint", action="store_true", help="프롬프트에 출력 형식 예시 추가")
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--batch", type=int, default=8)
args = ap.parse_args()

rows = [json.loads(l) for l in open("data/clean.jsonl", encoding="utf-8")]
rows = [r for r in rows if r["split"] == "eval"]
if args.limit:
    rows = rows[: args.limit]

tok = AutoTokenizer.from_pretrained(MODEL_ID)
tok.padding_side = "left"
# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_ID, dtype=torch.bfloat16, device_map="auto"
# )
model = load_model()

if args.adapter:
    from peft import PeftModel

    model = PeftModel.from_pretrained(model, args.adapter)
model.eval()

prompts = [
    tok.apply_chat_template(
        build_prompt(r["tools"], r["query_ko"], hint=args.hint),
        add_generation_prompt=True,
        tokenize=False,
    )
    for r in rows
]
order = sorted(range(len(rows)), key=lambda i: len(prompts[i]))
outputs = [None] * len(rows)

t0 = time.time()
for s in range(0, len(order), args.batch):
    idx = order[s : s + args.batch]
    enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=300, do_sample=False)
    texts = tok.batch_decode(out[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
    for i, t in zip(idx, texts):
        outputs[i] = t
    print(f"진행 {min(s + args.batch, len(order))}/{len(order)}", end="\r")
elapsed = time.time() - t0

res = defaultdict(lambda: {"strict": [], "lenient": []})
fails = []
with open(f"outputs/eval/eval_out_{args.name}.jsonl", "w", encoding="utf-8") as f:
    for r, t in zip(rows, outputs):
        s1 = score(parse_calls(t), r["answers"])
        s2 = score(parse_calls_lenient(t), r["answers"])
        for key in (len(r["answers"]), "전체"):
            res[key]["strict"].append(s1)
            res[key]["lenient"].append(s2)
        if not s2["exact"]:
            fails.append((r, t))
        f.write(json.dumps({"id": r["id"], "output": t, "strict": s1, "lenient": s2}, ensure_ascii=False) + "\n")


def rate(items, k):
    return f"{sum(x[k] for x in items) / len(items):.1%}"


print(f"\n\n[{args.name}] hint={args.hint} 평가 {len(rows)}건, 소요 {elapsed:.0f}초")
for key in (1, 2, 3, "전체"):
    if key not in res:
        continue
    d = res[key]
    label = f"호출 {key}개" if key != "전체" else "전체"
    print(f"\n{label} (n={len(d['strict'])})")
    for mode, name in (("strict", "엄격"), ("lenient", "관대")):
        v = d[mode]
        print(f"  {name}: JSON 유효 {rate(v,'valid')} / 도구명 일치 {rate(v,'name')} / 완전 일치 {rate(v,'exact')}")

print("\n관대 채점으로도 틀린 사례 5건:")
for r, t in fails[:5]:
    print(f"\n질의: {r['query_ko']}\n정답: {json.dumps(r['answers'], ensure_ascii=False)}\n출력: {t[:300]}")