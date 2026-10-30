import argparse
import json
import random
import time

import torch
from transformers import AutoTokenizer

from src.common import MAX_LEN, MODEL_ID, build_prompt, load_model

PATH = "data/external/FunctionChat-Bench/data/FunctionChat-Singlecall.jsonl"
TYPE_MAP = {
    "string": "str",
    "integer": "int",
    "number": "float",
    "boolean": "bool",
    "array": "list",
    "object": "dict",
}

ap = argparse.ArgumentParser()
ap.add_argument("--adapter", default=None)
ap.add_argument("--name", required=True)
ap.add_argument("--hint", action="store_true")
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--batch", type=int, default=4)
ap.add_argument("--note", action="store_true")
args = ap.parse_args()


def to_xlam(tool):
    f = tool["function"]
    schema = f.get("parameters") or {}
    required = set(schema.get("required") or [])
    params = {}
    for pname, spec in (schema.get("properties") or {}).items():
        p = {
            "description": spec.get("description", ""),
            "type": TYPE_MAP.get(spec.get("type"), str(spec.get("type"))),
        }
        if pname not in required:
            p["default"] = None
        params[pname] = p
    return {"name": f["name"], "description": f.get("description", ""), "parameters": params}


items = []
for line in open(PATH, encoding="utf-8"):
    row = json.loads(line)
    for tset in row["tools"]:
        tools = [to_xlam(t) for t in tset["content"]]
        for q, gt, acc in zip(row["query"], row["ground_truth"], row["acceptable_arguments"]):
            items.append(
                {
                    "function_name": row["function_name"],
                    "extraction_type": row["function_info"]["extraction_type"],
                    "tools_type": tset["type"],
                    "serial_num": q["serial_num"],
                    "query": q["content"],
                    "ground_truth": gt["content"],
                    "acceptable": acc["content"],
                    "tools": tools,
                }
            )
if args.limit:
    items = random.Random(0).sample(items, args.limit)
print(f"평가 항목 {len(items)}건")

tok = AutoTokenizer.from_pretrained(MODEL_ID)
tok.padding_side = "left"
model = load_model()
if args.adapter:
    from peft import PeftModel

    model = PeftModel.from_pretrained(model, args.adapter)
model.eval()

prompts = [
    tok.apply_chat_template(
        build_prompt(it["tools"], it["query"], hint=args.hint, note=args.note),
        add_generation_prompt=True,
        tokenize=False,
    )
    for it in items
]
lens = [len(tok(p)["input_ids"]) for p in prompts]
print(f"프롬프트 토큰 최대 {max(lens)}, {MAX_LEN} 초과 {sum(n > MAX_LEN for n in lens)}건")

order = sorted(range(len(items)), key=lambda i: lens[i])
outputs = [None] * len(items)
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

keys = ("function_name", "extraction_type", "tools_type", "serial_num", "query", "ground_truth", "acceptable")
with open(f"outputs/eval/fcb_out_{args.name}.jsonl", "w", encoding="utf-8") as f:
    for it, t in zip(items, outputs):
        f.write(json.dumps({**{k: it[k] for k in keys}, "output": t}, ensure_ascii=False) + "\n")
print(f"\n[{args.name}] 완료, 소요 {time.time() - t0:.0f}초")