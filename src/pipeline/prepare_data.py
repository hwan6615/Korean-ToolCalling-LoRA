import json

from transformers import AutoTokenizer

from src.common import MAX_LEN, MODEL_ID, build_prompt, target_text

tok = AutoTokenizer.from_pretrained(MODEL_ID)
rows = [json.loads(l) for l in open("data/clean.jsonl", encoding="utf-8")]
train = [r for r in rows if r["split"] == "train"]

out, lens, dropped = [], [], 0
for r in train:
    prompt = build_prompt(r["tools"], r["query_ko"])
    completion = [{"role": "assistant", "content": target_text(r["answers"])}]
    text = tok.apply_chat_template(prompt + completion, tokenize=False)
    n = len(tok(text)["input_ids"])
    if n > MAX_LEN:
        dropped += 1
        continue
    lens.append(n)
    out.append({"prompt": prompt, "completion": completion})

with open("data/train_sft.jsonl", "w", encoding="utf-8") as f:
    for o in out:
        f.write(json.dumps(o, ensure_ascii=False) + "\n")

lens.sort()
print(f"학습 샘플 {len(out)}건 (길이 초과로 제외 {dropped}건)")
print(f"토큰 길이 평균 {sum(lens)/len(lens):.0f}, p95 {lens[int(len(lens)*0.95)]}, 최대 {lens[-1]}")