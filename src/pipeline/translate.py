import argparse
import json
import random
import re
import time
from collections import Counter

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    LogitsProcessor,
    LogitsProcessorList,
)

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
IN_PATH, OUT_PATH = "data/selected.jsonl", "data/translated.jsonl"
BATCH_SIZE, MAX_NEW_TOKENS = 16, 200

SYSTEM = (
    "You translate English user requests into natural Korean, the way a Korean user "
    "would type them to an AI assistant. Use one consistent register: casual-polite "
    "requests ending in ~해줘, ~알려줘, ~가져와줘, ~구해줘, or questions ending in ~어?, ~할까?. "
    "Never end a sentence with ~다, ~이다 or ~하세요. "
    "Translate every clause faithfully; do not drop, merge or invent content. "
    "Write only in Korean (Hangul). Never use Chinese characters or Japanese. "
    "Do not answer the request, do not add explanations or parentheses. "
    "Some strings are required values: copy them into the Korean sentence exactly as "
    "written, without translating or transliterating them. Keep numbers, names and "
    "codes unchanged. Output only the Korean text."
)
FEWSHOT = [
    (
        "Get the 5 latest posts from the user 'techguru' on Mastodon.",
        ["techguru", "Mastodon"],
        "Mastodon에서 techguru 사용자의 최신 게시글 5개를 가져와줘.",
    ),
    (
        "What are the details of the airline with IATA code 'LH'?",
        ["LH"],
        "IATA 코드가 LH인 항공사의 상세 정보를 알려줘.",
    ),
    (
        "Retrieve the specifications for the motorcycle 'YAMAHA_YZF_R1' and the price of the car with ID 'CAR_204'.",
        ["YAMAHA_YZF_R1", "CAR_204"],
        "오토바이 YAMAHA_YZF_R1의 사양이랑 ID가 CAR_204인 자동차의 가격을 알려줘.",
    ),
    (
        "Calculate the standard deviation of the numbers 5, 10, 15, 20, 25, and find the GCD of 144 and 288.",
        [],
        "숫자 5, 10, 15, 20, 25의 표준 편차를 계산하고, 144와 288의 최대공약수도 구해줘.",
    ),
    (
        "Get the ice hockey match schedules for October 31, 2023, and the box score for the game with ID '2022010101'.",
        ["2022010101"],
        "2023년 10월 31일 아이스하키 경기 일정이랑 ID가 '2022010101'인 경기의 박스스코어를 가져와줘.",
    ),
    (
        "Find the longest palindromic substring in the string 'racecar'.",
        ["racecar"],
        "'racecar' 문자열에서 가장 긴 회문 부분 문자열을 찾아줘.",
    ),
]
HANZI = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]")
FULLWIDTH = re.compile(r"[（）？！：，。]")

class BanCJK(LogitsProcessor):
    def __init__(self, tok):
        self.ids = [i for i in range(len(tok)) if HANZI.search(tok.decode([i]))]
        self.mask = None

    def __call__(self, input_ids, scores):
        if self.mask is None:
            self.mask = torch.zeros(scores.shape[-1], dtype=torch.bool, device=scores.device)
            self.mask[self.ids] = True
        return scores.masked_fill(self.mask, float("-inf"))


def num_tokens(s):
    s = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", s)
    return set(re.findall(r"\d+(?:\.\d+)?", s))

def clean(ko):
    ko = ko.strip().split("\n")[0].strip()
    m = HANZI.search(ko)
    if m:
        head = ko[: m.start()].rstrip()
        if head and head[-1] in ".?!":
            return head
    return ko

def bad_ending(ko):
    parts = [p.strip().rstrip("'\"’” )") for p in re.split(r"[.?!]", ko)]
    return any(p.endswith("다") for p in parts if p)

def string_values(answers):
    vals = []
    for a in answers:
        args = a.get("arguments")
        if isinstance(args, dict):
            vals += [v for v in args.values() if isinstance(v, str) and v]
    return list(dict.fromkeys(vals))


def has_value(ko, v):
    pat = r"(?<![A-Za-z0-9])" + re.escape(v) + r"(?![A-Za-z0-9])"
    return re.search(pat, ko, re.I) is not None


def user_turn(query, vals):
    return f"English: {query}\nRequired values: {json.dumps(vals, ensure_ascii=False)}"


def build_messages(row, vals):
    msgs = [{"role": "system", "content": SYSTEM}]
    for q, v, ko in FEWSHOT:
        msgs.append({"role": "user", "content": user_turn(q, v)})
        msgs.append({"role": "assistant", "content": ko})
    msgs.append({"role": "user", "content": user_turn(row["query"], vals)})
    return msgs


def check(ko, vals, en):
    """통과하면 None, 실패하면 사유 문자열을 반환."""
    if not re.search(r"[가-힣]", ko):
        return "한글 없음"
    if HANZI.search(ko):
        return "한자/가나 혼입"
    if FULLWIDTH.search(ko) or ("(" in ko and "(" not in en):
        return "부가 설명/전각 부호"
    if bad_ending(ko):
        return "말투(다 종결)"
    if not 0.3 <= len(ko) / max(len(en), 1) <= 1.6:
        return "길이 이상"
    if not all(has_value(ko, v) for v in vals):
        return "필수값 누락"
    if not num_tokens(en) <= num_tokens(ko):
        return "숫자 누락/변경"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(IN_PATH, encoding="utf-8")]
    try:
        done = {json.loads(l)["id"] for l in open(OUT_PATH, encoding="utf-8")}
    except FileNotFoundError:
        done = set()
    pending = [r for r in rows if r["id"] not in done]
    if args.limit:
        pending = pending[: args.limit]
    pending.sort(key=lambda r: len(r["query"]))
    print(f"번역 대상 {len(pending)}건 (이미 완료 {len(done)}건)")

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.padding_side = "left"
    ban = LogitsProcessorList([BanCJK(tok)])
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb, device_map="auto"
    )

    t0 = time.time()
    results, reasons = [], Counter()
    with open(OUT_PATH, "a", encoding="utf-8") as f:
        for i in range(0, len(pending), BATCH_SIZE):
            batch = pending[i : i + BATCH_SIZE]
            vals_list = [string_values(r["answers"]) for r in batch]
            prompts = [
                tok.apply_chat_template(
                    build_messages(r, v), add_generation_prompt=True, tokenize=False
                )
                for r, v in zip(batch, vals_list)
            ]
            enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **enc, 
                    max_new_tokens=MAX_NEW_TOKENS, 
                    do_sample=False,
                    logits_processor=ban,
                )
            texts = tok.batch_decode(
                out[:, enc["input_ids"].shape[1] :], skip_special_tokens=True
            )
            for r, v, ko in zip(batch, vals_list, texts):
                ko = clean(ko)
                reason = check(ko, v, r["query"])
                reasons[reason or "통과"] += 1
                r2 = {
                    **r,
                    "query_en": r["query"],
                    "query_ko": ko,
                    "ok": reason is None,
                    "reason": reason,
                }
                results.append((r2, v))
                f.write(json.dumps(r2, ensure_ascii=False) + "\n")
            f.flush()
            print(f"진행 {min(i + BATCH_SIZE, len(pending))}/{len(pending)}", end="\r")

    el = time.time() - t0
    n = max(len(pending), 1)
    print("\n\n결과 집계:", dict(reasons))
    print(f"소요 {el:.0f}초 (건당 {el / n:.1f}초), "
          f"VRAM 최대 {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

    random.seed(0)
    oks = [x for x in results if x[0]["ok"]]
    fails = [x for x in results if not x[0]["ok"]]
    print("\n===== 통과 샘플(무작위 20건) =====")
    for r, v in random.sample(oks, min(20, len(oks))):
        print(f"\n{r['query_en']}\n  -> {r['query_ko']}\n  필수값: {v}")
    print("\n===== 실패 샘플(무작위 5건) =====")
    for r, v in random.sample(fails, min(5, len(fails))):
        print(f"\n[{r['reason']}] {r['query_en']}\n  -> {r['query_ko']}")


if __name__ == "__main__":
    main()