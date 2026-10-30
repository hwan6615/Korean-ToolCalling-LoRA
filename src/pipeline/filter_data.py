import json
import re
from collections import Counter

from src.pipeline.translate import check, string_values

LATIN = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
HANGUL = re.compile(r"[가-힣ㄱ-ㅎㅏ-ㅣ]")
GLUED = re.compile(r"[가-힣][A-Za-z]")


def norm(w):
    w = w.lower()
    return w[:-1] if w.endswith("s") else w


def stray_english(ko, en):
    en_words = {norm(w) for w in LATIN.findall(en)}
    return [w for w in LATIN.findall(ko) if len(w) >= 3 and norm(w) not in en_words]


def foreign_chars(ko, en):
    return sorted(c for c in set(ko) if ord(c) > 127 and not HANGUL.match(c) and c not in en)


def extra_check(ko, en):
    if foreign_chars(ko, en):
        return "다른 문자 혼입"
    if GLUED.search(ko):
        return "한글-영문 붙음"
    if stray_english(ko, en):
        return "영어 조각 혼입"
    return None


try:
    drop = {int(x) for x in open("data/drop_eval.txt", encoding="utf-8").read().split()}
except FileNotFoundError:
    drop = set()

rows = [json.loads(l) for l in open("data/translated.jsonl", encoding="utf-8")]
total, passed, reasons = Counter(), Counter(), Counter()
kept, examples, dropped = [], {}, 0

for r in rows:
    en, ko = r["query_en"], r["query_ko"]
    reason = check(ko, string_values(r["answers"]), en)
    if reason is None:
        reason = extra_check(ko, en)
        if reason:
            examples.setdefault(reason, []).append(ko)
    key = (r["split"], len(r["answers"]))
    total[key] += 1
    reasons[reason or "통과"] += 1
    if reason is not None:
        continue
    if r["split"] == "eval" and r["id"] in drop:
        dropped += 1
        continue
    passed[key] += 1
    kept.append({k: r[k] for k in ("id", "split", "query_en", "query_ko", "tools", "answers")})

with open("data/clean.jsonl", "w", encoding="utf-8") as f:
    for r in kept:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

with open("data/eval_review.txt", "w", encoding="utf-8") as f:
    for r in kept:
        if r["split"] == "eval":
            f.write(
                f"id={r['id']}\nEN: {r['query_en']}\nKO: {r['query_ko']}\n"
                f"CALL: {json.dumps(r['answers'], ensure_ascii=False)}\n\n"
            )

print("사유별 집계:", dict(reasons))
print("\n통과/전체 (split, 정답 호출 개수):")
for key in sorted(total):
    print(" ", key, f"{passed[key]}/{total[key]}")
print("\n최종:", {s: sum(r["split"] == s for r in kept) for s in ("train", "eval")},
      f"(검수로 제외한 eval {dropped}건)")
for reason, items in examples.items():
    print(f"\n[{reason}] 예시:")
    for ko in items[:5]:
        print("  ", ko)