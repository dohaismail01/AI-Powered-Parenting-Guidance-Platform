# -*- coding: utf-8 -*-
"""
Convert parenting_qa_dataset.json -> chat-format JSONL for fine-tuning Nile-Chat (Gemma-3 based).
Produces:
  - parenting_qa_train.jsonl  (95%)
  - parenting_qa_val.jsonl    (5%)
Run:  python dataset_build/make_finetune_data.py     (from the repo root)
"""
import json, random, os

HERE = os.path.dirname(os.path.abspath(__file__))   # dataset_build/
DATA = os.path.join(os.path.dirname(HERE), "data")  # repo/data
SRC = os.path.join(DATA, "parenting_qa_dataset.json")

# Parenting-assistant persona (Egyptian Arabic) — matches the tone of the answers.
SYSTEM_PROMPT = (
    "انت مساعد ذكي متخصص في تقديم النصايح والإرشادات للآباء والأمهات عن تربية الأطفال "
    "ورعايتهم. جاوب بطريقة دافية وداعمة وغير حكمية باللهجة المصرية، وقدّم نصايح عملية "
    "ومختصرة. ولما الموضوع يخص صحة الطفل أو سلامته أو تطوّره، انصح بلطف باستشارة طبيب "
    "أطفال أو مختص."
)

def main():
    data = json.load(open(SRC, encoding="utf-8"))
    records = []
    for r in data:
        q = r["question"].strip()
        a = r["answer"].strip()
        records.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": q},
                {"role": "assistant", "content": a},
            ],
            # keep provenance so you can trace back to the source chunk if needed
            "chunk_id": r["chunk_id"],
        })

    # deterministic shuffle + split
    random.seed(42)
    random.shuffle(records)
    n_val = max(1, round(len(records) * 0.05))
    val = records[:n_val]
    train = records[n_val:]

    def dump(path, rows):
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    dump(os.path.join(DATA, "parenting_qa_train.jsonl"), train)
    dump(os.path.join(DATA, "parenting_qa_val.jsonl"), val)

    print(f"total={len(records)}  train={len(train)}  val={len(val)}")
    print("wrote: parenting_qa_train.jsonl, parenting_qa_val.jsonl")

if __name__ == "__main__":
    main()
