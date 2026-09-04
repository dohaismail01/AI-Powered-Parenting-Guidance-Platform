# -*- coding: utf-8 -*-
"""
Build the master Q&A dataset from the authored batch files.

Reads  dataset_build/batches/batch_*.json   (each: list of {chunk_id, question, answer[, source_text]})
Writes data/parenting_qa_dataset.json        ({chunk_id, question, answer, source_text})

- For chunks, source_text is attached verbatim from data/chunks.json by chunk_id.
- Behavior/safety items (chunk_id like "safety-###") carry their own source_text.
- Duplicates (same chunk_id + question) and unknown chunk_ids are reported and skipped.

Run:  python dataset_build/build_dataset.py     (from the repo root)
"""
import json, glob, os

HERE = os.path.dirname(os.path.abspath(__file__))          # dataset_build/
ROOT = os.path.dirname(HERE)                               # repo root
DATA = os.path.join(ROOT, "data")
BATCH_DIR = os.path.join(HERE, "batches")
OUT = os.path.join(DATA, "parenting_qa_dataset.json")

chunks = json.load(open(os.path.join(DATA, "chunks.json"), encoding="utf-8"))
text_by_id = {c["chunk_id"]: c["text"] for c in chunks}

records = []
seen_pairs = set()
errors = []
done_chunk_ids = set()

for path in sorted(glob.glob(os.path.join(BATCH_DIR, "batch_*.json"))):
    data = json.load(open(path, encoding="utf-8"))
    for item in data:
        cid = item["chunk_id"]
        # Behavior/safety items carry their own source_text and are not tied to a chunk.
        explicit_src = item.get("source_text")
        if cid not in text_by_id and not explicit_src:
            errors.append(f"{os.path.basename(path)}: unknown chunk_id {cid}")
            continue
        src = text_by_id.get(cid, explicit_src)
        if cid in text_by_id:
            done_chunk_ids.add(cid)
        q = item["question"].strip()
        a = item["answer"].strip()
        key = (cid, q)
        if key in seen_pairs:
            errors.append(f"{os.path.basename(path)}: duplicate q for {cid}")
            continue
        seen_pairs.add(key)
        records.append({
            "chunk_id": cid,
            "question": q,
            "answer": a,
            "source_text": src,
        })

json.dump(records, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

if errors:
    print("ERRORS:")
    for e in errors:
        print("  ", e)
print(f"records: {len(records)}  |  chunks covered: {len(done_chunk_ids)}")
