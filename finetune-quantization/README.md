# Nile-Chat Parenting — Fine-tuning

Fine-tunes **[MBZUAI-Paris/Nile-Chat-4B](https://huggingface.co/MBZUAI-Paris/Nile-Chat-4B)** (Gemma-3 based, Egyptian-Arabic) into a warm, non-judgmental **parenting assistant** that answers in Egyptian colloquial Arabic and defers to a specialist on health/safety questions.

**Scope of this repo:** dataset → QLoRA fine-tune → GGUF quantization. The model is meant to run inside a **RAG pipeline** (retriever + prompt orchestration), which is built and owned separately — this repo delivers the model and the data behind it, not the retrieval stack.

## Model artifacts (Hugging Face)

| Artifact | Use | Link |
|---|---|---|
| LoRA adapter | load on top of `MBZUAI-Paris/Nile-Chat-4B` (GPU) | https://huggingface.co/dohaiismail/nile-chat-parenting-lora |
| GGUF (`q4_k_m`) | run on CPU / local hosting (llama.cpp, Ollama) | https://huggingface.co/dohaiismail/nile-chat-parenting-lora-gguf |

## Layout

```
data/
  chunks.json                     # source: 944 parenting text chunks (not modified)
  parenting_qa_dataset.json       # master Q&A, 507 pairs {chunk_id, question, answer, source_text}
  parenting_qa_train.jsonl        # training set (482) — chat format {messages, chunk_id}
  parenting_qa_val.jsonl          # validation set (25)
  parenting_heldout_testset.json  # 28 tricky questions + rubrics (never used in training)

dataset_build/
  build_dataset.py                # batches/ -> data/parenting_qa_dataset.json
  make_finetune_data.py           # dataset -> data/train.jsonl + val.jsonl (95/5, seed 42)
  batches/                        # authored Q&A source (batch_*.json)

notebooks/
  nile_chat_finetune_kaggle.ipynb # QLoRA fine-tune on Kaggle (T4) + HF push + GGUF export

eval/
  run_eval.py                     # generate answers over the held-out test set for grading

requirements.txt                  # deps to RUN the model (GGUF on CPU) — not for training
```

## Dataset

507 Q&A pairs grounded in the source chunks, in Egyptian colloquial Arabic:
- **483** grounded parenting pairs mined from the chunks (development, feeding, discipline, positive parenting, socialization, digital habits, …).
- **24 safety/refusal pairs** teaching the required behavior on **emergencies** (→ ER/doctor now), **dosages** (no numbers → doctor/pharmacist), **dangerous myths** (clear "no" + why), **no-diagnosis**, **caregiver crisis**, and **out-of-scope** questions.

Each pair carries `source_text` for provenance. The held-out test set has **0 overlap** with training and is graded against per-item rubrics (`must_include` / `must_avoid`, `safety_critical`).

### Reproduce the data
```bash
python dataset_build/build_dataset.py        # batches -> data/parenting_qa_dataset.json
python dataset_build/make_finetune_data.py   # -> data/parenting_qa_train.jsonl + val.jsonl
```
To grow the dataset, add a `dataset_build/batches/batch_*.json` file (`[{chunk_id, question, answer}]`, or `source_text` for behavior items) and rerun both scripts.

## Train (Kaggle)

Open `notebooks/nile_chat_finetune_kaggle.ipynb` on Kaggle (GPU: T4).
1. Attach a Kaggle dataset containing `parenting_qa_train.jsonl` + `parenting_qa_val.jsonl` (the notebook auto-discovers them by name). Add `parenting_heldout_testset.json` too for the eval cell.
2. Add your Hugging Face **write** token as the Kaggle secret `HF_TOKEN`, and set `HF_REPO` to your `username/repo`.
3. **Restart & Run All.** It fine-tunes with QLoRA (4-bit LoRA), pushes the adapter to HF (public), runs the held-out eval (`eval_results.json`), and exports a `q4_k_m` GGUF for CPU hosting.

Key config (notebook cells 6/8/12): 2 epochs, LR 1e-4, LoRA r=16 / dropout 0.05, `train_on_responses_only`, keep-best-checkpoint by val loss.

## Evaluate

The notebook has an eval cell that writes `eval_results.json`. Standalone:
```bash
python eval/run_eval.py --adapter <path-or-HF-repo> --out eval_outputs.json
```
Then read each answer against its rubric — **safety-critical items are hard fails** if wrong regardless of tone.

## Run the model (serving)

`requirements.txt` covers running the **GGUF** on CPU (`llama-cpp-python` + `huggingface-hub`). Retrieval/embeddings/API for the RAG pipeline are out of scope here.
