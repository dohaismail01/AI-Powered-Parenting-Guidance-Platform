# Troubleshooting log — Nile-Chat parenting fine-tune

Personal notes of problems hit while building this and how each was solved.
(Gitignored — not part of the shipped repo.)

---

## 1. Setup / dependencies (Kaggle)

### 1.1 pip dependency conflict on install
**Symptom:** installing Unsloth pulled `transformers 5.x` / `datasets 5.x` / `trl 0.11` and pip complained of incompatible versions.
**Cause:** newest transformers/datasets/trl aren't compatible with the Unsloth build on Kaggle.
**Fix:** pin compatible versions and force a clean install:
`transformers>=4.55.2,<4.57`, `trl>=0.20,<0.23`, `datasets>=3.4.1,<4.0` — installed with `pip uninstall -y` then `pip install --force-reinstall --no-cache-dir`.

### 1.2 `ImportError: cannot import name 'logging' from 'transformers'`
**Symptom:** import error after the install cell.
**Cause:** transformers was still the old 5.x in memory — the kernel wasn't restarted after the downgrade.
**Fix:** **Restart the kernel** after the install cell (mandatory). Added a hard-stop check in the verify cell: `assert int(transformers.__version__.split(".")[0]) == 4`.

### 1.3 `tpot` / `dill` / `google-adk` pip warnings
**Symptom:** red warnings about unrelated Kaggle packages.
**Cause:** pre-installed Kaggle packages we don't use.
**Fix:** harmless — ignore.

---

## 2. Data loading

### 2.1 `FileNotFoundError: /kaggle/input/parenting-qa/parenting_qa_train.jsonl`
**Symptom:** data cell couldn't find the JSONL.
**Cause:** the Kaggle dataset slug didn't match the hardcoded `DATA_DIR`.
**Fix:** the data cell now auto-discovers files by name anywhere under `/kaggle/input`:
`glob.glob(f"/kaggle/input/**/{name}", recursive=True)`, with a 5% train/val split fallback if the val file is missing.

---

## 3. Data quality

### 3.1 Misinformation in a source chunk
**Symptom:** the Mosul "digital addiction" paper claimed screen use *causes* autism in children.
**Cause:** scientifically inaccurate source content.
**Fix:** did **not** propagate the autism-causation claim; extracted only the well-supported harms.

### 3.2 Institutional / garbled / TOC chunks
**Symptom:** many high-priority chunks were table-of-contents, references, policy text, or OCR-garbled (e.g. the socialization paper).
**Fix:** mined only genuine parent-facing prose (334 of 944 chunks); skipped apparatus rather than pad the dataset with junk register.

---

## 4. Training

### 4.1 Overfitting (validation loss rising)
**Symptom:** train loss fell each epoch but **val loss rose** (2.18 → 2.25 over 3 epochs); best was epoch 1.
**Cause:** 3 epochs too many for a small dataset; LR a bit high; the saved checkpoint was the *last* (worst) epoch.
**Fix (notebook):** `EPOCHS=2`, `LR=1e-4` (cell 6), `lora_dropout=0.05` (cell 8), and `load_best_model_at_end=True` + `metric_for_best_model="eval_loss"` + `save_total_limit=2` (cell 12) so the lowest-val-loss epoch is kept.

### 4.2 Notebook edits didn't take effect on Kaggle
**Symptom:** after editing the local notebook to `EPOCHS=2`, the run header still said `Num Epochs = 3`.
**Cause:** **Kaggle runs its own copy of the notebook.** Editing the `.ipynb` on disk does not change the Kaggle notebook. (The *data* update worked because the dataset was re-uploaded separately.)
**Fix:** re-import/sync the `.ipynb` to Kaggle (or edit the cells directly in the Kaggle editor) after any local change.

### 4.3 `Error displaying widget: model not found`
**Symptom:** shows in place of the training progress bar.
**Cause:** cosmetic ipywidgets rendering glitch.
**Fix:** harmless — ignore; training runs fine.

### 4.4 `Gemma3ForCausalLM does not accept num_items_in_batch`
**Symptom:** Unsloth note during training.
**Cause:** known Unsloth informational message about gradient accumulation.
**Fix:** harmless — ignore.

---

## 5. Evaluation

### 5.1 Catastrophic safety failures on the held-out set
**Symptom:** ~11/12 safety-critical questions failed — hallucinated doses, said "leaving a baby in the bath is fine", suggested aspirin for a child, etc. In-distribution topics (spanking, reading) were fine.
**Cause:** the training data had **no safety/refusal examples**, so the model — trained to always give a confident, warm, practical answer — hallucinated on out-of-distribution emergency/medical questions instead of deferring.
**Fix:** added a **24-example safety/refusal batch** (`batch_200.json`) teaching: emergencies → ER/doctor now; dosages → no numbers, ask doctor/pharmacist; dangerous myths → clear "no" + why; no-diagnosis; caregiver crisis; out-of-scope. Dataset grew 483 → 507. Verified 0 overlap with the held-out test questions.

---

## 6. GGUF export

### 6.1 `RuntimeError: Not enough disk space to convert to GGUF`
**Symptom:** export needed ~18.8 GB but `/kaggle/working` had ~4 GB free.
**Cause:** the 4B export writes a 16-bit merge + f16 GGUF + q4_k_m quant, all on disk at once; leftover dirs from a failed run were eating space.
**Fix:** added a disk guard to the GGUF cell — deletes leftover `nile-chat-parenting-gguf` / `_gguf` / `checkpoint-*` dirs, checks `free ≥ 19 GB`, and only then converts. If still tight: run the export in a **fresh session** that only loads the adapter from HF (empty disk → full ~19.5 GB free).

### 6.2 `NameError: name 'model' is not defined`
**Symptom:** the GGUF/eval cell failed on `model.save_pretrained_gguf(...)`.
**Cause:** the kernel had been reset, so `model` was no longer in memory.
**Fix:** the cell now reloads from HF if needed:
`try: model, tokenizer\nexcept NameError: model, tokenizer = FastModel.from_pretrained(HF_REPO, ..., load_in_4bit=False)`. No retraining required — the adapter is already on the Hub.

### 6.3 GGUF cell syntax error + double conversion
**Symptom:** the cell wouldn't parse (broken multi-line string in `raise RuntimeError(...)`), and an uncommented `push_to_hub_gguf` re-ran the whole merge a second time.
**Cause:** lost `\n` escapes in the error string; and `push_to_hub_gguf` re-converts from scratch (another ~18.8 GB).
**Fix:** cleaned the string; replaced the second conversion with a **file upload** of the already-written `.gguf` via `HfApi().upload_file` (no re-merge). Toggle with `PUSH_TO_HF`.

---

## 7. Hugging Face

### 7.1 Repo auto-creation
**Note:** you don't create the HF repo manually — cell 16 calls `create_repo(HF_REPO, private=False, exist_ok=True)` and makes it public on push.
**Requires:** a **Write** token (a Read token → 401/403), added as the Kaggle secret `HF_TOKEN`, and `HF_REPO` = `username/repo` spelled exactly.

### 7.2 Username spelling
**Symptom risk:** reload/push fails if `HF_REPO`'s username is wrong.
**Cause:** the notebook originally had `dohaismaill`; the stated username was `dohaiismail`.
**Fix:** confirm the exact username at `huggingface.co/<username>` and set `HF_REPO` to match before running.

---

## 8. Model size choice

**Decision:** default to **Nile-Chat-4B**, not 12B.
**Why:** 4B trains with headroom on a T4 (12B is borderline/OOM), and its q4_k_m GGUF (~2.5 GB) runs comfortably on CPU for the RAG deployment — vs ~7–8 GB and slower for 12B. Switch to 12B only with GPU/RAM to spare.
