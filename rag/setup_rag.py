"""
Unattended, resumable RAG setup for a flaky network.

Downloads the three models the RAG backend needs, verifies the big
bge-m3 weight archive isn't corrupt, then builds the Chroma + BM25
indexes. Safe to re-run any number of times: every step is skipped if
already done, and every download resumes where it left off.

Run it in your OWN terminal (not tied to a Claude session):

    cd C:\\Users\\Doha\\Desktop\\AI-Powered-Parenting-Guidance-Platform\\rag
    py -3.11 setup_rag.py

Leave it running; on this network expect a couple of hours. If it dies
(DNS drop, timeout), just run it again — it picks up mid-file.

Skip the 2.49 GB GGUF (use Claude for generation instead) with:
    py -3.11 setup_rag.py --no-gguf      # then set GENERATION_PROVIDER="anthropic"
"""
from __future__ import annotations

import os
import sys
import time
import zipfile

# --- force the resilient download path (the xet backend fails on this network) ---
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

from huggingface_hub import hf_hub_download, snapshot_download  # noqa: E402

EMBED_REPO = "BAAI/bge-m3"
RERANK_REPO = "BAAI/bge-reranker-v2-m3"
GGUF_REPO = "dohaiismail/nile-chat-parenting-lora-gguf"
GGUF_FILE = "Nile-Chat-4B.Q4_K_M.gguf"

# Never pull the 2.27 GB ONNX blob or the demo images — the CPU/torch
# path never loads them, and they double the download.
IGNORE = ["onnx/*", "*.onnx", "*.onnx_data", "imgs/*", "*.h5", "*.msgpack"]

MAX_RETRIES = 100          # network drops are expected; keep resuming
RETRY_SLEEP_SECONDS = 15


def _retry(label: str, fn):
    """Call fn() with unlimited-ish retries on network errors. Returns fn()'s result."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn()
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001 - we genuinely want to retry anything network-ish
            print(f"  [{label}] attempt {attempt} failed: {type(exc).__name__}: {exc}")
            if attempt == MAX_RETRIES:
                print(f"  [{label}] giving up after {MAX_RETRIES} attempts.")
                raise
            print(f"  [{label}] resuming in {RETRY_SLEEP_SECONDS}s...")
            time.sleep(RETRY_SLEEP_SECONDS)


def _verify_bin(path: str) -> bool:
    """bge-m3's pytorch_model.bin is a zip; a truncated/corrupt copy passes the
    byte-count check but fails CRC. testzip() returns None when every entry is OK."""
    try:
        with zipfile.ZipFile(path) as zf:
            bad = zf.testzip()
        if bad is None:
            return True
        print(f"  [verify] CRC failure on '{bad}' inside {os.path.basename(path)}")
        return False
    except zipfile.BadZipFile:
        print(f"  [verify] {os.path.basename(path)} is not a valid archive")
        return False


def download_embed() -> str:
    print(f"[1/4] Downloading {EMBED_REPO} (weights + tokenizer, no ONNX)...")
    local = _retry(
        "bge-m3",
        lambda: snapshot_download(EMBED_REPO, ignore_patterns=IGNORE),
    )
    bin_path = os.path.join(local, "pytorch_model.bin")
    # Verify the big archive; if corrupt, delete just that file and re-pull once.
    if os.path.exists(bin_path) and not _verify_bin(bin_path):
        print("  [verify] deleting corrupt pytorch_model.bin and re-downloading...")
        real = os.path.realpath(bin_path)
        for p in {bin_path, real}:
            try:
                os.remove(p)
            except OSError:
                pass
        local = _retry(
            "bge-m3",
            lambda: snapshot_download(EMBED_REPO, ignore_patterns=IGNORE),
        )
        bin_path = os.path.join(local, "pytorch_model.bin")
        if not _verify_bin(bin_path):
            raise SystemExit("bge-m3 weights still corrupt after re-download. Check disk/network.")
    print(f"      OK -> {local}")
    return local


def download_reranker() -> str:
    print(f"[2/4] Downloading {RERANK_REPO}...")
    local = _retry(
        "reranker",
        lambda: snapshot_download(RERANK_REPO, ignore_patterns=IGNORE),
    )
    print(f"      OK -> {local}")
    return local


def download_gguf() -> str:
    print(f"[3/4] Downloading {GGUF_REPO}/{GGUF_FILE} (2.49 GB)...")
    path = _retry("gguf", lambda: hf_hub_download(repo_id=GGUF_REPO, filename=GGUF_FILE))
    print(f"      OK -> {path}")
    return path


def build_indexes(embed_path: str, rerank_path: str) -> None:
    print("[4/4] Building Chroma + BM25 indexes...")
    import subprocess

    env = dict(os.environ)
    env["EMBEDDING_MODEL_PATH"] = embed_path
    env["RERANKER_MODEL_PATH"] = rerank_path
    here = os.path.dirname(os.path.abspath(__file__))
    result = subprocess.run(
        [sys.executable, "build_index.py"], cwd=here, env=env
    )
    if result.returncode != 0:
        raise SystemExit(f"build_index.py failed (exit {result.returncode}).")
    print("      Indexes built.")


def main() -> None:
    want_gguf = "--no-gguf" not in sys.argv
    embed_path = download_embed()
    rerank_path = download_reranker()
    if want_gguf:
        download_gguf()
    else:
        print("[3/4] Skipping GGUF (--no-gguf). Set GENERATION_PROVIDER='anthropic' in config.py.")
    build_indexes(embed_path, rerank_path)

    print("\nDone. Next:")
    print("  1) Point the RAG server at the local models so it doesn't re-download:")
    print(f'       set EMBEDDING_MODEL_PATH={embed_path}')
    print(f'       set RERANKER_MODEL_PATH={rerank_path}')
    print("  2) cd rag && py -3.11 -m uvicorn app:app --port 8000")
    print("  3) cd voice_service && py -3.11 -m uvicorn stt_service:app --port 8001")
    print("  4) Open voice_service/parentwise_demo.html")


if __name__ == "__main__":
    main()
