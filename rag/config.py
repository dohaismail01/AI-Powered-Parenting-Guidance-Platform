"""
ParentWise — global configuration.
Centralizes all paths, model names, and tunable pipeline parameters
so every module reads from one place.
"""
import os
from pathlib import Path
 
# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
 
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_PDFS_DIR = RAW_DIR / "pdfs"
RAW_WEB_DIR = RAW_DIR / "web"
RAW_RESEARCH_DIR = RAW_DIR / "research"
 
PROCESSED_DIR = DATA_DIR / "processed"
CLEANED_DIR = PROCESSED_DIR / "cleaned"
CHUNKS_DIR = PROCESSED_DIR / "chunks"
 
EVAL_DIR = DATA_DIR / "evaluation"
EVAL_QUESTIONS_PATH = EVAL_DIR / "eval_questions.json"
 
VECTOR_STORE_DIR = ROOT_DIR / "vector_store"
CHROMA_DIR = VECTOR_STORE_DIR / "chroma"
BM25_DIR = VECTOR_STORE_DIR / "bm25"
 
CHROMA_COLLECTION_NAME = "parentwise_kb"
 
# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------
CHUNK_TARGET_TOKENS = 400          # target chunk size (300-500 recommended)
CHUNK_OVERLAP_TOKENS = 50
CHUNK_MIN_TOKENS = 60              # merge tiny trailing chunks into previous one
 
# --------------------------------------------------------------------------
# Embedding model
# --------------------------------------------------------------------------
# BAAI/bge-m3: multilingual, strong on Arabic (MIRACL benchmark), open-source,
# supports long context (up to 8192 tokens) and dense+sparse+colbert vectors.
# We use it in dense mode here.
# Overridable with EMBEDDING_MODEL_PATH so you can point at an already-downloaded
# local snapshot. FlagEmbedding snapshot-downloads the *whole* repo for a bare
# hub id, which drags in a ~2.3 GB ONNX blob the CPU/torch path never loads;
# giving it a local directory skips that entirely.
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_PATH", "BAAI/bge-m3")
EMBEDDING_DEVICE = "cpu"           # set to "cuda" if a GPU is available
EMBEDDING_BATCH_SIZE = 16
EMBEDDING_DIM = 1024
 
# --------------------------------------------------------------------------
# Reranker
# --------------------------------------------------------------------------
# Same ONNX-blob caveat as EMBEDDING_MODEL_PATH above — point this at a local
# snapshot directory to skip the unused ONNX download.
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_PATH", "BAAI/bge-reranker-v2-m3")
RERANKER_DEVICE = "cpu"
 
# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------
DENSE_TOP_K = 25
BM25_TOP_K = 25
RRF_K = 60                          # standard RRF damping constant
FUSED_TOP_K = 30                    # candidates passed to metadata ranking
RERANK_TOP_K = 3                    # final chunks passed to the LLM (was 8; fewer
                                    # context tokens = much faster CPU generation)
MIN_RERANK_SCORE = 0.1 
# Metadata-aware ranking weights (must stay small relative to retrieval score
# so a highly authoritative-but-irrelevant chunk cannot outrank a relevant one)
WEIGHT_RETRIEVAL_SCORE = 1.0
WEIGHT_AGE_MATCH = 0.15
WEIGHT_TOPIC_MATCH = 0.10
WEIGHT_SOURCE_PRIORITY = 0.08
 
# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------
# Pluggable — see src/generation/generator.py. Providers:
#   "nile_chat_gguf" -> the fine-tuned parenting model from the
#                        fine-tuning team, quantized GGUF, runs on CPU
#                        via llama-cpp-python. DEFAULT for this project.
#   "anthropic"       -> Claude API (fallback / comparison baseline).
#   "local_qwen"      -> placeholder for an unquantized local model.
#   "nile_chat_ollama" -> the SAME fine-tuned GGUF, served by Ollama, which uses
#                        the GPU (or an optimized CPU build) -> answers in seconds
#                        instead of minutes. DEFAULT for interactive/live use.
GENERATION_PROVIDER = os.getenv("GENERATION_PROVIDER", "nile_chat_ollama")
 
# --- nile_chat_gguf settings ---
# Hugging Face repo hosting the fine-tuned GGUF (see the fine-tuning
# team's README). Filename is auto-detected (prefers a q4_k_m file) if
# left as None — set it explicitly if the repo ever has more than one
# GGUF and the wrong one gets picked.
NILE_CHAT_GGUF_REPO = "dohaiismail/nile-chat-parenting-lora-gguf"
NILE_CHAT_GGUF_FILENAME = None
NILE_CHAT_CTX_SIZE = 2048       # context window (was 4096; trimmed prompt fits, faster)
NILE_CHAT_N_THREADS = 8         # physical-core count on this 16-logical CPU (HT hurts llama.cpp)
NILE_CHAT_N_GPU_LAYERS = 0      # 0 = pure CPU; raise if a GPU build of llama-cpp-python is set up
# Matches the fine-tuning team's own eval/run_eval.py generation settings
# (temperature=0.7, top_p=0.9) — using the same sampling settings they
# evaluated the model with, rather than guessing our own, since sampling
# params can noticeably change a fine-tuned model's behavior.
NILE_CHAT_TEMPERATURE = 0.7
NILE_CHAT_TOP_P = 0.9

# --- nile_chat_ollama settings (fast GPU/optimized serving of the same GGUF) ---
# Model name exactly as it appears in `ollama list`. Pulled once with:
#   ollama pull hf.co/dohaiismail/nile-chat-parenting-lora-gguf:Q4_K_M
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL", "hf.co/dohaiismail/nile-chat-parenting-lora-gguf:Q4_K_M"
)
OLLAMA_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "120"))
# Number of model layers to offload to GPU. 0 = CPU only. Kept at 0 because this
# machine's Ollama CUDA build errors ("device kernel image is invalid") and the
# 4 GB GPU is too small for the 4B model + context anyway; Ollama's AVX CPU
# kernels still answer in ~15-20s (vs minutes for the no-BLAS llama-cpp wheel).
# Raise it once a compatible Ollama/driver GPU build is in place.
OLLAMA_NUM_GPU = int(os.getenv("OLLAMA_NUM_GPU", "0"))
 
# --- anthropic settings (fallback provider) ---
GENERATION_MODEL = "claude-sonnet-4-6"
 
GENERATION_MAX_TOKENS = 400   # was 1200; answers are short, caps runaway CPU generation
GENERATION_TEMPERATURE = 0.3
 
# --------------------------------------------------------------------------
# Safety
# --------------------------------------------------------------------------
RISK_CATEGORIES = [
    "NORMAL",
    "MEDICAL",
    "MENTAL_HEALTH",
    "CHILD_ABUSE",
    "VIOLENCE",
    "SELF_HARM",
    "EMERGENCY",
]
HIGH_RISK_CATEGORIES = {"CHILD_ABUSE", "VIOLENCE", "SELF_HARM", "EMERGENCY"}
 
EGYPT_EMERGENCY_LINE = "الخط الساخن لحماية الطفل في مصر: 16000"
 