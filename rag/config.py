"""
ParentWise — global configuration.
Centralizes all paths, model names, and tunable pipeline parameters
so every module reads from one place.
"""
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
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
EMBEDDING_DEVICE = "cpu"           # set to "cuda" if a GPU is available
EMBEDDING_BATCH_SIZE = 16
EMBEDDING_DIM = 1024
 
# --------------------------------------------------------------------------
# Reranker
# --------------------------------------------------------------------------
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
RERANKER_DEVICE = "cpu"
 
# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------
DENSE_TOP_K = 25
BM25_TOP_K = 25
RRF_K = 60                          # standard RRF damping constant
FUSED_TOP_K = 30                    # candidates passed to metadata ranking
RERANK_TOP_K = 8                    # final chunks passed to the LLM
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
GENERATION_PROVIDER = "nile_chat_gguf"
 
# --- nile_chat_gguf settings ---
# Hugging Face repo hosting the fine-tuned GGUF (see the fine-tuning
# team's README). Filename is auto-detected (prefers a q4_k_m file) if
# left as None — set it explicitly if the repo ever has more than one
# GGUF and the wrong one gets picked.
NILE_CHAT_GGUF_REPO = "dohaiismail/nile-chat-parenting-lora-gguf"
NILE_CHAT_GGUF_FILENAME = None
NILE_CHAT_CTX_SIZE = 4096       # context window (prompt + retrieved chunks + answer)
NILE_CHAT_N_THREADS = None      # None -> let llama.cpp auto-detect CPU threads
NILE_CHAT_N_GPU_LAYERS = 0      # 0 = pure CPU; raise if a GPU build of llama-cpp-python is set up
# Matches the fine-tuning team's own eval/run_eval.py generation settings
# (temperature=0.7, top_p=0.9) — using the same sampling settings they
# evaluated the model with, rather than guessing our own, since sampling
# params can noticeably change a fine-tuned model's behavior.
NILE_CHAT_TEMPERATURE = 0.7
NILE_CHAT_TOP_P = 0.9
 
# --- anthropic settings (fallback provider) ---
GENERATION_MODEL = "claude-sonnet-4-6"
 
GENERATION_MAX_TOKENS = 1200
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
 