"""
Generation backend + a lightweight post-hoc grounding check.

The provider is pluggable (config.GENERATION_PROVIDER):
  - "nile_chat_ollama": the project's default. The SAME fine-tuned
    Nile-Chat-4B GGUF, served through a local Ollama daemon
    (config.OLLAMA_MODEL). Ollama's optimized kernels are much faster
    than the plain llama-cpp CPU wheel, and it can offload to GPU where
    the machine's Ollama CUDA build supports it (config.OLLAMA_NUM_GPU;
    kept at 0 = CPU on machines whose Ollama GPU build errors). No API
    key or per-call network egress.
  - "nile_chat_gguf": the same GGUF run in-process via llama-cpp-python
    (no external daemon). Correct but slow on a no-BLAS CPU wheel.
  - "anthropic": Claude API, kept as a fallback/comparison baseline —
    useful for sanity-checking retrieval quality independently of the
    fine-tuned model's behavior. Needs ANTHROPIC_API_KEY.
  - "local_qwen": placeholder for a non-quantized local alternative.

All the nile_chat_* providers serve the model that was actually
fine-tuned on this knowledge base, so they should give better-grounded,
correctly-toned Egyptian-Arabic answers than a general-purpose model.
"""
from __future__ import annotations

from config import (
    GENERATION_MAX_TOKENS,
    GENERATION_MODEL,
    GENERATION_PROVIDER,
    GENERATION_TEMPERATURE,
    NILE_CHAT_CTX_SIZE,
    NILE_CHAT_GGUF_FILENAME,
    NILE_CHAT_GGUF_REPO,
    NILE_CHAT_N_GPU_LAYERS,
    NILE_CHAT_N_THREADS,
    NILE_CHAT_TEMPERATURE,
    NILE_CHAT_TOP_P,
    OLLAMA_HOST,
    OLLAMA_MODEL,
    OLLAMA_NUM_GPU,
    OLLAMA_TIMEOUT_SECONDS,
)
from src.generation.prompt import SYSTEM_PROMPT, build_user_prompt
from src.query.query_analyzer import QueryAnalysis
from src.schema import RetrievedChunk

_nile_chat_model = None  # lazy-loaded singleton — loading a GGUF takes real time


def _resolve_gguf_path() -> str:
    """
    Download (or reuse the cached copy of) the fine-tuned GGUF from
    Hugging Face. If no filename is pinned in config, auto-detect it —
    preferring a q4_k_m quantization, since that's what the fine-tuning
    team's README documents as the CPU-serving artifact.
    """
    from huggingface_hub import hf_hub_download, list_repo_files

    filename = NILE_CHAT_GGUF_FILENAME
    if filename is None:
        files = list_repo_files(NILE_CHAT_GGUF_REPO)
        gguf_files = [f for f in files if f.endswith(".gguf")]
        if not gguf_files:
            raise RuntimeError(
                f"No .gguf file found in {NILE_CHAT_GGUF_REPO}. "
                "Check the repo, or set NILE_CHAT_GGUF_FILENAME in config.py explicitly."
            )
        preferred = [f for f in gguf_files if "q4_k_m" in f.lower()]
        filename = preferred[0] if preferred else gguf_files[0]

    return hf_hub_download(repo_id=NILE_CHAT_GGUF_REPO, filename=filename)


def _get_nile_chat_model():
    global _nile_chat_model
    if _nile_chat_model is None:
        from llama_cpp import Llama  # pip install llama-cpp-python

        model_path = _resolve_gguf_path()
        _nile_chat_model = Llama(
            model_path=model_path,
            n_ctx=NILE_CHAT_CTX_SIZE,
            n_threads=NILE_CHAT_N_THREADS,
            n_gpu_layers=NILE_CHAT_N_GPU_LAYERS,
            verbose=False,
        )
    return _nile_chat_model


def _build_gemma_prompt(system: str, user: str) -> str:
    """
    Nile-Chat-4B is Gemma-3-based. Gemma's chat template has no
    dedicated system-role turn, so the system prompt is folded into
    the start of the first user turn instead — the standard approach
    for Gemma-family models.
    """
    combined_user = f"{system.strip()}\n\n{user.strip()}"
    return f"<start_of_turn>user\n{combined_user}<end_of_turn>\n<start_of_turn>model\n"


def _generate_with_nile_chat_gguf(system: str, user: str) -> str:
    model = _get_nile_chat_model()
    prompt = _build_gemma_prompt(system, user)
    # Uses the fine-tuned model's own NILE_CHAT_* sampling settings rather than
    # the provider-agnostic GENERATION_TEMPERATURE — see config.py for why.
    output = model(
        prompt,
        max_tokens=GENERATION_MAX_TOKENS,
        temperature=NILE_CHAT_TEMPERATURE,
        top_p=NILE_CHAT_TOP_P,
        stop=["<end_of_turn>"],
    )
    return output["choices"][0]["text"].strip()


def _generate_with_nile_chat_ollama(system: str, user: str) -> str:
    """
    Generate with the same fine-tuned GGUF, but served by Ollama instead of
    llama-cpp-python. Ollama ships GPU/optimized-CPU builds, so this answers in
    seconds where the pure-CPU llama-cpp path takes minutes. We send the Gemma
    prompt we build ourselves with raw=True so Ollama does not re-apply a chat
    template on top of it — identical formatting to the llama-cpp path.
    """
    import requests

    prompt = _build_gemma_prompt(system, user)
    response = requests.post(
        f"{OLLAMA_HOST}/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "raw": True,
            "stream": False,
            "options": {
                "temperature": NILE_CHAT_TEMPERATURE,
                "top_p": NILE_CHAT_TOP_P,
                "num_predict": GENERATION_MAX_TOKENS,
                "num_gpu": OLLAMA_NUM_GPU,
                "stop": ["<end_of_turn>"],
            },
        },
        timeout=OLLAMA_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()["response"].strip()


def _generate_with_anthropic(system: str, user: str) -> str:
    import anthropic  # requires ANTHROPIC_API_KEY in the environment

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=GENERATION_MODEL,
        max_tokens=GENERATION_MAX_TOKENS,
        temperature=GENERATION_TEMPERATURE,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


def _generate_with_local_qwen(system: str, user: str) -> str:
    """
    Placeholder for a non-quantized local alternative. Requires
    `transformers` + a downloaded Qwen2.5-*-Instruct checkpoint. Left as
    a clearly-marked extension point — not needed now that
    nile_chat_gguf covers the offline/CPU use case.
    """
    raise NotImplementedError(
        "Wire up a local transformers.pipeline('text-generation', "
        "model='Qwen/Qwen2.5-7B-Instruct') call here if ever needed."
    )


_PROVIDERS = {
    "nile_chat_ollama": _generate_with_nile_chat_ollama,
    "nile_chat_gguf": _generate_with_nile_chat_gguf,
    "anthropic": _generate_with_anthropic,
    "local_qwen": _generate_with_local_qwen,
}


def generate_answer(query_analysis: QueryAnalysis, results: list[RetrievedChunk]) -> str:
    user_prompt = build_user_prompt(query_analysis, results)
    provider_fn = _PROVIDERS[GENERATION_PROVIDER]
    return provider_fn(SYSTEM_PROMPT, user_prompt)


def check_grounding(answer: str, results: list[RetrievedChunk]) -> bool:
    """
    Lightweight grounding heuristic: flags an answer as *possibly*
    unsupported if it shares very little vocabulary overlap with the
    retrieved context. This is a cheap first line of defense, not a
    substitute for a proper NLI-based faithfulness check (e.g. running
    each generated sentence against the context with an entailment
    model) — swap in RAGAS' faithfulness metric or a dedicated NLI
    model here once you have an evaluation set to validate the
    threshold against (see evaluate.py).
    """
    if not results:
        return False

    context_words = set()
    for result in results:
        context_words.update(result.chunk.text.split())

    answer_words = [w for w in answer.split() if len(w) > 2]
    if not answer_words:
        return True

    overlap = sum(1 for w in answer_words if w in context_words)
    overlap_ratio = overlap / len(answer_words)
    return overlap_ratio >= 0.15  # heuristic threshold — tune against evaluate.py