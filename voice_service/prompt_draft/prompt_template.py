"""
DRAFT prompt template linking RAG retrieval output to the fine-tuned LLM.

This is a placeholder draft, NOT the final version. Person 3 (teammate)
owns the actual "Context Injection Template" and 3-stage prompt chain -
replace this file's logic with hers once she delivers it.

Purpose in the meantime: lets Person 4 (integration) keep testing the
full pipeline end-to-end (STT -> retrieval -> LLM -> TTS) without
blocking on the final prompt design.
"""


def build_prompt(question: str, retrieved_chunks: list[dict]) -> str:
    """
    Combine a user's question with retrieved knowledge-base chunks into
    a single prompt for the fine-tuned model.

    `retrieved_chunks` is expected to be a list of dicts shaped like
    Chroma's query results, each with at least a "text" key (the chunk
    content) - matches what test_retrieval.py already returns.

    DRAFT ONLY: real prompt design (instruction wording, chunk
    formatting, safety framing) is Person 3's responsibility.
    """
    context_sections = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        text = chunk.get("text", "").strip()
        source = chunk.get("organization", "")
        context_sections.append(f"[{i}] ({source}) {text}")

    context_block = "\n\n".join(context_sections)

    prompt = f"""أنت مساعد تربية إيجابية. استخدم المعلومات دي من مصادر موثوقة للإجابة على السؤال:

{context_block}

السؤال: {question}

جاوب بالعامية المصرية، بناءً على المعلومات اللي فوق. لو السؤال طبي أو خطير، حوّلي لدكتور بدل ما تجاوبي بنفسك."""

    return prompt


if __name__ == "__main__":
    fake_chunks = [
        {"text": "الحمى فوق 39 درجة تعتبر خطيرة وتحتاج تدخل طبي فوري.", "organization": "WHO"},
        {"text": "تشنجات الحمى تحدث عادة عند الأطفال بين 6 أشهر و5 سنوات.", "organization": "UNICEF"},
    ]
    example_prompt = build_prompt("ابني عنده حمى 40 درجة، أعمل إيه؟", fake_chunks)
    print(example_prompt)