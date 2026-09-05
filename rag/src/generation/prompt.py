"""
Prompt construction.
 
Two things matter most here:
1. The context block is *structured*, not a dumb concatenation of raw
   chunk text — each chunk keeps its source/organization/section/page
   so the model can cite accurately instead of inventing citations.
2. The system prompt starts from the fine-tuning team's *exact* base
   system message (see their eval/run_eval.py) rather than a separate
   one written independently — the model was fine-tuned to answer in
   warm Egyptian-colloquial style against that specific message, and a
   differently-worded/formal-MSA system prompt risks fighting that
   training instead of building on it. RAG-specific rules (grounding in
   retrieved context, citing sources, age-awareness) are appended after
   it, in the same colloquial register, rather than replacing it.
"""
from __future__ import annotations
 
from src.query.query_analyzer import QueryAnalysis
from src.schema import RetrievedChunk
 
# Base message: verbatim from the fine-tuning team's eval/run_eval.py
# (the SYSTEM constant they trained/evaluated against). Do not reword
# this part — it's the message the model was actually tuned on.
_BASE_SYSTEM = (
    "انت مساعد ذكي متخصص في تقديم النصايح والإرشادات للآباء والأمهات عن "
    "تربية الأطفال ورعايتهم. جاوب بطريقة دافية وداعمة وغير حكمية باللهجة "
    "المصرية، وقدّم نصايح عملية ومختصرة. ولما الموضوع يخص صحة الطفل أو "
    "سلامته أو تطوّره، انصح بلطف باستشارة طبيب أطفال أو مختص."
)
 
# RAG-specific additions, kept in the same Egyptian-colloquial register
# so they read as a natural extension of the base message rather than
# a jarring shift into formal MSA.
_RAG_ADDITIONS = """\
 
قواعد إضافية للإجابة (مبنية على مصادر UNICEF وWHO وأبحاث عربية محكّمة):
- جاوب بس من المعلومات الموجودة في "السياق المسترجع" تحت. لو مش لاقي فيه \
حاجة كفاية تجاوب بيها، قول كده صراحة بدل ما تخمّن.
- ميّز بين الأدلة العلمية والنصايح العامة. لو مصدر واحد بس بيقول إن فيه \
"ارتباط" بين حاجتين، متحولهاش لعلاقة سببية أكيدة إلا لو الدليل نفسه بيقول كده.
- اذكر المصادر اللي استخدمتها في آخر إجابتك: اسم الجهة — عنوان المصدر \
(والصفحة لو موجودة)، من الـ metadata المرفقة مع كل مقطع بس، من غير ما تخترع \
تفاصيل استشهاد مش موجودة.
- خد بالك من عمر الطفل المذكور في السؤال، ووجّه إجابتك يناسب المرحلة \
العمرية دي لو المعلومة متاحة.
"""
 
SYSTEM_PROMPT = _BASE_SYSTEM + "\n" + _RAG_ADDITIONS
 
 
def format_source_line(chunk_data) -> str:
    chunk = chunk_data.chunk
    parts = [chunk.organization, chunk.title]
    if chunk.section:
        parts.append(chunk.section)
    label = " — ".join(parts)
    if chunk.page:
        label += f" — ص. {chunk.page}"
    return label
 
 
def build_context_block(results: list[RetrievedChunk]) -> str:
    """
    Deduplicate near-identical chunks (same doc_id + section) and format
    the remaining evidence with clear source boundaries, so the model
    can attribute each claim to a specific, citable source.
    """
    seen: set[tuple[str, str | None]] = set()
    blocks = []
 
    for i, result in enumerate(results, start=1):
        chunk = result.chunk
        dedup_key = (chunk.doc_id, chunk.section)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
 
        source_line = format_source_line(result)
        blocks.append(
            f"[مصدر {i}] {source_line}\n"
            f"الفئة العمرية: {chunk.age_range or 'غير محدد'}\n"
            f"المحتوى:\n{chunk.text}"
        )
 
    return "\n\n---\n\n".join(blocks)
 
 
def build_user_prompt(query_analysis: QueryAnalysis, results: list[RetrievedChunk]) -> str:
    context_block = build_context_block(results)
 
    age_line = f"عمر الطفل المذكور: {query_analysis.age} سنة" if query_analysis.age else ""
    topics_line = (
        f"المواضيع المرتبطة بالسؤال: {', '.join(query_analysis.topics)}"
        if query_analysis.topics
        else ""
    )
 
    return f"""\
سؤال الوالد/مقدم الرعاية:
{query_analysis.raw_query}
 
{age_line}
{topics_line}
 
السياق المسترجع (استخدميه فقط للإجابة، واذكري مصادره في النهاية):
 
{context_block if context_block else "لا يوجد سياق مسترجع كافٍ لهذا السؤال."}
"""
 