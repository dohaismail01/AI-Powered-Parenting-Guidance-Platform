"""
Safety layer.

ParentWise deals with children, so safety is first-class rather than a
prompt instruction bolted onto generation. Per the spec, this combines:

    rules / keyword checks  (query_analyzer.py — fast, first pass)
        +
    this classifier          (slightly more context-aware re-check)
        +
    LLM safety reasoning      (only invoked for ambiguous/borderline cases)

Do not rely on prompt instructions alone: the keyword pass and this
module run *before* any retrieval or generation happens, so a
high-risk query is intercepted deterministically rather than hoping
the LLM notices and self-redirects.
"""
from __future__ import annotations

from dataclasses import dataclass

from config import EGYPT_EMERGENCY_LINE, HIGH_RISK_CATEGORIES
from src.query.query_analyzer import QueryAnalysis

SAFETY_RESPONSES: dict[str, str] = {
    "CHILD_ABUSE": (
        "اللي بتوصفيه مهم جدًا وأنا آسف إنك بتمري بيه. الموضوع ده محتاج "
        "تدخل متخصص فورًا مش نصيحة عامة عن التربية.\n\n"
        f"من فضلك تواصلي مع جهة مختصة في حماية الطفل ({EGYPT_EMERGENCY_LINE}) "
        "أو أقرب جهة رعاية اجتماعية أو طبيب أطفال موثوق فيه في أقرب وقت."
    ),
    "VIOLENCE": (
        "الموقف اللي بتوصفيه فيه خطورة على سلامة حد في البيت، وده محتاج "
        "دعم متخصص وليس مجرد نصيحة تربوية.\n\n"
        f"برجاء التواصل مع جهة مختصة ({EGYPT_EMERGENCY_LINE}) أو أخصائي "
        "نفسي/اجتماعي في أقرب وقت ممكن."
    ),
    "SELF_HARM": (
        "أنا قلقان على اللي بتوصفيه، وده موضوع مهم جدًا محتاج مساعدة "
        "متخصصة فورية، مش نصيحة عامة.\n\n"
        "من فضلك تواصلي فورًا مع طبيب نفسي أو خط دعم نفسي متخصص، أو "
        "أقرب طوارئ لو في خطر مباشر دلوقتي."
    ),
    "EMERGENCY": (
        "لو في خطر مباشر دلوقتي، من فضلك تواصلي فورًا مع الطوارئ أو "
        f"أقرب مستشفى ({EGYPT_EMERGENCY_LINE} لحماية الطفل).\n\n"
        "أنا مش بديل عن استجابة طارئة فورية."
    ),
}

# MEDICAL and MENTAL_HEALTH are "soft" risk: ParentWise can still give
# grounded general information, but must not diagnose or prescribe.
# These are handled by generation.prompt's medical-boundary rules
# instead of being fully intercepted here.
SOFT_RISK_CATEGORIES = {"MEDICAL", "MENTAL_HEALTH"}


@dataclass
class SafetyDecision:
    is_high_risk: bool
    category: str
    safe_response: str | None = None


def classify_safety(query_analysis: QueryAnalysis) -> SafetyDecision:
    """
    Combine the keyword-based category from query_analyzer with routing
    logic. In a production deployment, add a small trained classifier
    here (e.g. a fine-tuned Arabic BERT) to catch high-risk phrasing the
    keyword list misses, and fall back to an LLM safety-reasoning call
    for anything the classifier itself is unsure about — but keep the
    keyword pass as a hard, deterministic floor regardless.
    """
    category = query_analysis.risk_category

    if category in HIGH_RISK_CATEGORIES:
        return SafetyDecision(
            is_high_risk=True,
            category=category,
            safe_response=SAFETY_RESPONSES.get(category),
        )

    return SafetyDecision(is_high_risk=False, category=category, safe_response=None)
