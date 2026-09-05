"""
Query understanding.

Runs before retrieval to pull out structured signal from the raw
Arabic question — child's age, likely topic(s), coarse intent, and a
first-pass risk read (refined later by the dedicated safety
classifier in src/safety/). Deliberately rule/keyword based: an LLM
call per query would be the expensive, slower option, and for a
domain this bounded (parenting topics, a known age range 0-12+)
keyword rules are reliable and fully inspectable/debuggable.

Example:
    "ابني عنده 5 سنين وبيعمل نوبات غضب ومش بيسمع الكلام"
    -> age=5, topics=["نوبات الغضب", "السلوك", "التأديب"], risk="low"
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.preprocessing.arabic_cleaner import normalize_for_retrieval

# Topic keyword map — extend this as the knowledge base grows. Keys are the
# canonical topic tags used in Chunk.topics (must match the KB's own tags).
TOPIC_KEYWORDS: dict[str, list[str]] = {
    "نوبات الغضب": ["نوبة غضب", "نوبات غضب", "بيزعق", "بيصرخ", "تانتروم"],
    "السلوك": ["سلوك", "بيعمل", "مش بيسمع الكلام", "عنيد", "عناد"],
    "التأديب الإيجابي": ["تأديب", "عقاب", "أعاقبه", "أضربه", "عقوبة"],
    "التواصل بين الوالدين والأبناء": ["مش بيتكلم", "التواصل", "بيتجاهلني"],
    "النوم": ["نوم", "مش بينام", "الكوابيس", "روتين النوم"],
    "وقت الشاشة": ["الموبايل", "التابلت", "الشاشة", "الألعاب الإلكترونية", "يوتيوب"],
    "التنمر": ["متنمر", "بيتنمر عليه", "تنمر"],
    "الغيرة": ["غيرة", "غيران من أخوه"],
    "الاستعداد المدرسي": ["المدرسة", "التحصيل الدراسي", "الواجب"],
    "أساليب المعاملة الوالدية": ["أسلوب التربية", "طريقة التربية"],
}

RISK_KEYWORDS: dict[str, list[str]] = {
    "CHILD_ABUSE": [
    # --------------------------------------------------------
    # Sexual abuse / harassment
    # --------------------------------------------------------
    "اعتداء جنسي",
    "اعتداء على الطفل",
    "اعتدى عليه",
    "اعتدى عليها",
    "يتعرض لاعتداء",
    "تعرض لاعتداء",
    "تحرش",
    "تحرش جنسي",
    "يتحرش به",
    "يتحرش بها",
    "حد لمس ابني",
    "حد لمس بنتي",
    "لمسات غير مناسبة",
    "لمس غير لائق",
    "حد لمسه بطريقة مش مريحة",
    "حد لمسها بطريقة مش مريحة",
    "حد لمسه بشكل غير مريح",
    "حد لمسها بشكل غير مريح",
    "حد لمسه بطريقة مش مناسبة",
    "حد لمسها بطريقة مش مناسبة",

    # --------------------------------------------------------
    # Severe physical abuse - active voice
    # --------------------------------------------------------
    "ضرب مبرح",
    "ضرب شديد",
    "ضرب قاسي",
    "ضرب بعنف",
    "بيضربه بقسوة",
    "بيضربها بقسوة",
    "بتضربه بقسوة",
    "بتضربها بقسوة",
    "بيضربه بعنف",
    "بيضربها بعنف",
    "بتضربه بعنف",
    "بتضربها بعنف",

    # --------------------------------------------------------
    # Severe physical abuse - passive voice
    # IMPORTANT: covers the exact failure you found
    # --------------------------------------------------------
    "بيتضرب بقسوة",
    "بتتضرب بقسوة",
    "بيتضرب بعنف",
    "بتتضرب بعنف",
    "بيتضرب جامد",
    "بتتضرب جامد",
    "بيتضرب بشدة",
    "بتتضرب بشدة",
    "يتعرض للضرب",
    "بتتعرض للضرب",
    "بيتعرّض للضرب",
    "الطفل بيتضرب",
    "الطفلة بتتضرب",
    "ابني بيتضرب",
    "بنتي بتتضرب",
    "ابني بيتضرب جامد",
    "بنتي بتتضرب جامد",
    "ابني بيتضرب بعنف",
    "بنتي بتتضرب بعنف",
    "حد بيضرب ابني",
    "حد بيضرب بنتي",

    # --------------------------------------------------------
    # Severe physical punishment
    # --------------------------------------------------------
    "بيبرحه ضرب",
    "بتبرحه ضرب",
    "ضرب بالحزام",
    "ضرب بالعصا",
    "ضرب بالعصاية",
    "ضرب بعصا",
    "ضرب بآلة",
    "بيستخدم الحزام في الضرب",
    "بيستخدم العصا في الضرب",
    "كدمات",
    "آثار ضرب",
    "اثار ضرب",
    "علامات ضرب",
    "جروح بسبب الضرب",

    # --------------------------------------------------------
    # Abuse / mistreatment
    # --------------------------------------------------------
    "إساءة",
    "اساءة",
    "إساءة معاملة",
    "اساءة معاملة",
    "سوء معاملة",

    # Generic constructions
    "بيسيء معاملة",
    "بتسيء معاملة",
    "في حد بيسيء معاملة",
    "حد بيسيء معاملة ابني",
    "حد بيسيء معاملة بنتي",

    # Specific constructions
    "بيسيء معاملته",
    "بيسيء معاملتها",
    "بتسيء معاملته",
    "بتسيء معاملتها",
    "بيسيء معاملة ابني",
    "بيسيء معاملة بنتي",
    "بتسيء معاملة ابني",
    "بتسيء معاملة بنتي",

    "إساءة للأطفال",
    "اساءة للأطفال",
    "سوء معاملة طفل",
    "سوء معاملة الأطفال",
    "إساءة معاملة طفل",
    "اساءة معاملة طفل",

    # --------------------------------------------------------
    # Abuse / violence verbs
    # --------------------------------------------------------
    "بيعنفه",
    "بيعنفها",
    "بتعنفه",
    "بتعنفها",
    "بيعنّف",
    "بتعنّف",

    # --------------------------------------------------------
    # Physical harm
    # --------------------------------------------------------
    "أذية الطفل",
    "اذية الطفل",
    "بيأذيه",
    "بيأذيها",
    "بتأذيه",
    "بتأذيها",
    "يؤذيه",
    "يؤذيها",
    "مؤذي للطفل",
    "مؤذية للطفل",
    "تعذيب الطفل",
    "يعذب الطفل",
    "بتعذب الطفل",

    # --------------------------------------------------------
    # Neglect / dangerous punishment
    # --------------------------------------------------------
    "حرمان من الأكل",
    "حرمان الطفل من الأكل",
    "منع الأكل عنه",
    "ممنوع ياكل كعقاب",
    "حبس الطفل",
    "بيحبس الطفل",
    "بتحبس الطفل",
    "حبسه في الأوضة",
    "حبسه في الغرفة",
    "يقفله عليه",
    "بتقفل عليه",
    "عقاب مؤذي",
    "عقاب عنيف",

    # --------------------------------------------------------
    # Explicit child violence
    # --------------------------------------------------------
    "عنف ضد طفل",
    "عنف ضد الأطفال",
    "عنف مع طفل",
    "طفل يتعرض للعنف",
    "ابني بيتعرض للضرب",
    "بنتي بتتعرض للضرب",
],
    "VIOLENCE": [
        # General violence
        "عنف أسري",
        "عنف منزلي",
        "عنف في البيت",

        # Violence between siblings / people
        "بيضرب اخته",
        "بيضرب أخته",
        "بيضرب اخوه",
        "بيضرب أخوه",
        "بتضرب اخوها",
        "بتضرب أخوها",
        "بتضرب اختها",
        "بتضرب أختها",

        # Threats
        "بيهدد",
        "بتهدد",
        "تهديد",
        "بيهددها",
        "بيهدده",
        "بتهدده",
        "بتهددها",
        "هددني",
        "هددها",
        "هدده",

        # Violent fights
        "بيضرب حد",
        "بتضرب حد",
        "بيضرب الناس",
        "بيضرب الآخرين",
        "شجار عنيف",
        "خناقة عنيفة",
        "بيتعارك بعنف",
        "بيتشاجر بعنف",
        "يؤذي الآخرين",
        "بيأذي الناس",

        # Destructive / potentially dangerous behaviour
        "بيكسر حاجات بعنف",
        "بيكسر في البيت بعنف",
        "تكسير عنيف",
        "بيخبط في الناس",
        "بيرمي حاجات على الناس",

        # Immediate danger wording
        "خايفة يأذي حد",
        "خايف يأذي حد",
        "خطر على نفسه",
        "خطر على الآخرين",
        "خطر على اخواته",
        "خطر على إخوته",
        "ممكن يؤذي حد",
        "ممكن يضرب حد جامد",
    ],
    "SELF_HARM": [
        "إيذاء نفسه", "إيذاء نفسها", "بيأذي نفسه", "بتأذي نفسها",
        "انتحار", "عايز يموت", "عايزة تموت", "عاوز يموت", "عاوزة تموت",
        "عايز أموت", "عايزة أموت", "نفسي أموت", "بيقول عايز يموت",
        "بتقول عايزة تموت", "مش عايز يعيش", "مش عايزة تعيش",
        "مش عايز اعيش", "مش عايزة اعيش", "يأذي نفسه", "تأذي نفسها",
        "عايز يختفي من الدنيا","عايزة تختفي من الدنيا","عايز يختفي",
        "عايزة تختفي","نفسه يختفي من الدنيا","نفسها تختفي من الدنيا",
        "مش عايز يكمل","مش عايزة تكمل","مش عايز يعيش","مش عايزة تعيش",


    ],
    "MENTAL_HEALTH": ["اكتئاب", "قلق شديد", "توحد", "فرط حركة", "ADHD", "نفسي""اضطراب فرط الحركة","فرط الحركة وتشتت الانتباه","فرط الحركة","تشتت الانتباه",
        "ADHD",],
    "MEDICAL": ["تشخيص", "دواء", "علاج طبي", "أعراض مرضية", "حمى", "مريض"],
    "EMERGENCY": ["حالة طارئة", "خطر عليه دلوقتي", "لسه حصل"],
}

_AGE_PATTERN = re.compile(r"(\d{1,2})\s*(?:سن[ةه]|سنين|شهر)")


@dataclass
class QueryAnalysis:
    raw_query: str
    age: int | None = None
    topics: list[str] = field(default_factory=list)
    intent: str = "parenting_advice"
    risk_category: str = "NORMAL"

    def to_dict(self) -> dict:
        return {
            "age": self.age,
            "topics": self.topics,
            "intent": self.intent,
            "risk": self.risk_category,
        }


def _extract_age(text: str) -> int | None:
    match = _AGE_PATTERN.search(text)
    if match:
        return int(match.group(1))
    return None


def _strip_al(word: str) -> str:
    """Strip a leading definite article (ال) so "الشاشة" and "شاشة"
    match the same underlying keyword."""
    return word[2:] if word.startswith("\u0627\u0644") else word


def _extract_topics(normalized_text: str) -> list[str]:
    matched = []
    text_words = set(normalized_text.split())
    text_words_no_al = {_strip_al(w) for w in text_words}

    for topic, keywords in TOPIC_KEYWORDS.items():
        for kw in keywords:
            normalized_kw = normalize_for_retrieval(kw)
            # exact substring match (original behavior)...
            if normalized_kw in normalized_text:
                matched.append(topic)
                break
            # ...or match ignoring a leading "ال" on either side
            if _strip_al(normalized_kw) in text_words_no_al:
                matched.append(topic)
                break
    return matched


def _extract_risk(normalized_text: str) -> str:
    """
    Detect risk categories using normalized token-aware matching.

    The query and every keyword are normalized with the same Arabic
    normalization function. Multi-word keywords are matched as tokens,
    rather than requiring the exact phrase to appear as one substring.

    This makes matching more robust to:
    - different surrounding words
    - punctuation
    - Arabic normalization differences
    - extra words between keyword concepts

    It does NOT perform morphological analysis. Variants such as active
    vs passive voice still need either explicit keyword coverage or a
    semantic/LLM safety layer.
    """

    # Convert the normalized query into tokens once.
    text_words = normalized_text.split()
    text_word_set = set(text_words)

    # Order matters: highest-severity categories first.
    categories = [
        "EMERGENCY",
        "SELF_HARM",
        "CHILD_ABUSE",
        "VIOLENCE",
        "MENTAL_HEALTH",
        "MEDICAL",
    ]

    for category in categories:
        for keyword in RISK_KEYWORDS[category]:

            normalized_keyword = normalize_for_retrieval(keyword)
            keyword_words = normalized_keyword.split()

            if not keyword_words:
                continue

            # ------------------------------------------------
            # 1. Exact phrase match
            # ------------------------------------------------
            # Keeps the strongest and most precise behavior.
            if normalized_keyword in normalized_text:
                return category

            # ------------------------------------------------
            # 2. Token-set match for multi-word concepts
            # ------------------------------------------------
            # Every token from the keyword must exist as a
            # complete token somewhere in the query.
            #
            # Example:
            #
            # keyword:
            # "بيسيء معاملة"
            #
            # query:
            # "في حد بيسيء معاملة ابني في الحضانة"
            #
            # -> matches safely.
            #
            if len(keyword_words) > 1:
                if all(word in text_word_set for word in keyword_words):
                    return category

            # ------------------------------------------------
            # 3. Exact single-token match
            # ------------------------------------------------
            else:
                if keyword_words[0] in text_word_set:
                    return category

    return "NORMAL"

def analyze_query(raw_query: str) -> QueryAnalysis:
    normalized = normalize_for_retrieval(raw_query)
    return QueryAnalysis(
        raw_query=raw_query,
        age=_extract_age(raw_query),
        topics=_extract_topics(normalized),
        intent="parenting_advice",
        risk_category=_extract_risk(normalized),
    )
