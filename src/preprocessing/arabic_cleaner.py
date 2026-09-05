"""
Arabic text normalization.

This step matters more for Arabic than it would for English: the same
word can appear in several surface forms (different alef/yaa variants,
optional diacritics, kashida stretching) that are semantically
identical but would otherwise be treated as different tokens by BM25
and can add noise to embeddings. Normalizing before indexing measurably
improves retrieval recall.

We deliberately keep two variants of the text:
  - `normalized`: heavier normalization, used for BM25 / lexical matching
  - `display`: lighter cleaning only, used for what the user actually
    reads (we don't want to show a parent a de-diacritized, flattened
    version of an official UNICEF/WHO guide).
"""
from __future__ import annotations

import re

# Arabic diacritics (tashkeel) block
_ARABIC_DIACRITICS = re.compile(
    r"[\u064B-\u065F\u0670\u06D6-\u06ED]"
)

# Tatweel / kashida (used to visually stretch words, carries no meaning)
_TATWEEL = re.compile(r"\u0640")

# Alef variants -> bare alef
_ALEF_VARIANTS = re.compile(r"[\u0622\u0623\u0625\u0671]")

# Yaa variants -> bare yaa (dotless alef maksura is often used interchangeably)
_ALEF_MAKSURA = re.compile(r"\u0649")

# Taa marbuta -> haa is a common normalization choice for retrieval;
# NOTE: this changes word identity slightly, only applied to the
# lexical/BM25 variant, never to the display text.
_TAA_MARBUTA = re.compile(r"\u0629")

_MULTI_WHITESPACE = re.compile(r"\s+")

# Eastern Arabic-Indic digits -> Western digits (helps age/number matching)
_EASTERN_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


import unicodedata

def clean_display_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)  # ← السطر الجديد: يحوّل أشكال العرض الخاصة لعربي عادي
    text = text.replace("\u200f", "").replace("\u200e", "")
    text = _TATWEEL.sub("", text)
    text = _MULTI_WHITESPACE.sub(" ", text)
    return text.strip()


def normalize_for_retrieval(text: str) -> str:
    """Heavier normalization used only for BM25 tokenization / embeddings,
    never shown to the user directly."""
    text = clean_display_text(text)
    text = _ARABIC_DIACRITICS.sub("", text)
    text = _ALEF_VARIANTS.sub("\u0627", text)          # -> ا
    text = _ALEF_MAKSURA.sub("\u064A", text)            # -> ي
    text = _TAA_MARBUTA.sub("\u0647", text)             # -> ه
    text = text.translate(_EASTERN_DIGITS)
    text = _MULTI_WHITESPACE.sub(" ", text)
    return text.strip()


def simple_arabic_tokenize(text: str) -> list[str]:
    """Minimal whitespace/punctuation tokenizer for BM25, applied on the
    normalized text. A proper morphological tokenizer (e.g. CAMeL Tools'
    disambiguator) can be swapped in here for better recall on inflected
    forms if evaluation shows it's needed."""
    normalized = normalize_for_retrieval(text)
    tokens = re.findall(r"[\u0600-\u06FF0-9]+", normalized)
    return [t for t in tokens if len(t) > 1]
