"""
Text preprocessing and review-statistics helpers.

This module keeps the ORIGINAL encoding logic used by the trained model
untouched. It only adds extra, purely descriptive statistics that are
computed independently of the model and never fed back into it, so the
prediction pipeline is 100% unaffected.
"""

from __future__ import annotations

import re
from typing import Dict, List


# ---------------------------------------------------------------------------
# Original encoding logic (unchanged) — required by the trained LSTM model
# ---------------------------------------------------------------------------

def encode_review(text: str, word_index: Dict[str, int]) -> List[int]:
    """Encode raw review text into the integer sequence the model expects.

    This mirrors the original preprocessing exactly:
    - lowercase + whitespace split (no retraining / re-tokenizing)
    - IMDB word_index lookup, offset by +3
    - unknown / out-of-vocab words map to index 2

    Args:
        text: Raw review text typed by the user.
        word_index: The IMDB word -> index mapping.

    Returns:
        A list of integer token ids (unpadded).
    """
    words = text.lower().split()
    encoded: List[int] = []

    for word in words:
        index = word_index.get(word)
        if index is not None and index < 10000:
            encoded.append(index + 3)
        else:
            encoded.append(2)

    return encoded


# ---------------------------------------------------------------------------
# Descriptive statistics (new, additive, non-invasive)
# ---------------------------------------------------------------------------

# Small curated lexicons used only for descriptive stats / emotion hints.
# These never influence the model's prediction — they're purely cosmetic.
POSITIVE_WORDS = {
    "amazing", "love", "loved", "great", "excellent", "wonderful", "fantastic",
    "brilliant", "beautiful", "perfect", "best", "enjoyed", "enjoy", "good",
    "masterpiece", "captivating", "charming", "delight", "delightful", "fun",
    "impressive", "outstanding", "superb", "touching", "recommend", "gem",
    "heartwarming", "hilarious", "stunning", "flawless", "gripping",
}

NEGATIVE_WORDS = {
    "bad", "terrible", "awful", "worst", "boring", "hate", "hated", "poor",
    "disappointing", "disappointed", "dull", "waste", "mess", "weak",
    "predictable", "cringe", "annoying", "mediocre", "forgettable", "flat",
    "confusing", "slow", "unwatchable", "painful", "horrible", "bland",
}

HAPPY_WORDS = POSITIVE_WORDS | {"joy", "cheerful", "uplifting", "warm"}
ANGRY_WORDS = {"furious", "angry", "rage", "infuriating", "hate", "hated", "outrageous"}
SAD_WORDS = {"sad", "tragic", "heartbreaking", "depressing", "tearjerker", "sorrow", "grief"}
EXCITED_WORDS = {"thrilling", "exciting", "epic", "explosive", "electrifying", "adrenaline", "intense"}


def get_review_stats(text: str) -> Dict[str, float]:
    """Compute descriptive statistics about a review.

    Args:
        text: Raw review text.

    Returns:
        Dictionary of descriptive metrics (counts, ratios, timing estimates).
    """
    words = text.split()
    word_count = len(words)
    char_count = len(text)

    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    sentence_count = max(len(sentences), 1 if text.strip() else 0)

    clean_words = [re.sub(r"[^a-zA-Z']", "", w).lower() for w in words]
    clean_words = [w for w in clean_words if w]

    avg_word_length = (
        sum(len(w) for w in clean_words) / len(clean_words) if clean_words else 0.0
    )
    longest_word = max(clean_words, key=len) if clean_words else ""

    reading_time_seconds = (word_count / 200) * 60  # 200 wpm average
    positive_hits = sum(1 for w in clean_words if w in POSITIVE_WORDS)
    negative_hits = sum(1 for w in clean_words if w in NEGATIVE_WORDS)

    unique_words = set(clean_words)
    lexical_diversity = (len(unique_words) / len(clean_words)) if clean_words else 0.0

    return {
        "word_count": word_count,
        "char_count": char_count,
        "sentence_count": sentence_count,
        "avg_word_length": round(avg_word_length, 2),
        "longest_word": longest_word,
        "reading_time_seconds": round(reading_time_seconds, 1),
        "positive_word_count": positive_hits,
        "negative_word_count": negative_hits,
        "lexical_diversity": round(lexical_diversity, 3),
    }


def estimate_emotions(text: str) -> Dict[str, int]:
    """Very lightweight keyword-based emotion-intensity estimate.

    Not a substitute for the model's sentiment prediction — purely a fun,
    descriptive add-on layered on top using simple business logic.

    Args:
        text: Raw review text.

    Returns:
        Dictionary mapping emotion name -> intensity score (0-100).
    """
    words = [re.sub(r"[^a-zA-Z']", "", w).lower() for w in text.split()]
    words = [w for w in words if w]
    total = max(len(words), 1)

    def score(lexicon: set) -> int:
        hits = sum(1 for w in words if w in lexicon)
        return min(100, int((hits / total) * 400))

    happy = score(HAPPY_WORDS)
    angry = score(ANGRY_WORDS)
    sad = score(SAD_WORDS)
    excited = score(EXCITED_WORDS)
    neutral = max(0, 100 - max(happy, angry, sad, excited))

    return {
        "Happy": happy,
        "Angry": angry,
        "Sad": sad,
        "Excited": excited,
        "Neutral": neutral,
    }