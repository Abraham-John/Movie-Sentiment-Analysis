"""
Model loading and prediction logic.

The prediction logic itself (encoding -> padding -> model.predict -> 0.5
threshold) is IDENTICAL to the original app. Nothing about how the model
makes its decision has changed — this module only adds timing, structured
output, and human-readable explanations around that same call.
"""

from __future__ import annotations

import os
import time
from typing import Dict, List

import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

from utils.preprocessing import encode_review

MODEL_CANDIDATE_PATHS = [
    os.path.join("models", "simple_rnn_imdb.h5"),
    "simple_rnn_imdb.h5",
]


@st.cache_resource(show_spinner=False)
def load_model() -> tf.keras.Model:
    """Load the trained LSTM/RNN model (cached across reruns).

    Looks in `models/simple_rnn_imdb.h5` first, then falls back to the
    project root, so this works whether or not the repo restructuring has
    been applied.

    Returns:
        The loaded Keras model.

    Raises:
        FileNotFoundError: If the model file cannot be found anywhere.
    """
    for path in MODEL_CANDIDATE_PATHS:
        if os.path.exists(path):
            return tf.keras.models.load_model(path)

    raise FileNotFoundError(
        "Could not find 'simple_rnn_imdb.h5'. Place it in ./models/ or the "
        "project root."
    )


@st.cache_resource(show_spinner=False)
def load_word_index() -> Dict[str, int]:
    """Load the IMDB word index (cached across reruns)."""
    from tensorflow.keras.datasets import imdb

    return imdb.get_word_index()


def predict_sentiment(review: str, model: tf.keras.Model, word_index: Dict[str, int]) -> Dict:
    """Run the (unchanged) prediction pipeline and package the result.

    Args:
        review: Raw review text.
        model: Loaded Keras model.
        word_index: IMDB word index mapping.

    Returns:
        Dict with sentiment label, emoji, confidence, both class
        probabilities, and processing time in milliseconds.
    """
    start = time.perf_counter()

    encoded: List[int] = encode_review(review, word_index)
    padded = pad_sequences([encoded], maxlen=500, padding="pre", truncating="pre")

    raw_prediction = float(model.predict(padded, verbose=0)[0][0])
    elapsed_ms = (time.perf_counter() - start) * 1000

    is_positive = raw_prediction >= 0.5
    confidence = raw_prediction if is_positive else 1 - raw_prediction

    return {
        "sentiment": "Positive" if is_positive else "Negative",
        "emoji": "😊" if is_positive else "😞",
        "confidence": confidence,
        "positive_probability": raw_prediction,
        "negative_probability": 1 - raw_prediction,
        "processing_time_ms": round(elapsed_ms, 2),
    }


def confidence_level(confidence: float) -> str:
    """Bucket a confidence score into High / Medium / Low."""
    if confidence >= 0.85:
        return "High"
    if confidence >= 0.65:
        return "Medium"
    return "Low"


def generate_explanation(result: Dict, stats: Dict) -> List[str]:
    """Generate simple, rule-based natural-language explanations.

    No external APIs are used — this is deterministic business logic built
    from the prediction confidence and the review's descriptive stats.

    Args:
        result: Output of `predict_sentiment`.
        stats: Output of `get_review_stats`.

    Returns:
        A list of short explanation bullet strings.
    """
    bullets: List[str] = []
    is_positive = result["sentiment"] == "Positive"
    level = confidence_level(result["confidence"])

    if is_positive:
        bullets.append("The review leans toward positive, optimistic language.")
        if stats["positive_word_count"] > 0:
            bullets.append(
                f"Detected {stats['positive_word_count']} positive keyword(s) "
                "commonly associated with favorable reviews."
            )
    else:
        bullets.append("The review leans toward negative, critical language.")
        if stats["negative_word_count"] > 0:
            bullets.append(
                f"Detected {stats['negative_word_count']} negative keyword(s) "
                "commonly associated with unfavorable reviews."
            )

    if level == "High":
        bullets.append("The model is highly confident in this prediction.")
    elif level == "Medium":
        bullets.append("The model shows moderate confidence — tone is fairly clear but mixed signals exist.")
    else:
        bullets.append("The model has lower confidence — the review may contain mixed or ambiguous sentiment.")

    if stats["word_count"] < 5:
        bullets.append("The review is very short, which can reduce prediction reliability.")

    if stats["positive_word_count"] > 0 and stats["negative_word_count"] > 0:
        bullets.append("Both positive and negative cues were found — this may be a mixed review.")

    return bullets