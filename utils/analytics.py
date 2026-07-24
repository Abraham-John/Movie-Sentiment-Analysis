"""Aggregate analytics computed over the prediction history."""

from __future__ import annotations

from typing import Dict

import pandas as pd


def compute_analytics(history_df: pd.DataFrame) -> Dict:
    """Compute summary analytics over the prediction history.

    Args:
        history_df: DataFrame with at least 'Sentiment' and 'Confidence' columns.

    Returns:
        Dictionary of summary statistics.
    """
    if history_df.empty:
        return {
            "total": 0,
            "positive": 0,
            "negative": 0,
            "avg_confidence": 0.0,
            "max_confidence": 0.0,
            "min_confidence": 0.0,
        }

    positive = int((history_df["Sentiment"] == "Positive").sum())
    negative = int((history_df["Sentiment"] == "Negative").sum())

    return {
        "total": len(history_df),
        "positive": positive,
        "negative": negative,
        "avg_confidence": round(history_df["Confidence"].mean(), 2),
        "max_confidence": round(history_df["Confidence"].max(), 2),
        "min_confidence": round(history_df["Confidence"].min(), 2),
    }