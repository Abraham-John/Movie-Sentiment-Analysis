"""Plotly chart builders used across the dashboard."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

ACCENT_POSITIVE = "#00D9C0"
ACCENT_NEGATIVE = "#FF6B6B"
ACCENT_PRIMARY = "#8C6CFF"
ACCENT_AMBER = "#FFB020"
BG_TRANSPARENT = "rgba(0,0,0,0)"


def confidence_gauge(confidence_pct: float, is_positive: bool) -> go.Figure:
    """Build a gauge chart showing prediction confidence."""
    color = ACCENT_POSITIVE if is_positive else ACCENT_NEGATIVE

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=confidence_pct,
            number={"suffix": "%", "font": {"size": 36, "color": "white"}},
            title={"text": "Prediction Confidence", "font": {"size": 16, "color": "#c9c9d6"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#8a8aa3"},
                "bar": {"color": color, "thickness": 0.28},
                "bgcolor": BG_TRANSPARENT,
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 65], "color": "rgba(255,255,255,0.06)"},
                    {"range": [65, 85], "color": "rgba(255,255,255,0.10)"},
                    {"range": [85, 100], "color": "rgba(255,255,255,0.14)"},
                ],
            },
        )
    )
    fig.update_layout(
        paper_bgcolor=BG_TRANSPARENT,
        plot_bgcolor=BG_TRANSPARENT,
        font={"color": "white"},
        height=260,
        margin=dict(l=20, r=20, t=50, b=10),
    )
    return fig


def probability_bar_chart(positive_prob: float, negative_prob: float) -> go.Figure:
    """Horizontal bar chart comparing positive vs. negative probability."""
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=["Positive"],
            x=[positive_prob * 100],
            orientation="h",
            marker_color=ACCENT_POSITIVE,
            text=[f"{positive_prob * 100:.2f}%"],
            textposition="inside",
            name="Positive",
        )
    )
    fig.add_trace(
        go.Bar(
            y=["Negative"],
            x=[negative_prob * 100],
            orientation="h",
            marker_color=ACCENT_NEGATIVE,
            text=[f"{negative_prob * 100:.2f}%"],
            textposition="inside",
            name="Negative",
        )
    )
    fig.update_layout(
        paper_bgcolor=BG_TRANSPARENT,
        plot_bgcolor=BG_TRANSPARENT,
        font={"color": "white"},
        xaxis={"range": [0, 100], "showgrid": False, "ticksuffix": "%"},
        yaxis={"showgrid": False},
        showlegend=False,
        height=180,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    return fig


def history_pie_chart(history_df: pd.DataFrame) -> go.Figure:
    """Pie chart of positive vs negative prediction counts."""
    counts = history_df["Sentiment"].value_counts()
    fig = go.Figure(
        go.Pie(
            labels=counts.index,
            values=counts.values,
            hole=0.55,
            marker_colors=[
                ACCENT_POSITIVE if label == "Positive" else ACCENT_NEGATIVE
                for label in counts.index
            ],
        )
    )
    fig.update_layout(
        paper_bgcolor=BG_TRANSPARENT,
        font={"color": "white"},
        height=300,
        margin=dict(l=10, r=10, t=30, b=10),
        title="Sentiment Split",
    )
    return fig


def history_bar_chart(history_df: pd.DataFrame) -> go.Figure:
    """Bar chart of average confidence per sentiment class."""
    grouped = history_df.groupby("Sentiment")["Confidence"].mean()
    fig = go.Figure(
        go.Bar(
            x=grouped.index,
            y=grouped.values,
            marker_color=[
                ACCENT_POSITIVE if label == "Positive" else ACCENT_NEGATIVE
                for label in grouped.index
            ],
            text=[f"{v:.1f}%" for v in grouped.values],
            textposition="outside",
        )
    )
    fig.update_layout(
        paper_bgcolor=BG_TRANSPARENT,
        plot_bgcolor=BG_TRANSPARENT,
        font={"color": "white"},
        yaxis={"range": [0, 105], "title": "Avg Confidence %"},
        height=300,
        margin=dict(l=10, r=10, t=30, b=10),
        title="Average Confidence by Sentiment",
    )
    return fig


def history_timeline(history_df: pd.DataFrame) -> go.Figure:
    """Line chart showing confidence trend over successive predictions."""
    fig = go.Figure(
        go.Scatter(
            x=list(range(1, len(history_df) + 1)),
            y=history_df["Confidence"],
            mode="lines+markers",
            line=dict(color=ACCENT_PRIMARY, width=3),
            marker=dict(size=7, color=ACCENT_AMBER),
        )
    )
    fig.update_layout(
        paper_bgcolor=BG_TRANSPARENT,
        plot_bgcolor=BG_TRANSPARENT,
        font={"color": "white"},
        xaxis={"title": "Prediction #", "showgrid": False},
        yaxis={"title": "Confidence %", "range": [0, 105]},
        height=320,
        margin=dict(l=10, r=10, t=30, b=10),
        title="Confidence Trend Over Time",
    )
    return fig