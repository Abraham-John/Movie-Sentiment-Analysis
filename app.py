"""
CineSense AI — Deep Learning Powered Movie Review Analyzer.

A production-styled Streamlit dashboard around an existing, already-trained
LSTM sentiment model. The prediction pipeline (encoding, padding, model call,
0.5 threshold) is unchanged from the original app — everything here is UI,
analytics, and developer-experience layered on top.
"""

from __future__ import annotations

import io
import json
from datetime import datetime

import pandas as pd
import streamlit as st

from utils.analytics import compute_analytics
from utils.prediction import (
    confidence_level,
    generate_explanation,
    load_model,
    load_word_index,
    predict_sentiment,
)
from utils.preprocessing import estimate_emotions, get_review_stats
from utils.visualization import (
    confidence_gauge,
    history_bar_chart,
    history_pie_chart,
    history_timeline,
    probability_bar_chart,
)

# ---------------------------------------------------------------------------
# Page configuration & styling
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="CineSense AI — Movie Sentiment Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_css(path: str) -> None:
    """Inject the external stylesheet into the app."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("Stylesheet not found — running with default Streamlit styling.")


load_css("assets/style.css")

EXAMPLE_REVIEWS = {
    "😊 Positive Example": (
        "This movie was an absolute masterpiece. The acting was superb, the "
        "story was beautifully told, and I was completely captivated from "
        "start to finish. I would highly recommend it to anyone."
    ),
    "😞 Negative Example": (
        "What a disappointing waste of time. The plot was boring and "
        "predictable, the acting felt flat, and the pacing was painfully "
        "slow. I would not recommend this film to anyone."
    ),
    "😐 Mixed Example": (
        "The visuals were stunning and the soundtrack was great, but the "
        "story felt confusing and some of the acting was pretty weak. "
        "Overall it had its moments but also dragged in places."
    ),
}

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------

st.session_state.setdefault("history", [])
st.session_state.setdefault("review_input", "")


def set_example_text(text: str) -> None:
    st.session_state.review_input = text


def clear_review_text() -> None:
    st.session_state.review_input = ""


# ---------------------------------------------------------------------------
# Cached resources (model + word index load once per server process)
# ---------------------------------------------------------------------------

try:
    model = load_model()
    word_index = load_word_index()
    model_load_error = None
except Exception as exc:  # noqa: BLE001 — surfaced to the user, not swallowed
    model, word_index = None, None
    model_load_error = str(exc)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown('<div class="sidebar-logo">🎬 CineSense AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="small-muted" style="text-align:center;">Deep Learning Sentiment Engine</p>',
        unsafe_allow_html=True,
    )
    st.divider()

    page = st.radio(
        "Navigation",
        ["🏠 Analyzer", "📜 History", "📊 Analytics", "ℹ️ About"],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("**💡 Quick Tips**")
    st.markdown(
        "- Write at least a full sentence for best accuracy\n"
        "- Try the example buttons below to see it in action\n"
        "- Check the *AI Explanation* panel to see why a prediction was made"
    )

    st.markdown("**🎯 Example Reviews**")
    for label, text in EXAMPLE_REVIEWS.items():
        st.button(label, on_click=set_example_text, args=(text,), key=f"ex_{label}")

    with st.expander("📁 About This Project"):
        st.write(
            "CineSense AI predicts whether a movie review expresses positive "
            "or negative sentiment, using a Recurrent Neural Network trained "
            "on the IMDB movie review dataset."
        )

    with st.expander("🧑‍💻 Developer Info"):
        st.markdown(
                    "Built as an end-to-end deep learning + Streamlit portfolio project\n"
                    "- **Creator:** Abraham John\n"
                    "- **GitHub:** https://github.com/Abraham-John\n"
                )

    with st.expander("🛠️ Technology Stack"):
        st.markdown(
            "- **Model:** TensorFlow / Keras LSTM\n"
            "- **Frontend:** Streamlit\n"
            "- **Charts:** Plotly\n"
            "- **Data:** Pandas / NumPy"
        )

    with st.expander("🧠 Model Information"):
        st.markdown(
            "- **Architecture:** Simple RNN / LSTM\n"
            "- **Vocabulary size:** 10,000 words\n"
            "- **Sequence length:** 500 tokens\n"
            "- **Output:** Binary sentiment probability"
        )

    with st.expander("📚 Dataset Information"):
        st.markdown(
            "- **Source:** Keras IMDB dataset\n"
            "- **Size:** 50,000 labeled movie reviews\n"
            "- **Classes:** Positive / Negative"
        )

    st.markdown(
        '<div class="sidebar-footer">Built with Streamlit + TensorFlow<br>'
        "© 2026 CineSense AI</div>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Hero section
# ---------------------------------------------------------------------------

st.markdown('<div class="hero-title">🎬 CineSense AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Deep Learning Powered Movie Review Analysis</div>',
    unsafe_allow_html=True,
)


st.markdown(
    '<p class="hero-description">Paste any movie review below and CineSense AI will '
    
    "predict its sentiment using an LSTM neural network trained on 50,000 IMDB "
    "reviews — complete with confidence scores, probability breakdowns, and a "
    "plain-language explanation of the result.</p>",
    unsafe_allow_html=True,
)
st.markdown('<div class="film-divider"></div>', unsafe_allow_html=True)

if model_load_error:
    st.error(
        "⚠️ Could not load the trained model.\n\n"
        f"`{model_load_error}`\n\n"
        "Make sure `simple_rnn_imdb.h5` is placed in `models/` or the project root."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Page: Analyzer
# ---------------------------------------------------------------------------

if page == "🏠 Analyzer":

    input_col, meta_col = st.columns([3, 1])

    with input_col:
        st.text_area(
            "📝 Enter a movie review",
            key="review_input",
            placeholder="Example: The movie was amazing, the story was beautiful...",
            height=170,
        )
        analyze_clicked = st.button(
            "🚀 Analyze Sentiment",
            use_container_width=True,
            type="primary"
        )

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        st.button(
            "Clear Review",
            on_click=clear_review_text,
            type="secondary"
        )

    with meta_col:
        live_stats = get_review_stats(st.session_state.review_input)
        st.markdown(
            f"""
            <div class="glass-card">
                <div class="small-muted">Words</div>
                <div class="stat-value">{live_stats['word_count']}</div>
                <div class="small-muted" style="margin-top:10px;">Characters</div>
                <div class="stat-value">{live_stats['char_count']}</div>
                <div class="small-muted" style="margin-top:10px;">Est. Reading Time</div>
                <div class="stat-value">{live_stats['reading_time_seconds']:.0f}s</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if analyze_clicked:
        review_text = st.session_state.review_input

        if review_text.strip() == "":
            st.warning("Please enter a review before analyzing.")
        else:
            with st.spinner("Analyzing sentiment..."):
                result = predict_sentiment(review_text, model, word_index)
                stats = get_review_stats(review_text)
                emotions = estimate_emotions(review_text)
                explanations = generate_explanation(result, stats)

            st.toast("Prediction complete!", icon="✅")

            st.session_state.history.append(
                {
                    "Review": review_text,
                    "Sentiment": result["sentiment"],
                    "Confidence": round(result["confidence"] * 100, 2),
                    "PositiveProb": round(result["positive_probability"] * 100, 2),
                    "NegativeProb": round(result["negative_probability"] * 100, 2),
                    "ProcessingTimeMs": result["processing_time_ms"],
                    "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
            )

            st.markdown('<div class="film-divider"></div>', unsafe_allow_html=True)

            # ---- Prediction dashboard -----------------------------------
            is_positive = result["sentiment"] == "Positive"
            level = confidence_level(result["confidence"])
            badge_class = {"High": "badge-high", "Medium": "badge-medium", "Low": "badge-low"}[level]

            col1, col2, col3 = st.columns([1.2, 1, 1])

            with col1:
                label_class = "result-label-positive" if is_positive else "result-label-negative"
                st.markdown(
                    f"""
                    <div class="glass-card result-card">
                        <div class="result-emoji">{result['emoji']}</div>
                        <div class="{label_class}">{result['sentiment']}</div>
                        <span class="badge {badge_class}">{level} Confidence</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with col2:
                st.markdown(
                    f"""
                    <div class="glass-card">
                        <div class="small-muted">Model Used</div>
                        <div class="stat-value" style="font-size:16px;">LSTM (IMDB)</div>
                        <div class="small-muted" style="margin-top:10px;">Processing Time</div>
                        <div class="stat-value" style="font-size:16px;">{result['processing_time_ms']} ms</div>
                        <div class="small-muted" style="margin-top:10px;">Prediction Time</div>
                        <div class="stat-value" style="font-size:16px;">{datetime.now().strftime('%H:%M:%S')}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with col3:
                st.plotly_chart(
                    confidence_gauge(result["confidence"] * 100, is_positive),
                    use_container_width=True,
                    config={"displayModeBar": False},
                )

            # ---- Probability breakdown -----------------------------------
            st.markdown("#### 📈 Probability Breakdown")
            pcol1, pcol2 = st.columns(2)
            with pcol1:
                st.markdown(
                    f"""
                    <div class="glass-card">
                        <div class="small-muted">Positive Probability</div>
                        <div class="stat-value" style="color:#00D9C0;">{result['positive_probability']*100:.2f}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with pcol2:
                st.markdown(
                    f"""
                    <div class="glass-card">
                        <div class="small-muted">Negative Probability</div>
                        <div class="stat-value" style="color:#FF6B6B;">{result['negative_probability']*100:.2f}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.plotly_chart(
                probability_bar_chart(result["positive_probability"], result["negative_probability"]),
                use_container_width=True,
                config={"displayModeBar": False},
            )

            # ---- AI explanation -------------------------------------------
            st.markdown("#### 🧩 AI Explanation")
            st.markdown(
                f'<div class="glass-card">{"".join(f"<p>• {e}</p>" for e in explanations)}</div>',
                unsafe_allow_html=True,
            )

            # ---- Review statistics -----------------------------------------
            st.markdown("#### 📊 Review Statistics")
            stat_items = [
                ("Word Count", stats["word_count"]),
                ("Character Count", stats["char_count"]),
                ("Sentence Count", stats["sentence_count"]),
                ("Avg Word Length", stats["avg_word_length"]),
                ("Longest Word", stats["longest_word"] or "—"),
                ("Reading Time", f"{stats['reading_time_seconds']:.0f}s"),
                ("Positive Words", stats["positive_word_count"]),
                ("Negative Words", stats["negative_word_count"]),
                ("Lexical Diversity", stats["lexical_diversity"]),
            ]
            stat_cols = st.columns(len(stat_items))
            for c, (label, value) in zip(stat_cols, stat_items):
                with c:
                    st.markdown(
                        f"""
                        <div class="stat-tile">
                            <div class="stat-value" style="font-size:16px;">{value}</div>
                            <div class="stat-label">{label}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            # ---- Emotion meter ---------------------------------------------
            st.markdown("#### 🎭 Emotion Meter")
            emotion_icons = {"Happy": "😊", "Angry": "😡", "Sad": "😢", "Excited": "🤩", "Neutral": "😐"}
            emo_cols = st.columns(len(emotions))
            for c, (name, score) in zip(emo_cols, emotions.items()):
                with c:
                    st.markdown(
                        f"""
                        <div class="emotion-card">
                            <div class="emotion-emoji">{emotion_icons[name]}</div>
                            <div class="emotion-score">{score}%</div>
                            <div class="emotion-name">{name}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

# ---------------------------------------------------------------------------
# Page: History
# ---------------------------------------------------------------------------

elif page == "📜 History":
    st.markdown("### 📜 Prediction History")

    if not st.session_state.history:
        st.info("No predictions yet — analyze a review to get started.")
    else:
        history_df = pd.DataFrame(st.session_state.history)

        sort_by = st.selectbox(
            "Sort by",
            ["Time (newest)", "Confidence (highest)"],
            width="stretch"
        )
        st.write("")
        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()

        st.write("")


        display_df = history_df.copy()
        if sort_by == "Time (newest)":
            display_df = display_df.iloc[::-1]
        else:
            display_df = display_df.sort_values("Confidence", ascending=False)

        for original_idx in display_df.index:
            row = history_df.loc[original_idx]
            css_class = "positive" if row["Sentiment"] == "Positive" else "negative"
            emoji = "😊" if row["Sentiment"] == "Positive" else "😞"
            preview = (row["Review"][:120] + "...") if len(row["Review"]) > 120 else row["Review"]

            hc1, hc2 = st.columns([12, 1], vertical_alignment="center")
            with hc1:
                st.markdown(
                    f"""
                    <div class="history-card {css_class}">
                        <div class="history-review-text">{emoji} {preview}</div>
                        <div class="history-meta">
                            {row['Sentiment']} · {row['Confidence']}% confidence · {row['Time']}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with hc2:
                st.write("")   
                if st.button("🗑️", key=f"del_{original_idx}", help="Delete Review"):
                    st.session_state.history.pop(original_idx)
                    st.rerun()

        st.markdown("#### ⬇️ Export History")
        e1, e2, e3, e4 = st.columns(4)

        with e1:
            st.download_button(
                "CSV",
                history_df.to_csv(index=False).encode("utf-8"),
                file_name="prediction_history.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with e2:
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                history_df.to_excel(writer, index=False, sheet_name="History")
            st.download_button(
                "Excel",
                excel_buffer.getvalue(),
                file_name="prediction_history.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        with e3:
            st.download_button(
                "JSON",
                json.dumps(st.session_state.history, indent=2).encode("utf-8"),
                file_name="prediction_history.json",
                mime="application/json",
                use_container_width=True,
            )
        with e4:
            txt_lines = [
                f"[{r['Time']}] {r['Sentiment']} ({r['Confidence']}%) — {r['Review']}"
                for r in st.session_state.history
            ]
            st.download_button(
                "TXT",
                "\n".join(txt_lines).encode("utf-8"),
                file_name="prediction_history.txt",
                mime="text/plain",
                use_container_width=True,
            )

# ---------------------------------------------------------------------------
# Page: Analytics
# ---------------------------------------------------------------------------

elif page == "📊 Analytics":
    st.markdown("### 📊 Analytics Dashboard")

    if not st.session_state.history:
        st.info("No predictions yet — analyze a review to see analytics.")
    else:
        history_df = pd.DataFrame(st.session_state.history)
        summary = compute_analytics(history_df)

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        metrics = [
            ("Total", summary["total"]),
            ("Positive", summary["positive"]),
            ("Negative", summary["negative"]),
            ("Avg Confidence", f"{summary['avg_confidence']}%"),
            ("Highest", f"{summary['max_confidence']}%"),
            ("Lowest", f"{summary['min_confidence']}%"),
        ]
        for c, (label, value) in zip([m1, m2, m3, m4, m5, m6], metrics):
            with c:
                st.markdown(
                    f"""
                    <div class="stat-tile">
                        <div class="stat-value" style="font-size:18px;">{value}</div>
                        <div class="stat-label">{label}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown('<div class="film-divider"></div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(history_pie_chart(history_df), use_container_width=True)
        with c2:
            st.plotly_chart(history_bar_chart(history_df), use_container_width=True)

        st.plotly_chart(history_timeline(history_df), use_container_width=True)

# ---------------------------------------------------------------------------
# Page: About
# ---------------------------------------------------------------------------

else:
    st.markdown("### ℹ️ About CineSense AI")
    st.markdown(
        """
        <div class="glass-card">
        <p><strong>CineSense AI</strong> is a deep learning powered sentiment analysis
        dashboard for movie reviews, built on a Recurrent Neural Network (LSTM)
        trained on the IMDB dataset of 50,000 labeled reviews.</p>
        <p>The application demonstrates an end-to-end ML product experience:
        text preprocessing, model inference, confidence scoring, explainability,
        historical tracking, and analytics — all wrapped in a polished,
        production-style interface.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### 🛠️ Technology Stack")
    tcol1, tcol2, tcol3, tcol4 = st.columns(4)
    for c, label in zip([tcol1, tcol2, tcol3, tcol4], ["TensorFlow / Keras", "Streamlit", "Plotly", "Pandas / NumPy"]):
        with c:
            st.markdown(
                f'<div class="stat-tile"><div class="stat-label">{label}</div></div>',
                unsafe_allow_html=True,
            )
