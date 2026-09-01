import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.predict import load_predictor
from src.scraper import scrape_google_news
from src.storage import clear_records, get_records, initialize_database, save_prediction

PROJECT_ROOT = Path(__file__).resolve().parent
METRICS_PATH = PROJECT_ROOT / "models" / "model_metrics.json"

st.set_page_config(
    page_title="SENTI-SCOPE",
    page_icon="S",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root {
        --navy: #10233f;
        --navy-light: #18395f;
        --blue: #1264d6;
        --teal: #0a9c8d;
        --sky: #eaf3ff;
        --page: #f4f8fd;
        --white: #ffffff;
        --border: #d2dfef;
        --muted: #4e647d;
        --green: #147b48;
        --green-bg: #e9f9ef;
        --red: #bd2540;
        --red-bg: #fff0f2;
    }

    .stApp {
        background: linear-gradient(135deg, #f8fbff 0%, #eaf3ff 48%, #f9fcff 100%);
        color: var(--navy);
        font-family: "DM Sans", sans-serif;
    }

    [data-testid="stHeader"] {
        background: rgba(248, 251, 255, 0.92);
        backdrop-filter: blur(12px);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d2747 0%, #174574 100%);
        border-right: 1px solid #2d5d90;
    }

    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    h1, h2, h3 {
        font-family: "Space Grotesk", sans-serif !important;
        color: var(--navy) !important;
    }

    p, label, span {
        font-family: "DM Sans", sans-serif;
    }

    .hero {
        border-radius: 22px;
        padding: 2.1rem 2.3rem;
        margin-bottom: 1.4rem;
        color: #ffffff;
        background: linear-gradient(115deg, #0b4388 0%, #1264d6 53%, #0a9c8d 100%);
        box-shadow: 0 14px 32px rgba(18, 100, 214, 0.22);
    }

    .hero h1 {
        margin: 0;
        color: #ffffff !important;
        font-size: 2.6rem;
        letter-spacing: -0.045em;
    }

    .hero p {
        margin: 0.48rem 0 0;
        color: #eef8ff !important;
        max-width: 800px;
        font-size: 1rem;
        line-height: 1.55;
    }

    .badge {
        display: inline-block;
        margin-bottom: 0.85rem;
        padding: 0.42rem 0.85rem;
        border: 1px solid rgba(255, 255, 255, 0.47);
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.16);
        color: #ffffff !important;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
    }

    .glow-dot {
        display: inline-block;
        width: 9px;
        height: 9px;
        margin-right: 7px;
        border-radius: 50%;
        background: #86ffbf;
        box-shadow: 0 0 0 4px rgba(134, 255, 191, 0.25);
    }

    .metric-card {
        min-height: 137px;
        padding: 1.05rem 1.1rem;
        border: 1px solid var(--border);
        border-radius: 17px;
        background: var(--white);
        box-shadow: 0 8px 21px rgba(16, 35, 63, 0.08);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 14px 29px rgba(18, 100, 214, 0.17);
    }

    .metric-icon {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 36px;
        height: 36px;
        border-radius: 10px;
        background: #dcecff;
        color: #0d4d9e;
        font-size: 0.78rem;
        font-weight: 800;
    }

    .metric-label {
        margin-top: 0.72rem;
        color: var(--muted);
        font-size: 0.75rem;
        font-weight: 800;
        letter-spacing: 0.03em;
    }

    .metric-value {
        margin-top: 0.22rem;
        color: var(--navy);
        font-family: "Space Grotesk", sans-serif;
        font-size: 1.7rem;
        font-weight: 700;
    }

    .panel {
        padding: 1.2rem;
        border: 1px solid var(--border);
        border-radius: 18px;
        background: #ffffff;
        box-shadow: 0 8px 22px rgba(16, 35, 63, 0.07);
    }

    .panel h3,
    .panel p {
        color: var(--navy) !important;
    }

    .section-kicker {
        color: #0e5ebc;
        font-size: 0.73rem;
        font-weight: 800;
        letter-spacing: 0.11em;
    }

    .small-muted {
        color: var(--muted) !important;
        font-size: 0.9rem;
        line-height: 1.55;
    }

    .status-line {
        margin: 0.6rem 0 1.35rem;
        padding: 0.88rem 1rem;
        border: 1px solid #a4debf;
        border-radius: 13px;
        background: #eafaf1;
        color: #155d38 !important;
    }

    .result-positive,
    .result-negative {
        margin-top: 1rem;
        padding: 1.45rem;
        border: 1px solid;
        border-radius: 17px;
    }

    .result-positive {
        border-color: #95d8b0;
        background: var(--green-bg);
    }

    .result-negative {
        border-color: #efadb9;
        background: var(--red-bg);
    }

    .sentiment-title {
        margin: 0.2rem 0;
        font-family: "Space Grotesk", sans-serif;
        font-size: 1.9rem;
        font-weight: 700;
    }

    .result-positive .sentiment-title {
        color: var(--green) !important;
    }

    .result-negative .sentiment-title {
        color: var(--red) !important;
    }

    [data-testid="stButton"] button {
        min-height: 43px;
        border: none !important;
        border-radius: 10px !important;
        background: linear-gradient(90deg, #0e55b4, #1477d1) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        transition: transform 0.2s ease, filter 0.2s ease !important;
    }

    [data-testid="stButton"] button:hover {
        transform: translateY(-2px);
        filter: brightness(1.08);
    }

    [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea {
        border: 1px solid #b8cae0 !important;
        border-radius: 10px !important;
        background: #ffffff !important;
        color: var(--navy) !important;
    }

    [data-testid="stTextInput"] label,
    [data-testid="stTextArea"] label,
    [data-testid="stSelectbox"] label,
    [data-testid="stSlider"] label {
        color: var(--navy) !important;
        font-weight: 700 !important;
    }

    [data-baseweb="select"] > div {
        border-color: #b8cae0 !important;
        background: #ffffff !important;
        color: var(--navy) !important;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 13px;
        overflow: hidden;
    }

    div[data-testid="stAlert"] {
        border-radius: 12px;
    }

    @media (max-width: 700px) {
        .hero {
            padding: 1.45rem;
        }

        .hero h1 {
            font-size: 2rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_predictor():
    return load_predictor()


@st.cache_data(ttl=5)
def load_records():
    return get_records()


def get_metrics():
    if not METRICS_PATH.exists():
        return None

    with open(METRICS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def show_hero(title, subtitle):
    st.markdown(
        f"""
        <div class="hero">
            <div class="badge"><span class="glow-dot"></span>LIVE SENTIMENT INTELLIGENCE</div>
            <h1>{title}</h1>
            <p>{subtitle}</p>
            <p style="font-size:0.86rem; margin-top:0.72rem;">
                Real-Time Sentiment Analysis System with Web Scraping
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric(icon, label, value):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-icon">{icon}</div>
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def predict_and_store(text, source, topic=""):
    result = get_predictor().predict_one(text)

    save_prediction(
        source=source,
        text=result["original_text"],
        sentiment=result["sentiment"],
        confidence=result["confidence"] or 0.0,
        topic=topic,
    )

    st.cache_data.clear()
    return result


def display_prediction(result):
    sentiment = result["sentiment"]
    confidence = result["confidence"]

    css_class = (
        "result-positive"
        if sentiment == "POSITIVE"
        else "result-negative"
    )

    confidence_text = (
        f"{confidence * 100:.2f}%"
        if confidence is not None
        else "Not available"
    )

    st.markdown(
        f"""
        <div class="{css_class}">
            <div class="small-muted">MODEL PREDICTION COMPLETE</div>
            <div class="sentiment-title">{sentiment}</div>
            <div class="small-muted">
                Confidence score: <b>{confidence_text}</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if result["probabilities"]:
        probability_df = pd.DataFrame(
            {
                "Sentiment": list(result["probabilities"].keys()),
                "Probability": [
                    score * 100
                    for score in result["probabilities"].values()
                ],
            }
        )

        fig = px.bar(
            probability_df,
            x="Sentiment",
            y="Probability",
            color="Sentiment",
            text="Probability",
            color_discrete_map={
                "POSITIVE": "#168c52",
                "NEGATIVE": "#d92d4c",
            },
        )

        fig.update_traces(
            texttemplate="%{text:.2f}%",
            textposition="outside",
        )

        fig.update_layout(
            height=330,
            yaxis_range=[0, 105],
            yaxis_title="Probability (%)",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#ffffff",
            font_color="#10233f",
            showlegend=False,
            margin=dict(l=10, r=10, t=25, b=10),
        )

        fig.update_xaxes(
            gridcolor="#e6edf6",
            title_font_color="#10233f",
            tickfont_color="#10233f",
        )

        fig.update_yaxes(
            gridcolor="#e6edf6",
            title_font_color="#10233f",
            tickfont_color="#10233f",
        )

        st.plotly_chart(fig, use_container_width=True)


def apply_filters(records, days=30, source="All", sentiment="All", keyword=""):
    if records.empty:
        return records

    filtered = records.copy()

    filtered["timestamp"] = pd.to_datetime(
        filtered["timestamp"],
        utc=True,
        errors="coerce",
    )

    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
    filtered = filtered[filtered["timestamp"] >= cutoff]

    if source != "All":
        filtered = filtered[filtered["source"] == source]

    if sentiment != "All":
        filtered = filtered[filtered["sentiment"] == sentiment]

    if keyword.strip():
        term = keyword.lower()

        filtered = filtered[
            filtered["topic"]
            .fillna("")
            .str.lower()
            .str.contains(term, na=False)
            |
            filtered["text"]
            .fillna("")
            .str.lower()
            .str.contains(term, na=False)
        ]

    return filtered


def style_chart(fig, title):
    fig.update_layout(
        title=title,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        font_color="#10233f",
        title_font_color="#10233f",
        legend_title_font_color="#10233f",
        xaxis=dict(
            gridcolor="#e6edf6",
            linecolor="#c7d6e8",
            tickfont_color="#10233f",
        ),
        yaxis=dict(
            gridcolor="#e6edf6",
            linecolor="#c7d6e8",
            tickfont_color="#10233f",
        ),
        margin=dict(l=10, r=10, t=52, b=10),
    )

    return fig


def dashboard():
    show_hero(
        "SENTI-SCOPE",
        "Monitor real-time positive and negative sentiment from manual text and public web content.",
    )

    records = load_records()
    metrics = get_metrics()

    total = len(records)

    positives = (
        int((records["sentiment"] == "POSITIVE").sum())
        if total
        else 0
    )

    negatives = (
        int((records["sentiment"] == "NEGATIVE").sum())
        if total
        else 0
    )

    positive_pct = positives * 100 / total if total else 0
    negative_pct = negatives * 100 / total if total else 0
    accuracy = metrics["production_accuracy"] * 100 if metrics else 0

    if metrics:
        st.markdown(
            f"""
            <div class="status-line">
                <span class="glow-dot"></span>
                Production model online: <b>{metrics["production_model"]}</b>
                &nbsp; | &nbsp; Real evaluation accuracy: <b>{accuracy:.2f}%</b>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.warning("Model artifacts not found. Run: python -m src.train_model")

    cards = st.columns(6)

    values = [
        ("#", "TOTAL ANALYZED", total),
        ("+", "POSITIVE POSTS", positives),
        ("-", "NEGATIVE POSTS", negatives),
        ("UP", "POSITIVE RATE", f"{positive_pct:.1f}%"),
        ("DN", "NEGATIVE RATE", f"{negative_pct:.1f}%"),
        ("AI", "MODEL ACCURACY", f"{accuracy:.2f}%" if metrics else "N/A"),
    ]

    for col, item in zip(cards, values):
        with col:
            render_metric(*item)

    st.write("")

    if records.empty:
        st.markdown(
            """
            <div class="panel">
                <div class="section-kicker">READY FOR ANALYSIS</div>
                <h3>Start with a real sentiment signal</h3>
                <p class="small-muted">
                    Go to Analyze Text for direct model predictions, or Live Web Analysis
                    to collect public news text and analyze it.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    left, right = st.columns([1, 1.35])

    with left:
        counts = records["sentiment"].value_counts().reset_index()
        counts.columns = ["Sentiment", "Count"]

        pie = px.pie(
            counts,
            names="Sentiment",
            values="Count",
            hole=0.66,
            color="Sentiment",
            color_discrete_map={
                "POSITIVE": "#168c52",
                "NEGATIVE": "#d92d4c",
            },
        )

        pie.update_traces(
            textinfo="percent+label",
            textfont_color="#10233f",
            marker=dict(line=dict(color="#ffffff", width=3)),
        )

        style_chart(pie, "Sentiment Distribution")
        st.plotly_chart(pie, use_container_width=True)

    with right:
        trend = records.copy()
        trend["date"] = trend["timestamp"].dt.date

        trend = trend.groupby(
            ["date", "sentiment"]
        ).size().reset_index(name="count")

        line = px.area(
            trend,
            x="date",
            y="count",
            color="sentiment",
            markers=True,
            color_discrete_map={
                "POSITIVE": "#168c52",
                "NEGATIVE": "#d92d4c",
            },
        )

        line.update_traces(line=dict(width=3))
        style_chart(line, "Live Sentiment Activity")
        st.plotly_chart(line, use_container_width=True)

    st.markdown(
        '<div class="section-kicker">LATEST PREDICTIONS</div>',
        unsafe_allow_html=True,
    )

    st.subheader("Recent analyzed text")

    recent = records[
        ["timestamp", "source", "topic", "text", "sentiment", "confidence"]
    ].head(12).copy()

    recent["timestamp"] = recent["timestamp"].dt.strftime(
        "%d %b %Y, %I:%M %p"
    )

    recent["confidence"] = (
        recent["confidence"] * 100
    ).round(2).astype(str) + "%"

    st.dataframe(
        recent,
        use_container_width=True,
        hide_index=True,
        column_config={
            "text": st.column_config.TextColumn(
                "Text",
                width="large",
            ),
            "confidence": st.column_config.TextColumn(
                "Confidence",
            ),
        },
    )


def analyze_text():
    show_hero(
        "Analyze Text",
        "Enter a review, comment, headline, or public text and receive a real model prediction.",
    )

    left, right = st.columns([1.35, 0.65])

    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)

        st.markdown(
            '<div class="section-kicker">MANUAL ANALYSIS</div>',
            unsafe_allow_html=True,
        )

        st.subheader("What does this text feel like?")

        example_type = st.selectbox(
            "Quick faculty demo examples",
            [
                "Custom text",
                "Positive product feedback",
                "Negative service feedback",
                "Positive technology opinion",
                "Negative technology opinion",
            ],
        )

        sample_texts = {
            "Custom text": "",
            "Positive product feedback": (
                "I absolutely love this product! It is fast, reliable, "
                "and worth every rupee."
            ),
            "Negative service feedback": (
                "This is the worst service ever. The support team did "
                "not solve my problem."
            ),
            "Positive technology opinion": (
                "This new technology makes daily work easier and more efficient."
            ),
            "Negative technology opinion": (
                "The latest update is frustrating, slow, and full of problems."
            ),
        }

        text = st.text_area(
            "Text to analyze",
            value=sample_texts[example_type],
            height=235,
            placeholder="Type or paste text here...",
        )

        if st.button(
            "ANALYZE SENTIMENT NOW",
            type="primary",
            use_container_width=True,
        ):
            try:
                result = predict_and_store(text, "Manual Input")
                st.session_state["latest_prediction"] = result
            except (ValueError, FileNotFoundError) as error:
                st.warning(str(error))
            except Exception:
                st.error(
                    "Unable to analyze this text. Confirm that trained model files exist."
                )

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown(
            """
            <div class="panel">
                <div class="section-kicker">HOW IT WORKS</div>
                <h3>One real ML pipeline</h3>
                <p class="small-muted">
                    1. Input text is cleaned using the same preprocessing used in training.
                </p>
                <p class="small-muted">
                    2. TF-IDF converts text into numeric features.
                </p>
                <p class="small-muted">
                    3. The saved Logistic Regression model returns sentiment and confidence.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if "latest_prediction" in st.session_state:
        st.write("")
        display_prediction(st.session_state["latest_prediction"])


def live_web():
    show_hero(
        "Live Web Analysis",
        "Fetch public Google News RSS headlines and snippets, then analyze every result with your saved model.",
    )

    st.markdown('<div class="panel">', unsafe_allow_html=True)

    first, second = st.columns([3, 1])

    with first:
        topic = st.text_input(
            "Topic or keyword",
            placeholder="technology, education, smartphones, electric vehicles...",
        )

    with second:
        max_items = st.selectbox(
            "Articles",
            [5, 10, 15],
            index=2,
        )

    if st.button(
        "FETCH AND ANALYZE LIVE WEB DATA",
        type="primary",
        use_container_width=True,
    ):
        try:
            progress = st.progress(
                0,
                text="Connecting to public Google News RSS...",
            )

            scraped_items = scrape_google_news(
                topic,
                max_items=max_items,
            )

            output = []

            for index, item in enumerate(scraped_items, start=1):
                progress.progress(
                    int(index / len(scraped_items) * 100),
                    text=f"Analyzing article {index} of {len(scraped_items)}...",
                )

                result = predict_and_store(
                    item["text"],
                    item["source"],
                    topic.strip(),
                )

                output.append(
                    {
                        "Source": item["source"],
                        "Headline / Public Snippet": item["text"],
                        "Sentiment": result["sentiment"],
                        "Confidence": (
                            f"{result['confidence'] * 100:.2f}%"
                            if result["confidence"] is not None
                            else "N/A"
                        ),
                        "Collected At": datetime.now(
                            timezone.utc
                        ).strftime("%d %b %Y, %I:%M %p UTC"),
                    }
                )

            progress.empty()
            st.session_state["web_results"] = pd.DataFrame(output)
            st.session_state["web_topic"] = topic.strip()

        except (ValueError, RuntimeError, FileNotFoundError) as error:
            st.error(str(error))
        except Exception as error:
            st.error(
                "Live web analysis could not be completed. Ensure internet is available "
                "and run: python -m pip install lxml"
            )
            st.caption(f"Technical detail: {error}")

    st.markdown("</div>", unsafe_allow_html=True)

    if "web_results" not in st.session_state:
        return

    results = st.session_state["web_results"]

    total = len(results)
    positives = int((results["Sentiment"] == "POSITIVE").sum())
    negatives = int((results["Sentiment"] == "NEGATIVE").sum())

    st.write("")

    st.markdown(
        f'<div class="section-kicker">LIVE RESULTS: {st.session_state.get("web_topic", "")}</div>',
        unsafe_allow_html=True,
    )

    st.subheader("Public web sentiment snapshot")

    cards = st.columns(5)

    values = [
        ("#", "TEXTS ANALYZED", total),
        ("+", "POSITIVE", positives),
        ("-", "NEGATIVE", negatives),
        ("UP", "POSITIVE RATE", f"{positives * 100 / total:.1f}%"),
        ("DN", "NEGATIVE RATE", f"{negatives * 100 / total:.1f}%"),
    ]

    for col, item in zip(cards, values):
        with col:
            render_metric(*item)

    counts = results["Sentiment"].value_counts().reset_index()
    counts.columns = ["Sentiment", "Count"]

    fig = px.bar(
        counts,
        x="Sentiment",
        y="Count",
        color="Sentiment",
        text="Count",
        color_discrete_map={
            "POSITIVE": "#168c52",
            "NEGATIVE": "#d92d4c",
        },
    )

    fig.update_traces(textposition="outside")
    style_chart(fig, "Live Web Sentiment Counts")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        results,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Headline / Public Snippet": st.column_config.TextColumn(
                width="large"
            )
        },
    )


def trends():
    show_hero(
        "Sentiment Trends",
        "Explore your actual saved predictions with filters for time, source, topic, and sentiment.",
    )

    records = load_records()

    if records.empty:
        st.info(
            "No stored records yet. Use Analyze Text or Live Web Analysis first."
        )
        return

    st.markdown('<div class="panel">', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        days = st.selectbox(
            "Time range",
            [1, 7, 30, 90, 365],
            index=2,
        )

    with c2:
        sources = ["All"] + sorted(
            records["source"].dropna().unique().tolist()
        )

        source = st.selectbox("Data source", sources)

    with c3:
        sentiment = st.selectbox(
            "Sentiment filter",
            ["All", "POSITIVE", "NEGATIVE"],
        )

    with c4:
        keyword = st.text_input("Topic or keyword")

    st.markdown("</div>", unsafe_allow_html=True)

    filtered = apply_filters(
        records,
        days=days,
        source=source,
        sentiment=sentiment,
        keyword=keyword,
    )

    if filtered.empty:
        st.warning("No actual predictions match these filters.")
        return

    filtered["date"] = filtered["timestamp"].dt.date

    grouped = filtered.groupby(
        ["date", "sentiment"]
    ).size().reset_index(name="count")

    trend_chart = px.line(
        grouped,
        x="date",
        y="count",
        color="sentiment",
        markers=True,
        color_discrete_map={
            "POSITIVE": "#168c52",
            "NEGATIVE": "#d92d4c",
        },
    )

    trend_chart.update_traces(line=dict(width=4))
    style_chart(trend_chart, "Filtered Sentiment Trend")
    st.plotly_chart(trend_chart, use_container_width=True)

    source_counts = filtered.groupby(
        ["source", "sentiment"]
    ).size().reset_index(name="count")

    source_chart = px.bar(
        source_counts,
        x="source",
        y="count",
        color="sentiment",
        barmode="group",
        color_discrete_map={
            "POSITIVE": "#168c52",
            "NEGATIVE": "#d92d4c",
        },
    )

    style_chart(source_chart, "Sentiment by Source")
    st.plotly_chart(source_chart, use_container_width=True)

    st.dataframe(
        filtered[
            ["timestamp", "source", "topic", "text", "sentiment", "confidence"]
        ],
        use_container_width=True,
        hide_index=True,
    )


def topic_analysis():
    show_hero(
        "Topic Intelligence",
        "Search actual stored predictions by topic or keyword and understand sentiment around a subject.",
    )

    records = load_records()

    if records.empty:
        st.info(
            "No stored data yet. Use Live Web Analysis to create topic-based records."
        )
        return

    topic = st.text_input(
        "Search topic or keyword",
        placeholder="technology",
    )

    if not topic.strip():
        st.markdown(
            """
            <div class="panel">
                <div class="section-kicker">TOPIC SEARCH</div>
                <h3>Enter a keyword to begin</h3>
                <p class="small-muted">
                    Use the same topic you entered in Live Web Analysis, for example technology.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    filtered = apply_filters(
        records,
        days=3650,
        keyword=topic,
    )

    if filtered.empty:
        st.warning(
            "No real stored predictions match this topic or keyword."
        )
        return

    total = len(filtered)

    positives = int(
        (filtered["sentiment"] == "POSITIVE").sum()
    )

    negatives = int(
        (filtered["sentiment"] == "NEGATIVE").sum()
    )

    cards = st.columns(3)

    values = [
        ("#", "RELATED TEXTS", total),
        ("+", "POSITIVE RATE", f"{positives * 100 / total:.1f}%"),
        ("-", "NEGATIVE RATE", f"{negatives * 100 / total:.1f}%"),
    ]

    for col, item in zip(cards, values):
        with col:
            render_metric(*item)

    counts = filtered["sentiment"].value_counts().reset_index()
    counts.columns = ["Sentiment", "Count"]

    fig = px.pie(
        counts,
        names="Sentiment",
        values="Count",
        hole=0.60,
        color="Sentiment",
        color_discrete_map={
            "POSITIVE": "#168c52",
            "NEGATIVE": "#d92d4c",
        },
    )

    fig.update_traces(textinfo="percent+label", textfont_color="#10233f")
    style_chart(fig, f"Topic Sentiment: {topic}")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        filtered[
            ["timestamp", "source", "topic", "text", "sentiment", "confidence"]
        ],
        use_container_width=True,
        hide_index=True,
    )


def performance():
    show_hero(
        "Model Performance",
        "Compare Naive Bayes and Logistic Regression using your actual Sentiment140 evaluation output.",
    )

    metrics = get_metrics()

    if not metrics:
        st.warning(
            "Metrics not found. Run this command first: python -m src.train_model"
        )
        return

    st.markdown(
        f"""
        <div class="status-line">
            <span class="glow-dot"></span>
            ACTIVE PRODUCTION MODEL: <b>{metrics["production_model"]}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

    rows = []

    for model_name, values in metrics["models"].items():
        rows.append(
            {
                "Model": model_name,
                "Accuracy": values["accuracy"] * 100,
                "Precision": values["precision"] * 100,
                "Recall": values["recall"] * 100,
                "F1-score": values["f1_score"] * 100,
            }
        )

    metrics_df = pd.DataFrame(rows)

    cards = st.columns(2)

    for col, row in zip(cards, rows):
        with col:
            status = (
                "PRODUCTION MODEL"
                if row["Model"] == metrics["production_model"]
                else "BASELINE MODEL"
            )

            st.markdown(
                f"""
                <div class="panel">
                    <div class="section-kicker">{status}</div>
                    <h3>{row["Model"]}</h3>
                    <div class="metric-value">{row["Accuracy"]:.2f}%</div>
                    <div class="small-muted">Accuracy</div>
                    <hr style="border-color:#dce5f0;">
                    <div class="small-muted">
                        Precision: <b>{row["Precision"]:.2f}%</b>
                        &nbsp; | &nbsp; Recall: <b>{row["Recall"]:.2f}%</b>
                        &nbsp; | &nbsp; F1-score: <b>{row["F1-score"]:.2f}%</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    long_df = metrics_df.melt(
        id_vars="Model",
        value_vars=["Accuracy", "Precision", "Recall", "F1-score"],
        var_name="Metric",
        value_name="Score",
    )

    fig = px.bar(
        long_df,
        x="Metric",
        y="Score",
        color="Model",
        barmode="group",
        text="Score",
        color_discrete_sequence=["#1264d6", "#0a9c8d"],
    )

    fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside",
    )

    fig.update_yaxes(range=[0, 105], title="Score (%)")
    style_chart(fig, "Actual Evaluation Metrics")
    st.plotly_chart(fig, use_container_width=True)

    selected = st.selectbox(
        "Inspect confusion matrix",
        ["Naive Bayes", "Logistic Regression"],
    )

    image_file = (
        selected.lower().replace(" ", "_")
        + "_confusion_matrix.png"
    )

    image_path = PROJECT_ROOT / "results" / image_file

    if image_path.exists():
        st.image(
            str(image_path),
            caption=f"{selected} Confusion Matrix",
            use_container_width=True,
        )
    else:
        st.info("Confusion matrix image not found. Train models again.")


def alerts():
    show_hero(
        "Sentiment Alerts",
        "Monitor real negative-sentiment levels from stored predictions and trigger a threshold-based alert.",
    )

    records = load_records()

    if records.empty:
        st.info("No analyzed records are stored yet. Alerts need real predictions.")
        return

    first, second = st.columns(2)

    with first:
        threshold = st.slider(
            "Negative sentiment alert threshold (%)",
            min_value=1,
            max_value=100,
            value=60,
        )

    with second:
        period = st.selectbox(
            "Analyze latest stored texts",
            [10, 25, 50, 100, "All"],
        )

    active = records if period == "All" else records.head(period)

    total = len(active)

    negative_count = int(
        (active["sentiment"] == "NEGATIVE").sum()
    )

    negative_pct = negative_count * 100 / total if total else 0

    cards = st.columns(3)

    values = [
        ("-", "CURRENT NEGATIVE RATE", f"{negative_pct:.2f}%"),
        ("!", "ALERT THRESHOLD", f"{threshold}%"),
        ("#", "TEXTS EVALUATED", total),
    ]

    for col, item in zip(cards, values):
        with col:
            render_metric(*item)

    if negative_pct >= threshold:
        st.markdown(
            f"""
            <div class="result-negative">
                <div class="small-muted">ALERT STATUS</div>
                <div class="sentiment-title">NEGATIVE SENTIMENT ALERT</div>
                <div class="small-muted">
                    Actual negative sentiment is <b>{negative_pct:.2f}%</b>, meeting or exceeding
                    your configured threshold of <b>{threshold}%</b>.
                    Reason: <b>{negative_count}</b> out of <b>{total}</b> real stored texts
                    are predicted NEGATIVE.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="result-positive">
                <div class="small-muted">ALERT STATUS</div>
                <div class="sentiment-title">SYSTEM STABLE</div>
                <div class="small-muted">
                    Current negative sentiment is <b>{negative_pct:.2f}%</b>, which is below
                    your configured threshold of <b>{threshold}%</b>.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def about():
    show_hero(
        "About The Project",
        "A complete real-time binary sentiment-analysis workflow for manual text and public web-news content.",
    )

    left, right = st.columns(2)

    with left:
        st.markdown(
            """
            <div class="panel">
                <div class="section-kicker">PROJECT OBJECTIVE</div>
                <h3>Real-time sentiment intelligence</h3>
                <p class="small-muted">
                    Train a classifier using Sentiment140 and use it to analyze
                    user-entered text and publicly available news headlines and snippets.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            """
            <div class="panel">
                <div class="section-kicker">SENTIMENT CLASSES</div>
                <h3>Binary classification only</h3>
                <p class="small-muted">
                    NEGATIVE maps from Sentiment140 target 0. POSITIVE maps from
                    Sentiment140 target 4. No artificial neutral label is added.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    st.markdown(
        """
        <div class="panel">
            <div class="section-kicker">END-TO-END PIPELINE</div>
            <h3>From public text to insight</h3>
            <p class="small-muted">
                Public web scraping -> local SQLite storage -> shared text preprocessing ->
                TF-IDF features -> Naive Bayes and Logistic Regression -> evaluation ->
                best-model selection -> real-time prediction -> dashboard -> alerts.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    initialize_database()

    if "nav" not in st.session_state:
        st.session_state["nav"] = "Dashboard"

    st.sidebar.markdown("## SENTI-SCOPE")
    st.sidebar.caption("REAL-TIME SENTIMENT INTELLIGENCE")
    st.sidebar.divider()

    nav_items = [
        "Dashboard",
        "Analyze Text",
        "Live Web Analysis",
        "Sentiment Trends",
        "Topic Intelligence",
        "Model Performance",
        "Alerts",
        "About Project",
    ]

    selected = st.sidebar.radio(
        "Explore System",
        nav_items,
        index=nav_items.index(st.session_state["nav"]),
    )

    st.session_state["nav"] = selected

    st.sidebar.divider()
    st.sidebar.markdown("### System Controls")

    records = load_records()
    st.sidebar.metric("Stored Predictions", len(records))

    if st.sidebar.button("Clear Stored Data", use_container_width=True):
        clear_records()
        st.cache_data.clear()
        st.success("Stored prediction data cleared.")

    st.sidebar.caption(
        "Training labels: 0 = Negative | 4 = Positive"
    )

    pages = {
        "Dashboard": dashboard,
        "Analyze Text": analyze_text,
        "Live Web Analysis": live_web,
        "Sentiment Trends": trends,
        "Topic Intelligence": topic_analysis,
        "Model Performance": performance,
        "Alerts": alerts,
        "About Project": about,
    }

    try:
        pages[selected]()
    except FileNotFoundError as error:
        st.error(str(error))
    except Exception as error:
        st.error(
            "A dashboard error occurred. Check PowerShell for the technical details."
        )
        st.caption(f"Technical detail: {error}")


if __name__ == "__main__":
    main()