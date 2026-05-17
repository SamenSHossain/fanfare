import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.analytics import (
    SENTIMENT_GAP_THRESHOLD,
    aggregate_sentiment_over_time,
    compute_engagement_rate,
    get_top_fans,
    get_trending_topics,
    keyword_sentiment_breakdown,
    weighted_mean_sentiment,
)
from src.cleaning import clean_comments
from src.alerts import (
    CORRECTION_METHOD,
    MIN_KEYWORD_MENTIONS,
    MIN_VIDEO_COMMENTS,
    run_alerts,
)
from src.fans import FEATURE_DISPLAY, run_fan_segmentation
from src.scoring import EMOTION_LABELS, UNCERTAIN_THRESHOLD, score_comments
from src.topics import run_topic_model
from src.youtube_client import YouTubeClient

# ── Config ─────────────────────────────────────────────────────────────────────
YOUTUBE_API_KEY = "AIzaSyBHCfCa25OzyRfLXSqWZ1IPjRgVAD6DgLg"
CHANNEL_HANDLE = "jaredmccain024"

EMOTION_COLORS: dict[str, str] = {
    "joy":      "#F1C40F",
    "sadness":  "#3498DB",
    "anger":    "#E74C3C",
    "fear":     "#E67E22",
    "disgust":  "#8E44AD",
    "surprise": "#1ABC9C",
    "neutral":  "#95A5A6",
}

st.set_page_config(
    page_title="Fanfare — Jared McCain Fan Intelligence",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
[data-testid="stMetricValue"] { font-size: 1.4rem; font-weight: 700; }
.stTabs [data-baseweb="tab"] { font-size: 0.88rem; font-weight: 600; }
.insight-box { padding: 14px 18px; border-radius: 8px; margin-bottom: 6px; }
</style>
""",
    unsafe_allow_html=True,
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Fanfare")
    st.caption("YouTube Fan Intelligence")
    st.divider()

    max_videos = st.slider("Videos to analyze", 5, 100, 95, step=5)
    max_comments = st.slider("Comments per video", 20, 200, 100, step=20)

    quota_estimate = max_videos * 2 + max_videos * (max_comments // 100)
    st.caption(f"Est. quota used: ~{quota_estimate} units (10,000/day free)")

    fetch_btn = st.button("Fetch & Analyze", type="primary", use_container_width=True)

    st.divider()
    st.caption("YouTube Data API v3 · No credit card required")


# ── Cached data loader ─────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def load_data(
    max_vids: int,
    max_coms: int,
) -> tuple[dict | None, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, dict]:
    client = YouTubeClient(YOUTUBE_API_KEY)

    channel = client.get_channel_info(handle=CHANNEL_HANDLE)
    if not channel:
        return None, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}, {}

    video_ids = client.get_video_ids(channel["uploads_playlist_id"], max_vids)
    if not video_ids:
        return channel, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}, {}

    videos_df = client.get_video_details(video_ids)
    if not videos_df.empty:
        videos_df = compute_engagement_rate(videos_df)
        videos_df["published_at"] = pd.to_datetime(videos_df["published_at"])
        videos_df = videos_df.sort_values("published_at", ascending=False).reset_index(
            drop=True
        )

    raw_df = client.get_all_comments(video_ids, max_coms)
    if raw_df.empty:
        return channel, videos_df, pd.DataFrame(), pd.DataFrame(), {}, {}

    # Stage 1: clean — adds cleaned_text, is_spam, language, is_duplicate
    records = clean_comments(raw_df.to_dict("records"))
    comments_df = pd.DataFrame(records)

    # Stage 2: sentiment + emotion on clean subset only, using cleaned_text
    clean_mask = ~comments_df["is_spam"] & ~comments_df["is_duplicate"]
    if clean_mask.any():
        clean_records = comments_df[clean_mask].to_dict("records")
        scored = score_comments(clean_records, text_col="cleaned_text")
        scored_df = pd.DataFrame(scored)
        score_cols = [
            "sentiment_label", "neg_prob", "neu_prob", "pos_prob", "sentiment_score",
            "emotion",
            *[f"emotion_{e}" for e in EMOTION_LABELS],
        ]
        for col in score_cols:
            if col in scored_df.columns:
                comments_df.loc[clean_mask, col] = scored_df[col].values

    # Shared clean+scored slice reused by Stage 3 and Stage 4
    clean_comments_df = comments_df[clean_mask].reset_index(drop=True) if clean_mask.any() else pd.DataFrame()

    # Stage 3: semantic topic modeling on clean, scored comments
    # Runs after sentiment so topic stats include sentiment_label / sentiment_score.
    # Embeddings are computed fresh here (all-MiniLM-L6-v2); pass them via the
    # embeddings= parameter if a future step already produces sentence vectors.
    topics_df = pd.DataFrame()
    if not clean_comments_df.empty:
        topics_df, _ = run_topic_model(clean_comments_df)

    # Stage 4: fan segmentation — k-means on per-commenter feature matrix.
    # Runs in ingestion so Streamlit reads precomputed cluster assignments;
    # no clustering happens on page load.
    fan_segments: dict = {}
    if not clean_comments_df.empty:
        fan_segments = run_fan_segmentation(clean_comments_df)

    # Stage 5: statistically corrected alerts.
    # Three families (sentiment spike, velocity anomaly, keyword shift) each
    # corrected for multiple comparisons via Bonferroni (default) or BH.
    # Streamlit only renders the already-decided alert list.
    alerts_result: dict = {}
    if not clean_comments_df.empty and not videos_df.empty:
        alerts_result = run_alerts(clean_comments_df, videos_df)

    return channel, videos_df, comments_df, topics_df, fan_segments, alerts_result


# ── Page header ────────────────────────────────────────────────────────────────
st.title("Fanfare — Jared McCain Fan Intelligence")
st.caption(
    "Engagement, sentiment, and community analysis for marketing & social media leads"
)

# ── Fetch ──────────────────────────────────────────────────────────────────────
if fetch_btn:
    with st.spinner(
        f"Fetching up to {max_videos} videos and {max_comments} comments each…"
    ):
        channel, videos_df, comments_df, topics_df, fan_segments, alerts_result = load_data(max_videos, max_comments)
    if channel is None:
        st.error("Could not load channel @jaredmccain024. Check that the API key is valid and the channel is public.")
        st.stop()

    st.session_state.update(
        channel=channel,
        videos_df=videos_df,
        comments_df=comments_df,
        topics_df=topics_df,
        fan_segments=fan_segments,
        alerts_result=alerts_result,
        data_loaded=True,
    )

if not st.session_state.get("data_loaded"):
    st.info(
        "Click **Fetch & Analyze** in the sidebar to begin."
    )
    st.stop()

channel: dict = st.session_state.channel
videos_df: pd.DataFrame = st.session_state.videos_df
comments_df: pd.DataFrame = st.session_state.comments_df
topics_df: pd.DataFrame = st.session_state.get("topics_df", pd.DataFrame())
fan_segments: dict = st.session_state.get("fan_segments", {})
alerts_result: dict = st.session_state.get("alerts_result", {})

# adf: all clean (non-spam, non-duplicate), scored comments — includes Uncertain.
#      Used for counts, topic models, fan activity, and the raw comment table.
# cdf: "confident" subset of adf — Uncertain labels excluded.
#      Used for any metric that involves averaging sentiment scores so that
#      comments where the model cannot confidently assign a class do not
#      compress the signal toward zero (fake-neutral problem).
adf: pd.DataFrame = (
    comments_df[~comments_df["is_spam"] & ~comments_df["is_duplicate"]].dropna(
        subset=["sentiment_score"]
    )
    if not comments_df.empty and "is_spam" in comments_df.columns
    else comments_df
)
cdf: pd.DataFrame = (
    adf[adf["sentiment_label"] != "Uncertain"]
    if not adf.empty and "sentiment_label" in adf.columns
    else adf
)

# ── Channel hero ───────────────────────────────────────────────────────────────
c1, c2 = st.columns([1, 8])
with c1:
    if channel.get("thumbnail"):
        st.image(channel["thumbnail"], width=72)
with c2:
    st.subheader(channel["title"])
    handle_display = channel.get("custom_url") or channel_handle
    st.caption(f"youtube.com/{handle_display.lstrip('@')}")

st.divider()

# ── Top-level metrics ──────────────────────────────────────────────────────────
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Subscribers", f"{channel['subscriber_count']:,}")
m2.metric("Total Views", f"{channel['view_count']:,}")
m3.metric("Total Videos", f"{channel['video_count']:,}")
m4.metric("Videos Analyzed", len(videos_df) if not videos_df.empty else 0)
m5.metric("Comments Fetched", len(comments_df) if not comments_df.empty else 0)
_filtered = len(comments_df) - len(adf) if not comments_df.empty else 0
m6.metric("After Cleaning", len(adf), delta=f"-{_filtered} spam/dupe" if _filtered else None)

st.divider()

# ── Key Insights banner ────────────────────────────────────────────────────────
st.subheader("Key Insights")
st.caption("What the data says and what to do about it")

def _build_insights(
    videos_df: pd.DataFrame, comments_df: pd.DataFrame
) -> list[dict]:
    insights = []

    # Sentiment stats exclude Uncertain so the average reflects only comments
    # the model could confidently classify (see UNCERTAIN_THRESHOLD rationale)
    certain = (
        comments_df[comments_df["sentiment_label"] != "Uncertain"]
        if not comments_df.empty and "sentiment_label" in comments_df.columns
        else comments_df
    )

    if not certain.empty and "sentiment_label" in certain.columns:
        pos_pct = (certain["sentiment_label"] == "Positive").mean() * 100
        neg_pct = (certain["sentiment_label"] == "Negative").mean() * 100
        avg_score = certain["sentiment_score"].mean()
        sentiment_color = "green" if avg_score >= 0.05 else ("red" if avg_score <= -0.05 else "orange")
        insights.append({
            "color": sentiment_color,
            "icon": "😊" if avg_score >= 0.05 else ("😠" if avg_score <= -0.05 else "😐"),
            "title": f"{pos_pct:.0f}% of fan comments are positive",
            "action": (
                f"Fan sentiment is strong — lean into it with behind-the-scenes content and replies."
                if avg_score >= 0.1
                else f"Mixed reactions ({neg_pct:.0f}% negative) — review critical comments tab for recurring concerns."
            ),
        })

    if not videos_df.empty:
        top_video = videos_df.loc[videos_df["view_count"].idxmax()]
        insights.append({
            "color": "blue",
            "icon": "🎬",
            "title": f"Most viral: \"{top_video['title'][:55]}{'…' if len(top_video['title'])>55 else ''}\" — {top_video['view_count']:,} views",
            "action": f"Engagement rate: {top_video['engagement_rate']:.2f}%. Identify what made this video pop and replicate the format.",
        })

        avg_er = videos_df["engagement_rate"].mean()
        best_er = videos_df.loc[videos_df["engagement_rate"].idxmax()]
        if best_er["video_id"] != top_video["video_id"]:
            insights.append({
                "color": "violet",
                "icon": "📈",
                "title": f"Highest engagement: \"{best_er['title'][:50]}{'…' if len(best_er['title'])>50 else ''}\" — {best_er['engagement_rate']:.2f}%",
                "action": f"Channel avg is {avg_er:.2f}%. This video drove outsized fan interaction — study its hook, length, and topic.",
            })

    if not comments_df.empty:
        top_fans = get_top_fans(comments_df, top_n=5)
        if not top_fans.empty:
            top_fan = top_fans.iloc[0]
            insights.append({
                "color": "orange",
                "icon": "⭐",
                "title": f"Super fan: {top_fan['author']} — {int(top_fan['comment_count'])} comments across {int(top_fan['videos_commented'])} videos",
                "action": "Consider a shout-out, early access, or DM to convert this fan into an ambassador.",
            })

        topics_df = get_trending_topics(comments_df, top_n=5)
        if not topics_df.empty:
            top_word = topics_df.iloc[0]["word"]
            kw_sent = keyword_sentiment_breakdown(comments_df, [top_word])
            if not kw_sent.empty:
                kw_score = kw_sent.iloc[0]["avg_sentiment"]
                kw_emoji = "✅" if kw_score >= 0.05 else ("⚠️" if kw_score <= -0.05 else "🔵")
                insights.append({
                    "color": "green" if kw_score >= 0.05 else ("red" if kw_score <= -0.05 else "gray"),
                    "icon": kw_emoji,
                    "title": f"Top fan keyword: \"{top_word}\" ({kw_sent.iloc[0]['mentions']} mentions, sentiment {kw_score:+.2f})",
                    "action": (
                        f"Fans react positively to \"{top_word}\" content — amplify it in captions and titles."
                        if kw_score >= 0.05
                        else f"Fans use \"{top_word}\" in a critical context — investigate comments under the Topics tab."
                    ),
                })

    return insights


insights = _build_insights(videos_df, adf)
if insights:
    cols = st.columns(len(insights))
    for col, ins in zip(cols, insights):
        with col:
            st.markdown(
                f"**{ins['icon']} {ins['title']}**\n\n{ins['action']}",
            )
            st.divider()
else:
    st.info("Load data to see key insights.")

st.divider()

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_alerts, tab_eng, tab_sent, tab_fans, tab_topics, tab_table = st.tabs(
    ["🔔 Alerts", "📊 Engagement", "💬 Sentiment", "🏆 Top Fans", "🔥 Trending Topics", "📋 All Videos"]
)

# ┌─ Alerts ─────────────────────────────────────────────────────────────────────
with tab_alerts:
    st.subheader("Signal Alerts")
    st.caption(
        "Three families of hypothesis tests, each corrected for multiple comparisons. "
        "An alert fires only when the signal clears the corrected threshold **and** "
        "the effect size is large enough to act on.  "
        f"Correction method: **{CORRECTION_METHOD.upper()}** (edit `CORRECTION_METHOD` in `src/alerts.py` to switch to BH / FDR)."
    )

    if not alerts_result:
        st.info("Run **Fetch & Analyze** to generate alerts.")
    else:
        _a_alerts   = alerts_result.get("alerts", [])
        _a_summary  = alerts_result.get("summary", {})
        _a_families = alerts_result.get("families", {})
        _a_method   = alerts_result.get("correction", CORRECTION_METHOD)

        _tested  = _a_summary.get("tested", 0)
        _passed  = _a_summary.get("passed", 0)
        _naive   = _a_summary.get("naive_count", 0)
        _blocked = _naive - _passed

        # ── Summary banner ───────────────────────────────────────────────────
        st.markdown(
            f"**{_tested} potential signals tested · "
            f"{_passed} passed {_a_method.upper()} correction**"
            + (f" · {_blocked} suppressed vs. naïve α=0.05" if _blocked > 0 else
               " · same count as naïve α=0.05 (no inflation detected)")
        )

        # Per-family breakdown
        fam_cols = st.columns(3)
        _fam_labels = {
            "sentiment_spike":  "Sentiment spike",
            "velocity_anomaly": "Velocity anomaly",
            "keyword_shift":    "Keyword shift",
        }
        for col, (fam_key, fam_label) in zip(fam_cols, _fam_labels.items()):
            fam = _a_families.get(fam_key, {})
            with col:
                st.metric(
                    fam_label,
                    f"{fam.get('corrected_pass', 0)} / {fam.get('m', 0)} tests",
                    delta=f"naïve: {fam.get('naive_pass', 0)}",
                )
                st.caption(fam.get("description", ""))

        st.divider()

        if not _a_alerts:
            st.success(
                "No signals cleared the corrected threshold this run — "
                "the channel is performing within its normal statistical range."
            )
        else:
            _FAMILY_ICON = {
                "sentiment_spike":  "💬",
                "velocity_anomaly": "📈",
                "keyword_shift":    "🔑",
            }
            for alert in _a_alerts:
                fam   = alert["family"]
                sev   = alert.get("severity", "info")
                icon  = _FAMILY_ICON.get(fam, "🔔")
                title = alert["title"]

                with st.expander(f"{icon} {title}", expanded=True):
                    left, right = st.columns([3, 2])

                    with left:
                        # Recommended action — always present (contract enforced in alerts.py)
                        if sev == "warning":
                            st.warning(f"**Recommended action:** {alert['action']}")
                        else:
                            st.info(f"**Recommended action:** {alert['action']}")

                    with right:
                        st.markdown(f"**What changed:** {alert['magnitude_label']}")
                        st.markdown(f"**Sample size (n):** {alert['n']:,} comments")

                        method = alert.get("correction_method", _a_method)
                        raw_p  = alert.get("p_raw")
                        adj_p  = alert.get("p_adj")
                        thresh = alert.get("corrected_threshold")

                        if fam == "velocity_anomaly":
                            # z-score family: no formal p-value
                            st.markdown(
                                f"**Corrected threshold cleared:** {thresh}  \n"
                                f"**Normal-approx p (reference only):** {raw_p:.5f}"
                            )
                        else:
                            st.markdown(
                                f"**Raw p-value:** {raw_p:.5f}  \n"
                                f"**Adjusted p ({method}):** {adj_p:.5f}  \n"
                                f"**Corrected threshold:** {thresh:.5f}"
                            )

        # ── Methodology note ─────────────────────────────────────────────────
        with st.expander("Methodology — how correction works"):
            st.markdown(
                """
**Why multiple-comparisons correction?**
Every ingestion run tests many hypotheses simultaneously — one per video
(sentiment), one per video (velocity), one per keyword.  At a naïve α = 0.05
threshold, with no true change you'd still expect 5 % of tests to fire by
chance.  Across 50 videos that's ~2–3 phantom alerts per run even in a
completely uneventful week.

**Bonferroni (default)**
Per-test threshold = α / m (where m = tests in the family).
Controls FWER: the probability of *any* false positive in the family ≤ α.
Conservative — may suppress real signals when many genuine changes occur.

**Benjamini-Hochberg (BH)**
Controls FDR ≤ α: the *expected fraction* of fired alerts that are false
positives.  Fires more alerts when many real changes happen simultaneously.
Switch by setting `CORRECTION_METHOD = "bh"` in `src/alerts.py`.

**Velocity family (z-score, no formal p-value)**
Comment counts are right-skewed (Poisson-ish), so a t-test would be
misleading.  Instead we use z-score vs. the channel distribution and
translate the Bonferroni correction to an equivalent sigma threshold:
z* = Φ⁻¹(1 − α/(2m)).  This approximation is conservative (real tails
are heavier than normal) — we tend to under-fire, not over-fire.

**Sample guard**
No alert fires from fewer than {min_video} comments (sentiment / velocity)
or {min_kw} keyword mentions.  Sample size is shown on every alert.

**Effect-size gate**
Applied *after* correction: a statistically significant but negligible
change (|Δ| < 0.08 sentiment points) is not surfaced.  This prevents
large-n trivial effects from flooding the feed.
""".format(
                    min_video=MIN_VIDEO_COMMENTS,
                    min_kw=MIN_KEYWORD_MENTIONS,
                )
            )


# ┌─ Engagement ─────────────────────────────────────────────────────────────────
with tab_eng:
    if videos_df.empty:
        st.warning("No video data available.")
    else:
        st.subheader("Video Performance")
        sorted_asc = videos_df.sort_values("published_at")

        fig_views = px.bar(
            sorted_asc,
            x="published_at",
            y="view_count",
            color="engagement_rate",
            color_continuous_scale="Blues",
            hover_data=["title", "like_count", "comment_count", "engagement_rate"],
            labels={
                "published_at": "Published",
                "view_count": "Views",
                "engagement_rate": "Engagement %",
            },
            title="Views Per Video (color = engagement rate)",
        )
        fig_views.update_layout(coloraxis_colorbar_title="Eng %", xaxis_title="")
        st.plotly_chart(fig_views, use_container_width=True)

        col_left, col_right = st.columns(2)

        with col_left:
            top10 = videos_df.nlargest(10, "view_count").copy()
            top10["label"] = top10["title"].apply(
                lambda t: t[:40] + "…" if len(t) > 40 else t
            )
            fig_top = px.bar(
                top10,
                x="view_count",
                y="label",
                orientation="h",
                color="engagement_rate",
                color_continuous_scale="Teal",
                title="Top 10 Videos by Views",
                labels={"view_count": "Views", "label": "", "engagement_rate": "Eng %"},
            )
            fig_top.update_layout(yaxis_categoryorder="total ascending")
            st.plotly_chart(fig_top, use_container_width=True)

        with col_right:
            fig_scatter = px.scatter(
                videos_df,
                x="view_count",
                y="like_count",
                size="comment_count",
                size_max=40,
                color="engagement_rate",
                color_continuous_scale="Viridis",
                hover_data=["title", "engagement_rate", "comment_count"],
                title="Views vs Likes (bubble size = comment volume)",
                labels={
                    "view_count": "Views",
                    "like_count": "Likes",
                    "engagement_rate": "Eng %",
                },
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Avg Views / Video", f"{int(videos_df['view_count'].mean()):,}")
        s2.metric("Avg Likes / Video", f"{int(videos_df['like_count'].mean()):,}")
        s3.metric("Avg Comments / Video", f"{int(videos_df['comment_count'].mean()):,}")
        s4.metric("Avg Engagement Rate", f"{videos_df['engagement_rate'].mean():.2f}%")


# ┌─ Sentiment ──────────────────────────────────────────────────────────────────
with tab_sent:
    if adf.empty:
        st.warning("No comments passed the spam/duplicate filter — comments may be disabled on these videos.")
    else:
        st.subheader("Fan Comment Sentiment")

        # ── 4-state counts ──────────────────────────────────────────────────────
        _label_counts = adf["sentiment_label"].value_counts()
        _total = len(adf)
        _pos_n   = int(_label_counts.get("Positive",  0))
        _neu_n   = int(_label_counts.get("Neutral",   0))
        _neg_n   = int(_label_counts.get("Negative",  0))
        _unc_n   = int(_label_counts.get("Uncertain", 0))

        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Positive",  f"{_pos_n:,}",  f"{_pos_n/_total*100:.1f}%")
        sc2.metric("Neutral",   f"{_neu_n:,}",  f"{_neu_n/_total*100:.1f}%")
        sc3.metric("Negative",  f"{_neg_n:,}",  f"{_neg_n/_total*100:.1f}%")
        sc4.metric("Uncertain", f"{_unc_n:,}",  f"{_unc_n/_total*100:.1f}%")

        if _unc_n:
            st.info(
                f"**{_unc_n:,} comments ({_unc_n/_total*100:.1f}%) have uncertain sentiment** — "
                f"the model's max class probability was below {UNCERTAIN_THRESHOLD:.0%}. "
                "These are excluded from the average score and trend line (see scoring.py for rationale)."
            )

        # ── Pie + Gauge ─────────────────────────────────────────────────────────
        # Both metrics use cdf (certain comments only — uncertain excluded).
        # Gauge primary = like-weighted mean; flat mean shown as secondary.
        weighted_score, flat_score = weighted_mean_sentiment(cdf) if not cdf.empty else (0.0, 0.0)
        gap = round(weighted_score - flat_score, 4)

        col_pie, col_gauge = st.columns(2)

        with col_pie:
            fig_pie = px.pie(
                values=_label_counts.values,
                names=_label_counts.index,
                color=_label_counts.index,
                color_discrete_map={
                    "Positive":  "#27AE60",
                    "Neutral":   "#95A5A6",
                    "Negative":  "#E74C3C",
                    "Uncertain": "#9B59B6",
                },
                hole=0.45,
                title="Overall Sentiment Distribution (all scored comments)",
            )
            fig_pie.update_traces(
                textinfo="percent+label",
                textfont_size=13,
                pull=[0.08 if n == "Uncertain" else 0 for n in _label_counts.index],
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_gauge:
            bar_color = (
                "#27AE60"
                if weighted_score >= 0.05
                else ("#E74C3C" if weighted_score <= -0.05 else "#95A5A6")
            )
            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=round(weighted_score, 3),
                    number={"font": {"size": 40}},
                    delta={
                        "reference": 0,
                        "increasing": {"color": "#27AE60"},
                        "decreasing": {"color": "#E74C3C"},
                    },
                    title={
                        "text": (
                            "Like-Weighted Sentiment Score<br>"
                            "<sup>-1 = very negative · +1 = very positive<br>"
                            f"Flat mean: {flat_score:+.3f} · "
                            f"Uncertain excluded ({_unc_n:,})</sup>"
                        )
                    },
                    gauge={
                        "axis": {
                            "range": [-1, 1],
                            "tickvals": [-1, -0.5, 0, 0.5, 1],
                        },
                        "bar": {"color": bar_color, "thickness": 0.25},
                        "steps": [
                            {"range": [-1, -0.05], "color": "#FADBD8"},
                            {"range": [-0.05, 0.05], "color": "#F4F6F7"},
                            {"range": [0.05, 1], "color": "#D5F5E3"},
                        ],
                        "threshold": {
                            "line": {"color": "#2C3E50", "width": 3},
                            "thickness": 0.8,
                            "value": weighted_score,
                        },
                    },
                )
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        # ── Gap signal ──────────────────────────────────────────────────────────
        gm1, gm2, gm3 = st.columns(3)
        gm1.metric("Like-Weighted Score", f"{weighted_score:+.3f}")
        gm2.metric("Flat Mean Score",     f"{flat_score:+.3f}")
        gm3.metric("Gap (weighted − flat)", f"{gap:+.3f}",
                   delta=f"{'↑ liked comments more positive' if gap > 0 else '↓ liked comments more negative'}"
                   if abs(gap) >= SENTIMENT_GAP_THRESHOLD else None)

        if abs(gap) >= SENTIMENT_GAP_THRESHOLD:
            if gap > 0:
                st.warning(
                    f"**Endorsed majority is more positive than average** (gap {gap:+.3f}): "
                    "the comments fans actively liked skew more positive than the full distribution. "
                    "The community is amplifying optimism — critics may be posting but not resonating."
                )
            else:
                st.warning(
                    f"**Loud minority signal detected** (gap {gap:+.3f}): "
                    "the comments fans liked are notably more negative than the flat average. "
                    "Even fans who aren't posting criticism are endorsing it — a concern worth monitoring."
                )

        # ── Trend ───────────────────────────────────────────────────────────────
        # aggregate_sentiment_over_time returns avg_sentiment (like-weighted)
        # and flat_avg_sentiment so both can be plotted for consistency with
        # the headline gauge numbers.
        timeline_df = aggregate_sentiment_over_time(videos_df, adf)
        if not timeline_df.empty:
            st.subheader("Sentiment Trend Across Videos")
            st.caption(
                "Solid line = like-weighted mean · Dashed = flat mean · "
                "Uncertain comments excluded. Gap between lines indicates where "
                "liked and average-comment sentiment diverge."
            )
            # Round for cleaner hover display
            timeline_df["flat_avg_sentiment"] = timeline_df["flat_avg_sentiment"].round(3)
            timeline_df["avg_sentiment"] = timeline_df["avg_sentiment"].round(3)

            fig_trend = px.line(
                timeline_df,
                x="published_at",
                y="avg_sentiment",
                markers=True,
                hover_data={
                    "title": True,
                    "comment_count": True,
                    "positive_pct": True,
                    "negative_pct": True,
                    "flat_avg_sentiment": True,
                },
                labels={
                    "published_at": "Published",
                    "avg_sentiment": "Like-Weighted Sentiment",
                    "flat_avg_sentiment": "Flat Mean",
                },
                title="Fan sentiment per video (like-weighted, confident comments only)",
            )
            # Flat mean as secondary dashed trace
            fig_trend.add_scatter(
                x=timeline_df["published_at"],
                y=timeline_df["flat_avg_sentiment"],
                mode="lines",
                name="Flat mean",
                line=dict(color="#BDC3C7", width=1.5, dash="dash"),
                hovertemplate="Flat: %{y:.3f}<extra></extra>",
            )
            fig_trend.add_hline(
                y=0,
                line_dash="dot",
                line_color="#95A5A6",
                annotation_text="Neutral",
            )
            fig_trend.add_hrect(
                y0=0.05, y1=1, fillcolor="#27AE60", opacity=0.05, line_width=0
            )
            fig_trend.add_hrect(
                y0=-1, y1=-0.05, fillcolor="#E74C3C", opacity=0.05, line_width=0
            )
            st.plotly_chart(fig_trend, use_container_width=True)

        # Model attribution
        st.caption(
            "Sentiment: **RoBERTa** (cardiffnlp/twitter-roberta-base-sentiment-latest) EN · "
            "**XLM-RoBERTa** (cardiffnlp/twitter-xlm-roberta-base-sentiment) non-EN · "
            "Emotion: **DistilRoBERTa** (j-hartmann/emotion-english-distilroberta-base) · "
            f"Uncertain threshold: max class prob < {UNCERTAIN_THRESHOLD:.0%}"
        )

        # ── Global emotion distribution ──────────────────────────────────────────
        _has_emotion = "emotion" in adf.columns and adf["emotion"].notna().any()
        if _has_emotion:
            st.subheader("Emotion Distribution")
            emo_counts = (
                adf["emotion"].dropna().value_counts().reset_index()
            )
            emo_counts.columns = ["emotion", "count"]
            emo_counts["pct"] = (emo_counts["count"] / emo_counts["count"].sum() * 100).round(1)

            fig_emo = px.bar(
                emo_counts,
                x="emotion",
                y="count",
                color="emotion",
                color_discrete_map=EMOTION_COLORS,
                text=emo_counts["pct"].apply(lambda p: f"{p:.1f}%"),
                title="Fan Comment Emotion Breakdown (all clean comments)",
                labels={"count": "Comments", "emotion": "Emotion"},
                category_orders={"emotion": list(EMOTION_LABELS)},
            )
            fig_emo.update_traces(textposition="outside")
            fig_emo.update_layout(showlegend=False, xaxis_title="")
            st.plotly_chart(fig_emo, use_container_width=True)

        # ── Video Deep Dive ──────────────────────────────────────────────────────
        st.subheader("Video Deep Dive")
        _vid_options = (
            videos_df[["video_id", "title"]].drop_duplicates()
            if not videos_df.empty
            else pd.DataFrame()
        )
        if _vid_options.empty or "video_id" not in adf.columns:
            st.info("Load data to enable per-video drill-down.")
        else:
            _sel_title = st.selectbox(
                "Select a video",
                _vid_options["title"].tolist(),
                key="deepdive_video",
            )
            _sel_vid_id = _vid_options.loc[
                _vid_options["title"] == _sel_title, "video_id"
            ].iloc[0]
            _vid_comments = adf[adf["video_id"] == _sel_vid_id]

            if _vid_comments.empty:
                st.info("No clean comments for this video.")
            else:
                _vw, _vf = weighted_mean_sentiment(_vid_comments[_vid_comments["sentiment_label"] != "Uncertain"])
                _v_unc = (_vid_comments["sentiment_label"] == "Uncertain").sum()
                _v_total = len(_vid_comments)

                vm1, vm2, vm3 = st.columns(3)
                vm1.metric("Comments (clean)", f"{_v_total:,}")
                vm2.metric("Like-Weighted Sentiment", f"{_vw:+.3f}")
                vm3.metric("Uncertain", f"{_v_unc:,}", f"{_v_unc/_v_total*100:.1f}%")

                dd_col1, dd_col2 = st.columns(2)

                with dd_col1:
                    _v_sent = _vid_comments["sentiment_label"].value_counts().reset_index()
                    _v_sent.columns = ["label", "count"]
                    fig_v_sent = px.bar(
                        _v_sent,
                        x="label",
                        y="count",
                        color="label",
                        color_discrete_map={
                            "Positive": "#27AE60", "Neutral": "#95A5A6",
                            "Negative": "#E74C3C", "Uncertain": "#9B59B6",
                        },
                        title="Sentiment breakdown",
                        labels={"label": "", "count": "Comments"},
                    )
                    fig_v_sent.update_layout(showlegend=False)
                    st.plotly_chart(fig_v_sent, use_container_width=True)

                with dd_col2:
                    if _has_emotion and "emotion" in _vid_comments.columns:
                        _v_emo = (
                            _vid_comments["emotion"].dropna()
                            .value_counts().reset_index()
                        )
                        _v_emo.columns = ["emotion", "count"]
                        fig_v_emo = px.bar(
                            _v_emo,
                            x="emotion",
                            y="count",
                            color="emotion",
                            color_discrete_map=EMOTION_COLORS,
                            title="Emotion breakdown",
                            labels={"emotion": "", "count": "Comments"},
                            category_orders={"emotion": list(EMOTION_LABELS)},
                        )
                        fig_v_emo.update_layout(showlegend=False)
                        st.plotly_chart(fig_v_emo, use_container_width=True)
                    else:
                        st.info("Emotion data not available (VADER path or model not loaded).")

                # Sample comments by dominant emotion for this video
                if _has_emotion and "emotion" in _vid_comments.columns:
                    st.markdown("##### Sample Comments by Emotion")
                    _dom_emotions = (
                        _vid_comments["emotion"].dropna()
                        .value_counts().head(4).index.tolist()
                    )
                    emo_tabs = st.tabs([e.capitalize() for e in _dom_emotions])
                    for tab, emo in zip(emo_tabs, _dom_emotions):
                        with tab:
                            _emo_sample = (
                                _vid_comments[_vid_comments["emotion"] == emo]
                                .nlargest(5, "like_count")
                            )
                            for _, row in _emo_sample.iterrows():
                                score_str = f" · sentiment {row['sentiment_score']:+.2f}" if "sentiment_score" in row else ""
                                st.markdown(
                                    f"**{row['author']}**{score_str} · "
                                    f"{int(row.get('like_count', 0))} likes\n\n"
                                    f"{str(row['text'])[:300]}"
                                )
                                st.divider()

        # ── Sample comments — three panes ───────────────────────────────────────
        st.subheader("Sample Comments")
        col_pos, col_neg, col_unc = st.columns(3)

        with col_pos:
            st.markdown("##### Most Positive")
            for _, row in cdf.nlargest(5, "sentiment_score").iterrows():
                st.success(
                    f"**{row['author']}** · score {row['sentiment_score']:+.2f}\n\n{str(row['text'])[:250]}"
                )

        with col_neg:
            st.markdown("##### Most Critical")
            for _, row in cdf.nsmallest(5, "sentiment_score").iterrows():
                st.error(
                    f"**{row['author']}** · score {row['sentiment_score']:+.2f}\n\n{str(row['text'])[:250]}"
                )

        with col_unc:
            st.markdown("##### Uncertain")
            unc_sample = adf[adf["sentiment_label"] == "Uncertain"].head(5)
            if unc_sample.empty:
                st.info("No uncertain comments in this dataset.")
            else:
                for _, row in unc_sample.iterrows():
                    prob_str = (
                        f"pos {row['pos_prob']:.2f} / neu {row['neu_prob']:.2f} / neg {row['neg_prob']:.2f}"
                        if "pos_prob" in row
                        else ""
                    )
                    st.warning(
                        f"**{row['author']}**"
                        + (f" · {prob_str}" if prob_str else "")
                        + f"\n\n{str(row['text'])[:250]}"
                    )



# ┌─ Top Fans ───────────────────────────────────────────────────────────────────
# "Super fan" is now defined as membership in the Advocates segment — a data-
# driven label that emerges from the actual fan distribution rather than the
# previous magic-number thresholds (comment_count >= 2 AND sentiment >= 0.05).
with tab_fans:
    if adf.empty:
        st.warning("No comments passed the spam/duplicate filter — cannot segment fans.")
    elif not fan_segments or not fan_segments.get("clusters"):
        st.info(
            "Fan segmentation not available — corpus may be too small or "
            "scikit-learn is not installed."
        )
    else:
        _clusters   = fan_segments["clusters"]
        _fans_df    = fan_segments["fans_df"]
        _k_log      = fan_segments["k_log"]
        _k_chosen   = fan_segments["k_chosen"]
        _n_fans     = len(_fans_df)

        st.subheader("Fan Community Segments")
        st.caption(
            f"k-means · k={_k_chosen} · {_n_fans:,} unique commenters · "
            "features z-scored · random_state=42 (deterministic) · "
            "segments computed during ingestion, not page load"
        )

        # ── K-selection log ──────────────────────────────────────────────────
        if _k_log:
            with st.expander("K-selection log — silhouette + inertia per k tried"):
                st.caption(
                    "Silhouette maximised to select k. "
                    "Inertia logged as elbow sanity-check (not used for selection)."
                )
                log_df = pd.DataFrame(_k_log)
                log_df["chosen"] = log_df["k"] == _k_chosen
                st.dataframe(
                    log_df.style.apply(
                        lambda r: ["font-weight: bold"] * len(r) if r["chosen"] else [""] * len(r),
                        axis=1,
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

        # ── Segment cards ────────────────────────────────────────────────────
        st.subheader("Segments")
        for cluster in _clusters:
            pct = cluster["size"] / _n_fans * 100
            header = (
                f"{cluster['label']} · "
                f"{cluster['size']} fans ({pct:.0f}%) · "
                f"mean sentiment {cluster['mean_sentiment']:+.3f}"
            )
            with st.expander(header, expanded=True):
                # Action — hard requirement: always shown, never blank
                st.info(f"**Recommended action:** {cluster['action']}")

                left_col, right_col = st.columns([1, 1])

                with left_col:
                    # Centroid profile: z-scores relative to the average fan
                    z_vals = [cluster["centroid_z"][f] for f in cluster["centroid_z"]]
                    f_labels = [FEATURE_DISPLAY.get(f, f) for f in cluster["centroid_z"]]
                    bar_colors = ["#27AE60" if v >= 0 else "#E74C3C" for v in z_vals]

                    fig_cent = go.Figure(
                        go.Bar(
                            x=z_vals,
                            y=f_labels,
                            orientation="h",
                            marker_color=bar_colors,
                            hovertemplate="%{y}: %{x:+.2f}σ<extra></extra>",
                        )
                    )
                    fig_cent.add_vline(x=0, line_dash="dash", line_color="#95A5A6")
                    fig_cent.update_layout(
                        title="Centroid profile (σ from average fan)",
                        xaxis=dict(range=[-2.5, 2.5], title="Standard deviations"),
                        yaxis_title="",
                        height=260,
                        margin=dict(l=0, r=0, t=30, b=0),
                        showlegend=False,
                    )
                    st.plotly_chart(fig_cent, use_container_width=True)

                    # Original-unit centroid for reference
                    orig = cluster["centroid"]
                    st.caption(
                        f"comments: {orig['comment_count']:.1f} · "
                        f"videos: {orig['videos_commented']:.1f} · "
                        f"sentiment: {orig['avg_sentiment']:+.3f} · "
                        f"likes: {orig['likes_earned']:.0f} · "
                        f"recency: {orig['recency']:.0f} days · "
                        f"consistency: {orig['consistency']:.1f} weeks"
                    )

                with right_col:
                    st.markdown("**Representative fans** (closest to centroid)")
                    for fan in cluster["examples"]:
                        sent_str = f"{fan['avg_sentiment']:+.3f}" if fan.get("avg_sentiment") is not None else "—"
                        st.markdown(
                            f"**{fan['author']}** · "
                            f"{int(fan['comment_count'])} comments · "
                            f"{int(fan['videos_commented'])} videos · "
                            f"sentiment {sent_str} · "
                            f"{int(fan['likes_earned'])} likes · "
                            f"{int(fan['recency'])}d ago"
                        )

        # ── Scatter: all fans, colored by segment ────────────────────────────
        if len(_fans_df) >= 5:
            st.subheader("Fan Map — Activity × Sentiment")
            st.caption("Bubble size = likes earned · color = segment")

            _color_map = {c["label"]: c["color"] for c in _clusters}
            fig_scatter = px.scatter(
                _fans_df,
                x="comment_count",
                y="avg_sentiment",
                size="likes_earned",
                size_max=40,
                color="cluster_label",
                color_discrete_map=_color_map,
                hover_data={
                    "author": True,
                    "videos_commented": True,
                    "likes_earned": True,
                    "recency": True,
                    "consistency": True,
                },
                title="Fan Activity vs Sentiment (all segmented commenters)",
                labels={
                    "comment_count":  "Comments",
                    "avg_sentiment":  "Avg Sentiment",
                    "cluster_label":  "Segment",
                    "likes_earned":   "Likes",
                    "recency":        "Days ago",
                    "consistency":    "Weeks active",
                },
            )
            fig_scatter.add_hline(y=0, line_dash="dash", line_color="#95A5A6")
            st.plotly_chart(fig_scatter, use_container_width=True)

        # ── Fan lookup table ─────────────────────────────────────────────────
        st.subheader("Fan Lookup")
        st.caption("Full table — sort any column, search by name in your browser")
        _display_fans = _fans_df[[
            "author", "cluster_label", "comment_count", "videos_commented",
            "avg_sentiment", "likes_earned", "recency", "consistency",
        ]].copy()
        _display_fans.columns = [
            "Fan", "Segment", "Comments", "Videos",
            "Avg Sentiment", "Likes Earned", "Days Since Last", "Weeks Active",
        ]
        _display_fans = _display_fans.sort_values("Comments", ascending=False).reset_index(drop=True)
        st.dataframe(_display_fans, use_container_width=True, hide_index=True)


# ┌─ Trending Topics ─────────────────────────────────────────────────────────────
with tab_topics:
    if adf.empty:
        st.warning("No comments passed the spam/duplicate filter — cannot extract topics.")
    elif topics_df.empty:
        st.info(
            "Semantic topic model returned no topics — corpus may be too small "
            "or BERTopic dependencies are not installed. "
            "Install with: `pip install bertopic sentence-transformers`"
        )
    else:
        st.subheader("Semantic Topics in Fan Comments")
        st.caption(
            "BERTopic · all-MiniLM-L6-v2 embeddings · TruncatedSVD (UMAP-free) · "
            "KeyBERTInspired labels · prominence = Σ log(1 + likes) · "
            "sentiment requires ≥ 20 confident comments per topic"
        )

        # ── Action signal color map ───────────────────────────────────────────────
        def _topic_color(action: str) -> str:
            if action.startswith("✅"):
                return "#27AE60"
            if action.startswith("⚠️"):
                return "#E74C3C"
            if action.startswith("🔵"):
                return "#3498DB"
            return "#BDC3C7"

        topic_colors = topics_df["action"].apply(_topic_color).tolist()

        # ── Prominence bar chart ─────────────────────────────────────────────────
        fig_prom = go.Figure(
            go.Bar(
                x=topics_df["prominence"],
                y=topics_df["label"],
                orientation="h",
                marker_color=topic_colors,
                customdata=topics_df[["n_comments", "n_certain", "action"]].values,
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Prominence: %{x:.1f}<br>"
                    "Comments: %{customdata[0]}<br>"
                    "Certain: %{customdata[1]}<br>"
                    "Signal: %{customdata[2]}<extra></extra>"
                ),
            )
        )
        fig_prom.update_layout(
            title="Topic Prominence (Σ log(1+likes) — like-weighted size)",
            xaxis_title="Prominence",
            yaxis_categoryorder="total ascending",
            height=max(350, len(topics_df) * 28),
            showlegend=False,
            margin=dict(l=0),
        )
        st.plotly_chart(fig_prom, use_container_width=True)

        # ── Sentiment scatter (topics with enough certain comments) ───────────────
        scored_topics = topics_df[topics_df["has_sentiment"]].copy()
        if not scored_topics.empty:
            st.subheader("Topic Sentiment vs Prominence")
            fig_tsent = px.scatter(
                scored_topics,
                x="prominence",
                y="weighted_sentiment",
                size="n_comments",
                size_max=50,
                text="label",
                color="weighted_sentiment",
                color_continuous_scale=[
                    [0, "#E74C3C"],
                    [0.5, "#BDC3C7"],
                    [1, "#27AE60"],
                ],
                color_continuous_midpoint=0,
                range_color=[-0.6, 0.6],
                hover_data={"n_comments": True, "n_certain": True, "flat_sentiment": True},
                title="Topic Size vs Like-Weighted Sentiment (bubble = comment count)",
                labels={
                    "prominence": "Prominence (Σ log(1+likes))",
                    "weighted_sentiment": "Like-Weighted Sentiment",
                    "n_comments": "Comments",
                    "n_certain": "Certain",
                    "flat_sentiment": "Flat Mean",
                },
            )
            fig_tsent.add_hline(
                y=0, line_dash="dash", line_color="#95A5A6", annotation_text="Neutral"
            )
            fig_tsent.update_traces(textposition="top center", textfont_size=9)
            st.plotly_chart(fig_tsent, use_container_width=True)

        # ── Per-topic cards ───────────────────────────────────────────────────────
        st.subheader("Topic Details")
        for _, row in topics_df.iterrows():
            action = row["action"]
            color = _topic_color(action)
            label = row["label"]
            n_com = int(row["n_comments"])
            n_cert = int(row["n_certain"])
            prom = float(row["prominence"])

            if row["has_sentiment"]:
                ws = float(row["weighted_sentiment"])
                fs = float(row["flat_sentiment"])
                sent_str = f"weighted {ws:+.3f} · flat {fs:+.3f} · n={n_cert}"
            else:
                sent_str = f"{action}"  # already "— (n=X)"

            header = f"{action}  **{label}** — {n_com} comments · prominence {prom:.1f} · {sent_str}"

            with st.expander(header):
                ex1, ex2 = st.columns([1, 1])
                with ex1:
                    st.markdown(f"**Action signal:** {action}")
                    st.markdown(f"**Comments in topic:** {n_com:,}")
                    st.markdown(f"**Certain (non-Uncertain):** {n_cert:,}")
                    st.markdown(f"**Prominence:** {prom:.3f}")
                    if row["has_sentiment"]:
                        st.markdown(f"**Like-weighted sentiment:** {ws:+.3f}")
                        st.markdown(f"**Flat mean sentiment:** {fs:+.3f}")
                    else:
                        st.markdown(
                            f"**Sentiment:** not shown — fewer than 20 confident comments (n={n_cert})"
                        )
                with ex2:
                    st.markdown("**Top comments by likes:**")
                    examples = row.get("examples") or []
                    if examples:
                        for i, ex in enumerate(examples, 1):
                            st.markdown(f"{i}. {str(ex)[:300]}")
                    else:
                        st.caption("No examples available.")


# ┌─ All Videos ──────────────────────────────────────────────────────────────────
with tab_table:
    if videos_df.empty:
        st.warning("No video data loaded.")
    else:
        st.subheader("Full Video Dataset")

        display_df = videos_df[
            [
                "title",
                "published_at",
                "view_count",
                "like_count",
                "comment_count",
                "engagement_rate",
                "duration_secs",
            ]
        ].copy()
        display_df["published_at"] = display_df["published_at"].dt.strftime("%Y-%m-%d")
        display_df["duration"] = display_df["duration_secs"].apply(
            lambda s: f"{s // 60}m {s % 60}s" if s else "N/A"
        )
        display_df = display_df.drop(columns=["duration_secs"])
        display_df.columns = [
            "Title",
            "Published",
            "Views",
            "Likes",
            "Comments",
            "Engagement %",
            "Duration",
        ]

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Views": st.column_config.NumberColumn(format="%d"),
                "Likes": st.column_config.NumberColumn(format="%d"),
                "Comments": st.column_config.NumberColumn(format="%d"),
                "Engagement %": st.column_config.NumberColumn(format="%.2f%%"),
            },
        )

        st.download_button(
            "Download Videos CSV",
            display_df.to_csv(index=False),
            file_name="jared_mccain_videos.csv",
            mime="text/csv",
        )

        if not comments_df.empty:
            st.subheader("Comment Explorer")
            st.caption(
                f"{len(adf):,} used in analytics · "
                f"{comments_df['is_spam'].sum():,} spam · "
                f"{comments_df['is_duplicate'].sum():,} near-duplicates"
            )

            # ── Filters ─────────────────────────────────────────────────────────
            _filt_col1, _filt_col2, _filt_col3 = st.columns(3)

            with _filt_col1:
                _sent_opts = ["Positive", "Neutral", "Negative", "Uncertain"]
                _sent_filter = st.multiselect(
                    "Filter by sentiment",
                    options=_sent_opts,
                    default=[],
                    key="explorer_sentiment",
                )

            with _filt_col2:
                _has_emo_col = "emotion" in comments_df.columns and comments_df["emotion"].notna().any()
                if _has_emo_col:
                    _present_emotions = sorted(comments_df["emotion"].dropna().unique().tolist())
                    _emo_filter = st.multiselect(
                        "Filter by emotion",
                        options=_present_emotions,
                        default=[],
                        key="explorer_emotion",
                    )
                else:
                    _emo_filter = []
                    st.caption("Emotion data not available")

            with _filt_col3:
                _spam_filter = st.selectbox(
                    "Show",
                    options=["Clean only", "All (including spam/dupe)", "Spam only"],
                    key="explorer_spam",
                )

            # Apply filters
            _explorer_df = comments_df.copy()
            if _spam_filter == "Clean only" and "is_spam" in _explorer_df.columns:
                _explorer_df = _explorer_df[
                    ~_explorer_df["is_spam"] & ~_explorer_df["is_duplicate"]
                ]
            elif _spam_filter == "Spam only" and "is_spam" in _explorer_df.columns:
                _explorer_df = _explorer_df[
                    _explorer_df["is_spam"] | _explorer_df["is_duplicate"]
                ]
            if _sent_filter and "sentiment_label" in _explorer_df.columns:
                _explorer_df = _explorer_df[
                    _explorer_df["sentiment_label"].isin(_sent_filter)
                ]
            if _emo_filter and "emotion" in _explorer_df.columns:
                _explorer_df = _explorer_df[
                    _explorer_df["emotion"].isin(_emo_filter)
                ]

            st.caption(f"Showing {len(_explorer_df):,} comments")

            _display_cols = [
                "video_id", "author", "text",
                "sentiment_label", "sentiment_score",
                "emotion",
                "is_spam", "is_duplicate", "language",
                "like_count", "published_at",
            ]
            _comment_display = _explorer_df[
                [c for c in _display_cols if c in _explorer_df.columns]
            ].copy()
            _comment_display.columns = [
                c.replace("_", " ").title() for c in _comment_display.columns
            ]
            st.dataframe(_comment_display, use_container_width=True, hide_index=True)
            st.download_button(
                "Download filtered CSV",
                _comment_display.to_csv(index=False),
                file_name="jared_mccain_comments_filtered.csv",
                mime="text/csv",
            )
