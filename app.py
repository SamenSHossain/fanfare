import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.analytics import (
    aggregate_sentiment_over_time,
    compute_engagement_rate,
    get_top_fans,
    get_trending_topics,
    keyword_sentiment_breakdown,
)
from src.sentiment import batch_analyze
from src.youtube_client import YouTubeClient

# ── Config ─────────────────────────────────────────────────────────────────────
YOUTUBE_API_KEY = "AIzaSyBHCfCa25OzyRfLXSqWZ1IPjRgVAD6DgLg"
CHANNEL_HANDLE = "jaredmccain024"

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
    st.caption("YouTube Data API v3 · VADER Sentiment · No credit card required")


# ── Cached data loader ─────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def load_data(
    max_vids: int,
    max_coms: int,
) -> tuple[dict | None, pd.DataFrame, pd.DataFrame]:
    client = YouTubeClient(YOUTUBE_API_KEY)

    channel = client.get_channel_info(handle=CHANNEL_HANDLE)
    if not channel:
        return None, pd.DataFrame(), pd.DataFrame()

    video_ids = client.get_video_ids(channel["uploads_playlist_id"], max_vids)
    if not video_ids:
        return channel, pd.DataFrame(), pd.DataFrame()

    videos_df = client.get_video_details(video_ids)
    if not videos_df.empty:
        videos_df = compute_engagement_rate(videos_df)
        videos_df["published_at"] = pd.to_datetime(videos_df["published_at"])
        videos_df = videos_df.sort_values("published_at", ascending=False).reset_index(
            drop=True
        )

    comments_df = client.get_all_comments(video_ids, max_coms)
    if not comments_df.empty:
        comments_df = batch_analyze(comments_df)

    return channel, videos_df, comments_df


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
        channel, videos_df, comments_df = load_data(max_videos, max_comments)
    if channel is None:
        st.error("Could not load channel @jaredmccain024. Check that the API key is valid and the channel is public.")
        st.stop()

    st.session_state.update(
        channel=channel,
        videos_df=videos_df,
        comments_df=comments_df,
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
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Subscribers", f"{channel['subscriber_count']:,}")
m2.metric("Total Views", f"{channel['view_count']:,}")
m3.metric("Total Videos", f"{channel['video_count']:,}")
m4.metric("Videos Analyzed", len(videos_df) if not videos_df.empty else 0)
m5.metric("Comments Scraped", len(comments_df) if not comments_df.empty else 0)

st.divider()

# ── Key Insights banner ────────────────────────────────────────────────────────
st.subheader("Key Insights")
st.caption("What the data says and what to do about it")

def _build_insights(
    videos_df: pd.DataFrame, comments_df: pd.DataFrame
) -> list[dict]:
    insights = []

    if not comments_df.empty and "sentiment_label" in comments_df.columns:
        pos_pct = (comments_df["sentiment_label"] == "Positive").mean() * 100
        neg_pct = (comments_df["sentiment_label"] == "Negative").mean() * 100
        avg_score = comments_df["sentiment_score"].mean()
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


insights = _build_insights(videos_df, comments_df)
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
tab_eng, tab_sent, tab_fans, tab_topics, tab_table = st.tabs(
    ["📊 Engagement", "💬 Sentiment", "🏆 Top Fans", "🔥 Trending Topics", "📋 All Videos"]
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
    if comments_df.empty:
        st.warning("No comments were fetched — comments may be disabled on these videos.")
    else:
        st.subheader("Fan Comment Sentiment")

        counts = comments_df["sentiment_label"].value_counts()
        avg_score = comments_df["sentiment_score"].mean()

        col_pie, col_gauge = st.columns(2)

        with col_pie:
            fig_pie = px.pie(
                values=counts.values,
                names=counts.index,
                color=counts.index,
                color_discrete_map={
                    "Positive": "#27AE60",
                    "Neutral": "#95A5A6",
                    "Negative": "#E74C3C",
                },
                hole=0.45,
                title="Overall Sentiment Distribution",
            )
            fig_pie.update_traces(
                textinfo="percent+label", textfont_size=14, pull=[0.03, 0, 0]
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_gauge:
            bar_color = (
                "#27AE60"
                if avg_score >= 0.05
                else ("#E74C3C" if avg_score <= -0.05 else "#95A5A6")
            )
            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=round(avg_score, 3),
                    number={"font": {"size": 40}},
                    delta={
                        "reference": 0,
                        "increasing": {"color": "#27AE60"},
                        "decreasing": {"color": "#E74C3C"},
                    },
                    title={
                        "text": "Average Sentiment Score<br><sup>-1 = very negative · +1 = very positive</sup>"
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
                            "value": avg_score,
                        },
                    },
                )
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        timeline_df = aggregate_sentiment_over_time(videos_df, comments_df)
        if not timeline_df.empty:
            st.subheader("Sentiment Trend Across Videos")
            fig_trend = px.line(
                timeline_df,
                x="published_at",
                y="avg_sentiment",
                markers=True,
                hover_data=["title", "comment_count", "positive_pct", "negative_pct"],
                labels={
                    "published_at": "Published",
                    "avg_sentiment": "Avg Sentiment Score",
                },
                title="How fan sentiment has changed video-by-video",
            )
            fig_trend.add_hline(
                y=0,
                line_dash="dash",
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

        st.subheader("Sample Comments")
        col_pos, col_neg = st.columns(2)

        with col_pos:
            st.markdown("##### Most Positive")
            for _, row in comments_df.nlargest(5, "sentiment_score").iterrows():
                st.success(
                    f"**{row['author']}** · score {row['sentiment_score']:+.2f}\n\n{str(row['text'])[:250]}"
                )

        with col_neg:
            st.markdown("##### Most Critical")
            for _, row in comments_df.nsmallest(5, "sentiment_score").iterrows():
                st.error(
                    f"**{row['author']}** · score {row['sentiment_score']:+.2f}\n\n{str(row['text'])[:250]}"
                )


# ┌─ Top Fans ───────────────────────────────────────────────────────────────────
with tab_fans:
    if comments_df.empty:
        st.warning("No comments data — cannot identify top fans.")
    else:
        st.subheader("Most Active Community Members")
        top_fans_df = get_top_fans(comments_df, top_n=20)

        if top_fans_df.empty:
            st.info("Not enough comment data to build a fan leaderboard.")
        else:
            col_chart, col_table = st.columns([3, 2])

            with col_chart:
                fig_fans = px.bar(
                    top_fans_df.head(15),
                    x="comment_count",
                    y="author",
                    orientation="h",
                    color="avg_sentiment",
                    color_continuous_scale=[
                        [0, "#E74C3C"],
                        [0.5, "#BDC3C7"],
                        [1, "#27AE60"],
                    ],
                    color_continuous_midpoint=0,
                    range_color=[-0.6, 0.6],
                    title="Top 15 Fans by Comment Volume (color = avg sentiment)",
                    labels={
                        "comment_count": "Comments",
                        "author": "",
                        "avg_sentiment": "Avg Sentiment",
                    },
                )
                fig_fans.update_layout(yaxis_categoryorder="total ascending")
                st.plotly_chart(fig_fans, use_container_width=True)

            with col_table:
                st.markdown("##### Fan Leaderboard")
                display = top_fans_df[
                    [
                        "author",
                        "comment_count",
                        "videos_commented",
                        "avg_sentiment",
                        "total_likes_received",
                    ]
                ].copy()
                display.columns = [
                    "Fan",
                    "Comments",
                    "Videos",
                    "Avg Sentiment",
                    "Likes Earned",
                ]
                st.dataframe(display, use_container_width=True, hide_index=True)

            st.subheader("Super Fans")
            st.caption(
                "High activity + positive sentiment — ideal for shout-outs, giveaways, or ambassador programs"
            )

            super_fans = top_fans_df[
                (top_fans_df["comment_count"] >= 2)
                & (top_fans_df["avg_sentiment"] >= 0.05)
            ].head(10)

            if super_fans.empty:
                st.info(
                    "No super fans identified yet — try increasing the comments-per-video limit."
                )
            else:
                for rank, (_, fan) in enumerate(super_fans.iterrows(), 1):
                    emoji = "🟢" if fan["avg_sentiment"] > 0.2 else "🟡"
                    st.markdown(
                        f"**#{rank} {fan['author']}** {emoji} — "
                        f"{int(fan['comment_count'])} comments across {int(fan['videos_commented'])} video(s) · "
                        f"Sentiment: {fan['avg_sentiment']:+.3f} · "
                        f"Likes earned: {int(fan['total_likes_received'])}"
                    )

            if len(top_fans_df) >= 5:
                st.subheader("Fan Activity vs Sentiment")
                fig_fan_scatter = px.scatter(
                    top_fans_df,
                    x="comment_count",
                    y="avg_sentiment",
                    size="total_likes_received",
                    size_max=40,
                    text="author",
                    color="avg_sentiment",
                    color_continuous_scale=[
                        [0, "#E74C3C"],
                        [0.5, "#BDC3C7"],
                        [1, "#27AE60"],
                    ],
                    color_continuous_midpoint=0,
                    range_color=[-0.6, 0.6],
                    title="Fan Activity vs Sentiment (bubble = likes received on comments)",
                    labels={
                        "comment_count": "# Comments",
                        "avg_sentiment": "Avg Sentiment Score",
                    },
                )
                fig_fan_scatter.add_hline(
                    y=0, line_dash="dash", line_color="#95A5A6"
                )
                fig_fan_scatter.update_traces(
                    textposition="top center", textfont_size=9
                )
                st.plotly_chart(fig_fan_scatter, use_container_width=True)


# ┌─ Trending Topics ─────────────────────────────────────────────────────────────
with tab_topics:
    if comments_df.empty:
        st.warning("No comments data — cannot extract topics.")
    else:
        st.subheader("What Fans Are Talking About")
        topics_df = get_trending_topics(comments_df, top_n=30)

        if topics_df.empty:
            st.info("Not enough comment text to extract topics.")
        else:
            col_bar, col_stats = st.columns([3, 1])

            with col_bar:
                fig_topics = px.bar(
                    topics_df,
                    x="count",
                    y="word",
                    orientation="h",
                    color="count",
                    color_continuous_scale="Tealgrn",
                    title="Top 30 Keywords in Fan Comments",
                    labels={"count": "Mentions", "word": ""},
                )
                fig_topics.update_layout(
                    yaxis_categoryorder="total ascending",
                    showlegend=False,
                    height=700,
                )
                st.plotly_chart(fig_topics, use_container_width=True)

            with col_stats:
                st.markdown("##### Quick Stats")
                total_words = topics_df["count"].sum()
                for _, row in topics_df.head(10).iterrows():
                    pct = row["count"] / total_words * 100
                    st.metric(row["word"], f"{row['count']:,}", f"{pct:.1f}% share")

            st.subheader("Keyword Sentiment Breakdown")
            st.caption(
                "How fans feel when each keyword appears — spot what to amplify vs. watch"
            )

            kw_sentiment = keyword_sentiment_breakdown(
                comments_df, topics_df.head(15)["word"].tolist()
            )

            if not kw_sentiment.empty:
                fig_kw = px.scatter(
                    kw_sentiment,
                    x="mentions",
                    y="avg_sentiment",
                    size="mentions",
                    size_max=50,
                    text="keyword",
                    color="avg_sentiment",
                    color_continuous_scale=[
                        [0, "#E74C3C"],
                        [0.5, "#BDC3C7"],
                        [1, "#27AE60"],
                    ],
                    color_continuous_midpoint=0,
                    range_color=[-0.5, 0.5],
                    title="Keyword Frequency vs Associated Sentiment",
                    labels={
                        "mentions": "Mentions",
                        "avg_sentiment": "Avg Sentiment Score",
                    },
                )
                fig_kw.add_hline(
                    y=0,
                    line_dash="dash",
                    line_color="#95A5A6",
                    annotation_text="Neutral",
                )
                fig_kw.update_traces(textposition="top center", textfont_size=11)
                st.plotly_chart(fig_kw, use_container_width=True)

                kw_sentiment["signal"] = kw_sentiment["avg_sentiment"].apply(
                    lambda s: "✅ Amplify"
                    if s >= 0.1
                    else ("⚠️ Monitor" if s <= -0.1 else "🔵 Neutral")
                )
                kw_display = kw_sentiment[
                    ["keyword", "mentions", "avg_sentiment", "positive_pct", "signal"]
                ].copy()
                kw_display.columns = [
                    "Keyword",
                    "Mentions",
                    "Avg Sentiment",
                    "Positive %",
                    "Action",
                ]
                st.dataframe(kw_display, use_container_width=True, hide_index=True)


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
            st.subheader("All Comments")
            comment_display = comments_df[
                [
                    "video_id",
                    "author",
                    "text",
                    "sentiment_label",
                    "sentiment_score",
                    "like_count",
                    "published_at",
                ]
            ].copy()
            comment_display.columns = [
                "Video ID",
                "Author",
                "Comment",
                "Sentiment",
                "Score",
                "Likes",
                "Date",
            ]
            st.dataframe(comment_display, use_container_width=True, hide_index=True)
            st.download_button(
                "Download Comments CSV",
                comment_display.to_csv(index=False),
                file_name="jared_mccain_comments.csv",
                mime="text/csv",
            )
