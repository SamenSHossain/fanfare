import re
from collections import Counter

import pandas as pd

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "as", "is", "was", "are", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "i", "me", "my", "we", "our",
    "you", "your", "he", "she", "his", "her", "it", "its", "they", "their",
    "this", "that", "these", "those", "what", "which", "who", "how", "when",
    "where", "why", "all", "just", "so", "if", "up", "out", "not", "no",
    "like", "get", "go", "got", "going", "know", "think", "see", "let",
    "much", "more", "very", "really", "also", "about", "from", "into",
    "than", "then", "there", "too", "some", "any", "one", "two", "new",
    "s", "t", "re", "ve", "ll", "d", "m", "im", "u", "ur", "r", "lol",
    "its", "its", "dont", "cant", "wont", "im", "hes", "shes", "thats",
    "yeah", "yep", "yes", "nah", "oh", "hey", "hi", "bro", "man", "guy",
}


def compute_engagement_rate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    safe_views = df["view_count"].replace(0, 1)
    df["engagement_rate"] = (
        (df["like_count"] + df["comment_count"]) / safe_views * 100
    ).round(2)
    return df


def get_top_fans(comments_df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    if comments_df.empty or "author" not in comments_df.columns:
        return pd.DataFrame()

    agg = (
        comments_df.groupby(["author", "author_channel_id"])
        .agg(
            comment_count=("text", "count"),
            videos_commented=("video_id", "nunique"),
            avg_sentiment=("sentiment_score", "mean"),
            total_likes_received=("like_count", "sum"),
        )
        .reset_index()
        .sort_values("comment_count", ascending=False)
        .head(top_n)
    )
    agg["avg_sentiment"] = agg["avg_sentiment"].round(3)
    return agg


def get_trending_topics(comments_df: pd.DataFrame, top_n: int = 30) -> pd.DataFrame:
    if comments_df.empty:
        return pd.DataFrame()

    text = " ".join(comments_df["text"].astype(str).tolist()).lower()
    words = re.findall(r"\b[a-z]{3,}\b", text)
    filtered = [w for w in words if w not in _STOPWORDS]
    counts = Counter(filtered).most_common(top_n)
    return pd.DataFrame(counts, columns=["word", "count"])


def aggregate_sentiment_over_time(
    videos_df: pd.DataFrame, comments_df: pd.DataFrame
) -> pd.DataFrame:
    if comments_df.empty or videos_df.empty:
        return pd.DataFrame()

    per_video = (
        comments_df.groupby("video_id")
        .agg(
            avg_sentiment=("sentiment_score", "mean"),
            comment_count=("text", "count"),
            positive_pct=(
                "sentiment_label",
                lambda x: round((x == "Positive").mean() * 100, 1),
            ),
            negative_pct=(
                "sentiment_label",
                lambda x: round((x == "Negative").mean() * 100, 1),
            ),
        )
        .reset_index()
    )

    merged = videos_df[["video_id", "title", "published_at"]].merge(
        per_video, on="video_id", how="inner"
    )
    merged["published_at"] = pd.to_datetime(merged["published_at"])
    merged = merged.sort_values("published_at")
    merged["short_title"] = merged["title"].apply(
        lambda t: t[:35] + "…" if len(t) > 35 else t
    )
    return merged


def keyword_sentiment_breakdown(
    comments_df: pd.DataFrame, keywords: list[str]
) -> pd.DataFrame:
    rows = []
    for word in keywords:
        pattern = r"\b" + re.escape(word) + r"\b"
        mask = comments_df["text"].str.lower().str.contains(pattern, regex=True, na=False)
        subset = comments_df[mask]
        if not subset.empty:
            rows.append(
                {
                    "keyword": word,
                    "mentions": len(subset),
                    "avg_sentiment": round(subset["sentiment_score"].mean(), 3),
                    "positive_pct": round(
                        (subset["sentiment_label"] == "Positive").mean() * 100, 1
                    ),
                }
            )
    return pd.DataFrame(rows)
