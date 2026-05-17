import re
from collections import Counter

import numpy as np
import pandas as pd

# Gap between like-weighted and flat mean above which we surface the
# "loud minority vs endorsed majority" signal to the analyst.
SENTIMENT_GAP_THRESHOLD: float = 0.10

try:
    from sklearn.feature_extraction.text import TfidfVectorizer as _TfidfVectorizer
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

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


def weighted_mean_sentiment(
    df: pd.DataFrame,
    score_col: str = "sentiment_score",
    likes_col: str = "like_count",
) -> tuple[float, float]:
    """Return (like_weighted_mean, flat_mean) for a confident-comment DataFrame.

    Weight formula: w_i = log(1 + like_count_i)

    Why log, not raw likes
    ----------------------
    YouTube comment likes follow a heavy-tailed distribution — a handful of
    comments accumulate thousands of likes while the median sits near zero.
    Using raw likes as weights would hand a single viral comment (e.g. 10 000
    likes) the same aggregate influence as hundreds of ordinary ones combined,
    replacing the community's aggregate signal with one person's opinion.
    log(1 + k) compresses the dynamic range: a 100-like comment gets ~5× the
    weight of a 0-like comment rather than 100×. This keeps endorsed comments
    meaningfully more influential than ignored ones while preventing outliers
    from dominating. The +1 shift ensures zero-like comments contribute weight
    log(1) = 0 (they are not excluded, but they receive no endorsement bonus).

    When total weight is zero (no comment has any likes), weighted falls back
    to the flat mean so the return value is always well-defined.

    Returns (0.0, 0.0) on an empty frame.
    """
    if df.empty:
        return 0.0, 0.0

    scores = df[score_col].to_numpy(dtype=float)
    weights = np.log1p(df[likes_col].clip(lower=0).to_numpy(dtype=float))

    flat = float(np.mean(scores))
    total_w = weights.sum()

    weighted = flat if total_w == 0 else float(np.dot(scores, weights) / total_w)
    return round(weighted, 4), round(flat, 4)


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

    # comment_count and activity metrics include all clean comments
    activity = (
        comments_df.groupby(["author", "author_channel_id"])
        .agg(
            comment_count=("text", "count"),
            videos_commented=("video_id", "nunique"),
            total_likes_received=("like_count", "sum"),
        )
        .reset_index()
    )

    # avg_sentiment only from comments with a confident label — uncertain
    # comments would pull per-fan averages toward zero the same way they
    # distort the global average (see UNCERTAIN_THRESHOLD comment in scoring.py)
    if "sentiment_label" in comments_df.columns and "sentiment_score" in comments_df.columns:
        certain = comments_df[comments_df["sentiment_label"] != "Uncertain"]
        sentiment = (
            certain.groupby(["author", "author_channel_id"])["sentiment_score"]
            .mean()
            .round(3)
            .reset_index()
            .rename(columns={"sentiment_score": "avg_sentiment"})
        )
        agg = activity.merge(sentiment, on=["author", "author_channel_id"], how="left")
    else:
        agg = activity
        agg["avg_sentiment"] = float("nan")

    return agg.sort_values("comment_count", ascending=False).head(top_n)


def get_trending_topics(comments_df: pd.DataFrame, top_n: int = 30) -> pd.DataFrame:
    if comments_df.empty:
        return pd.DataFrame(columns=["word", "count"])

    texts = comments_df["text"].astype(str).str.lower().tolist()

    if _SKLEARN_AVAILABLE and len(texts) >= 2:
        try:
            vec = _TfidfVectorizer(
                ngram_range=(1, 2),   # unigrams + bigrams
                max_features=500,
                stop_words=list(_STOPWORDS),
                min_df=2,             # phrase must appear in at least 2 comments
                token_pattern=r"(?u)\b[a-z]{3,}\b",
                sublinear_tf=True,    # log(tf+1) — prevents ultra-common words dominating
            )
            tfidf = vec.fit_transform(texts)
            names = vec.get_feature_names_out()
            agg_tfidf = tfidf.sum(axis=0).A1
            doc_freq = (tfidf > 0).sum(axis=0).A1  # comments containing the phrase

            top_idx = agg_tfidf.argsort()[::-1][:top_n]
            return pd.DataFrame({
                "word": names[top_idx],
                "count": doc_freq[top_idx].astype(int),
                "tfidf_score": agg_tfidf[top_idx].round(4),
            }).reset_index(drop=True)
        except ValueError:
            pass  # too few terms — fall through to unigram fallback

    # Fallback: plain unigram frequency (no sklearn or too few docs)
    text = " ".join(texts)
    words = re.findall(r"\b[a-z]{3,}\b", text)
    filtered = [w for w in words if w not in _STOPWORDS]
    counts = Counter(filtered).most_common(top_n)
    return pd.DataFrame(counts, columns=["word", "count"])


def aggregate_sentiment_over_time(
    videos_df: pd.DataFrame, comments_df: pd.DataFrame
) -> pd.DataFrame:
    if comments_df.empty or videos_df.empty:
        return pd.DataFrame()

    # Exclude uncertain comments so they don't compress per-video averages
    # toward zero (see UNCERTAIN_THRESHOLD comment in scoring.py for rationale)
    if "sentiment_label" in comments_df.columns:
        certain = comments_df[comments_df["sentiment_label"] != "Uncertain"].copy()
    else:
        certain = comments_df.copy()

    if certain.empty:
        return pd.DataFrame()

    # Pre-compute per-row log-like weights so the weighted mean can be
    # aggregated with standard groupby (no groupby.apply needed)
    certain["_w"] = np.log1p(certain["like_count"].clip(lower=0))
    certain["_wscore"] = certain["sentiment_score"] * certain["_w"]

    per_video = (
        certain.groupby("video_id")
        .agg(
            _total_w=("_w", "sum"),
            _total_wscore=("_wscore", "sum"),
            flat_avg_sentiment=("sentiment_score", "mean"),
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

    # Where no comment has any likes, fall back to flat mean
    per_video["avg_sentiment"] = per_video.apply(
        lambda r: (
            r["flat_avg_sentiment"]
            if r["_total_w"] == 0
            else r["_total_wscore"] / r["_total_w"]
        ),
        axis=1,
    ).round(4)
    per_video = per_video.drop(columns=["_total_w", "_total_wscore"])

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
    for phrase in keywords:
        parts = phrase.split()
        if len(parts) == 1:
            pattern = r"\b" + re.escape(phrase) + r"\b"
        else:
            # Bigram: word boundary on each end, flexible whitespace between tokens
            pattern = r"\b" + r"\s+".join(re.escape(p) for p in parts) + r"\b"

        mask = comments_df["text"].str.lower().str.contains(pattern, regex=True, na=False)
        subset = comments_df[mask]
        if not subset.empty:
            rows.append(
                {
                    "keyword": phrase,
                    "mentions": len(subset),
                    "avg_sentiment": round(subset["sentiment_score"].mean(), 3),
                    "positive_pct": round(
                        (subset["sentiment_label"] == "Positive").mean() * 100, 1
                    ),
                }
            )
    return pd.DataFrame(rows)
