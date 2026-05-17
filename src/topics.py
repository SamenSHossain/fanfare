"""
Semantic topic modeling for YouTube fan comments via BERTopic.

Public API
----------
run_topic_model(comments_df, embeddings=None) → (topics_df, topic_assignments)

topics_df columns
-----------------
topic_id           int    BERTopic internal ID; -1 = outlier bucket (not returned)
label              str    Human-readable: top-4 keywords joined with " · "
n_comments         int    Total comments assigned to this topic
n_certain          int    Comments with non-Uncertain sentiment label
prominence         float  Σ log(1 + like_count) — like-weighted size, same formula
                          as the global sentiment weighting in analytics.py so both
                          signals are on a comparable scale
weighted_sentiment float  Like-weighted mean sentiment; NaN when n_certain < MIN_N
flat_sentiment     float  Arithmetic mean sentiment; NaN when n_certain < MIN_N
has_sentiment      bool   n_certain >= MIN_TOPIC_SENTIMENT_N
action             str    "✅ Amplify" / "⚠️ Monitor" / "🔵 Neutral" / "—"
examples           list   Up to MAX_EXAMPLES most-liked comments in the topic

topic_assignments  list[int]   Parallel to input comments_df; topic_id per row.

Embedding reuse
---------------
run_topic_model() accepts an optional ``embeddings`` parameter. The current
sentiment models (RoBERTa, XLM-RoBERTa) are classification pipelines that
produce class logits, not reusable sentence vectors, so this module encodes
the corpus with all-MiniLM-L6-v2 by default. If a future component (semantic
search, cosine-similarity deduplication, recommendation) already computes
sentence embeddings over the same corpus, pass them here via ``embeddings``
to skip the redundant encoder pass.

Dimensionality reduction
------------------------
Uses TruncatedSVD (sklearn) rather than UMAP as BERTopic's umap_model.
TruncatedSVD is linear (SVD) and ~5× faster than UMAP on corpora of this
size, with no meaningful loss in cluster quality given that the
all-MiniLM-L6-v2 embeddings are already well-structured.  Switch to UMAP
by replacing the svd= argument with UMAP() in run_topic_model if you need
topology-preserving dimensionality reduction for a much larger corpus.
"""

from __future__ import annotations

try:
    from bertopic import BERTopic
    from bertopic.representation import KeyBERTInspired
    from sentence_transformers import SentenceTransformer
    from sklearn.decomposition import TruncatedSVD
    _BERTOPIC_AVAILABLE = True
except ImportError:
    _BERTOPIC_AVAILABLE = False

import numpy as np
import pandas as pd

from src.analytics import weighted_mean_sentiment

# ── Configuration ─────────────────────────────────────────────────────────────

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Minimum confident (non-Uncertain) comments before we show topic sentiment.
# Below this, weighted_sentiment is NaN and action is "—".  Protects against
# publishing sentiment averages backed by a handful of noisy scores.
MIN_TOPIC_SENTIMENT_N: int = 20

# BERTopic: minimum documents per cluster.  Smaller = more fine-grained topics.
# At 9 500 comments this typically yields 15–30 usable topics.
_MIN_CLUSTER_SIZE: int = 20

# Reduced dimensionality before HDBSCAN (n_components for TruncatedSVD).
# Higher = more variance preserved; lower = faster clustering.  10 is a
# reasonable middle ground for 384-dim sentence-transformer output.
_SVD_COMPONENTS: int = 10

# Target number of topics after BERTopic's hierarchical merging pass.
# "auto" disables merging; a fixed int caps the final count.
_NR_TOPICS: int | str = 25

MAX_EXAMPLES: int = 3


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_label(keywords: list[tuple[str, float]], n: int = 4) -> str:
    """Turn the top-n BERTopic keywords into a readable topic label."""
    words = [w for w, _ in keywords[:n] if len(w) > 2]
    return " · ".join(w.capitalize() for w in words) if words else "General"


def _action_signal(ws: float, has_sentiment: bool, n_certain: int) -> str:
    if not has_sentiment:
        return f"— (n={n_certain})"
    if ws >= 0.10:
        return "✅ Amplify"
    if ws <= -0.10:
        return "⚠️ Monitor"
    return "🔵 Neutral"


def _build_topics_df(
    comments_df: pd.DataFrame,
    topic_assignments: list[int],
    topic_model: "BERTopic",
) -> pd.DataFrame:
    """Aggregate per-topic stats and enforce the sample guard."""
    df = comments_df.reset_index(drop=True).copy()
    df["_tid"] = topic_assignments

    rows: list[dict] = []

    for tid, group in df.groupby("_tid"):
        if tid == -1:
            continue  # outlier bucket — not displayed

        # Label from BERTopic keyword representation
        try:
            raw_keywords = topic_model.get_topic(tid) or []
        except Exception:
            raw_keywords = []
        label = _make_label(raw_keywords)

        n_comments = len(group)

        # Certain = comments the model confidently classified
        if "sentiment_label" in group.columns:
            certain = group[group["sentiment_label"] != "Uncertain"]
        else:
            certain = group
        n_certain = len(certain)

        # Prominence: Σ log(1 + like_count) — same weighting as the global
        # sentiment metric so headline and topic-level numbers are comparable
        prominence = float(np.log1p(group["like_count"].clip(lower=0)).sum())

        # Sample guard
        has_sentiment = n_certain >= MIN_TOPIC_SENTIMENT_N
        if has_sentiment and "sentiment_score" in certain.columns and not certain.empty:
            ws, fs = weighted_mean_sentiment(certain)
        else:
            ws, fs = float("nan"), float("nan")

        action = _action_signal(ws, has_sentiment, n_certain)

        # Representative examples: most-liked comments in the topic
        examples = (
            group.nlargest(MAX_EXAMPLES, "like_count")["text"]
            .fillna("")
            .tolist()
        )

        rows.append(
            {
                "topic_id":           tid,
                "label":              label,
                "n_comments":         n_comments,
                "n_certain":          n_certain,
                "prominence":         round(prominence, 3),
                "weighted_sentiment": round(ws, 4) if has_sentiment else float("nan"),
                "flat_sentiment":     round(fs, 4) if has_sentiment else float("nan"),
                "has_sentiment":      has_sentiment,
                "action":             action,
                "examples":           examples,
            }
        )

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .sort_values("prominence", ascending=False)
        .reset_index(drop=True)
    )


# ── Public API ────────────────────────────────────────────────────────────────

def run_topic_model(
    comments_df: pd.DataFrame,
    embeddings: np.ndarray | None = None,
    nr_topics: int | str = _NR_TOPICS,
    min_topic_size: int = _MIN_CLUSTER_SIZE,
) -> tuple[pd.DataFrame, list[int]]:
    """
    Fit BERTopic on clean fan comments; return (topics_df, topic_assignments).

    Parameters
    ----------
    comments_df:
        Non-spam, non-duplicate comments. Required columns: cleaned_text,
        like_count, text.  Optional but used when present: sentiment_score,
        sentiment_label, author.
    embeddings:
        Optional pre-computed sentence embeddings (shape N × D).  Pass to
        avoid re-encoding when sentence vectors already exist from another
        pipeline step.  If None, vectors are computed from EMBEDDING_MODEL.
    nr_topics:
        Target topic count after BERTopic's hierarchical merge.  "auto"
        skips merging; an int caps the result.  Default: _NR_TOPICS.
    min_topic_size:
        Minimum documents per HDBSCAN cluster.  Larger = fewer, broader
        topics.  Default: _MIN_CLUSTER_SIZE.

    Returns
    -------
    topics_df : DataFrame
        One row per topic with stats (see module docstring).  Sorted by
        prominence descending.  Empty if modelling fails or corpus too small.
    topic_assignments : list[int]
        Parallel to comments_df rows; BERTopic topic ID for each comment.
        -1 = outlier (comment not assigned to any topic).
    """
    if not _BERTOPIC_AVAILABLE:
        return pd.DataFrame(), []

    texts = comments_df["cleaned_text"].fillna("").astype(str).tolist()

    if len(texts) < min_topic_size * 2:
        # Too few documents to form any meaningful cluster
        return pd.DataFrame(), [-1] * len(texts)

    # ── 1. Sentence embeddings ────────────────────────────────────────────────
    # The embedder must be passed to BERTopic even when supplying pre-computed
    # embeddings, because KeyBERTInspired re-embeds representative documents
    # internally to extract semantically coherent keywords.
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    if embeddings is None:
        embeddings = embedder.encode(
            texts,
            show_progress_bar=False,
            batch_size=64,
            normalize_embeddings=True,
        )

    # ── 2. BERTopic setup ────────────────────────────────────────────────────
    # TruncatedSVD replaces UMAP (see module docstring for rationale).
    # n_components=10 preserves more variance than the BERTopic default of 5.
    svd = TruncatedSVD(n_components=_SVD_COMPONENTS, random_state=42)

    # KeyBERTInspired produces more semantically coherent keyword labels than
    # the default c-TF-IDF representation.
    representation = KeyBERTInspired()

    topic_model = BERTopic(
        embedding_model=embedder,
        umap_model=svd,
        representation_model=representation,
        language="english",
        min_topic_size=min_topic_size,
        nr_topics=nr_topics,
        calculate_probabilities=False,
        verbose=False,
    )

    try:
        topic_assignments, _ = topic_model.fit_transform(texts, embeddings=embeddings)
    except Exception:
        return pd.DataFrame(), [-1] * len(texts)

    # ── 3. Build output DataFrame ─────────────────────────────────────────────
    topics_df = _build_topics_df(comments_df, list(topic_assignments), topic_model)

    return topics_df, list(topic_assignments)
