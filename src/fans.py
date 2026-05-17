"""
Fan community segmentation via k-means on a multi-dimensional feature matrix.

Public API
----------
build_fan_features(comments_df)  →  pd.DataFrame   (one row per commenter)
run_fan_segmentation(comments_df, k_override=None)  →  dict

Output dict keys
----------------
fans_df   : pd.DataFrame   per-commenter features + cluster_id + cluster_label
clusters  : list[dict]     one entry per cluster (see _cluster_record docstring)
k_log     : list[dict]     silhouette + inertia for each k tried
k_chosen  : int

Feature matrix (six dimensions per commenter)
---------------------------------------------
comment_count     total comments in the analyzed video window
videos_commented  distinct videos they commented on
avg_sentiment     mean transformer sentiment_score (Uncertain excluded;
                  falls back to 0.0 when all comments are Uncertain)
likes_earned      sum of like_count across their comments
recency           days since their most recent comment (0 = today,
                  higher = more lapsed — lower z-score is better)
consistency       distinct ISO year-week buckets containing ≥1 comment;
                  a fan active every week scores higher than one who
                  posted 20 comments in a single burst

Scaling choice
--------------
StandardScaler (z-score) is used before k-means.  k-means minimises
within-cluster sum of squared Euclidean distances, so a feature with a
large numeric range (likes_earned ∈ [0, 10 000+]) would dominate a
feature in a small range (avg_sentiment ∈ [-1, +1]) without scaling.
Z-scoring puts every feature on a ±3σ scale so the algorithm treats
them as equally important before the data speaks for itself.

K selection
-----------
Silhouette coefficient is maximised across k=2..8 (capped at
n_fans // MIN_FANS_PER_K so sparse corpora do not over-segment).
Inertia (elbow) is logged alongside each k as a sanity check; it is
NOT used for selection because the "bend" requires subjective detection
while silhouette gives a single comparable scalar.  k is reproducible
via random_state=42 — segments do not reshuffle across ingestion runs.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

# ── Configuration ─────────────────────────────────────────────────────────────

FEATURE_COLS: list[str] = [
    "comment_count",
    "videos_commented",
    "avg_sentiment",
    "likes_earned",
    "recency",
    "consistency",
]

# comment_count, videos_commented, and likes_earned follow a heavy right-tailed
# distribution typical of social media data: most fans have 1 comment / 0 likes,
# a handful have dozens / thousands.  Without transformation, Euclidean distance
# in k-means is dominated by these extremes, collapsing all meaningful structure
# into one giant "everyone else" cluster vs. a tiny outlier cluster.
# log1p compresses the tail so each feature contributes proportionally to
# clustering decisions.  avg_sentiment, recency, and consistency are already
# reasonably bounded and are left untransformed.
_LOG_FEATURES: frozenset[str] = frozenset(["comment_count", "videos_commented", "likes_earned"])

FEATURE_DISPLAY: dict[str, str] = {
    "comment_count":    "Activity (comments)",
    "videos_commented": "Breadth (videos)",
    "avg_sentiment":    "Sentiment",
    "likes_earned":     "Influence (likes earned)",
    "recency":          "Recency (days ago — lower is better)",
    "consistency":      "Consistency (weeks active)",
}

K_RANGE = range(2, 9)
MIN_FANS_PER_K = 8    # require at least this many fans per cluster
RANDOM_STATE = 42     # fixed seed → deterministic segment IDs across runs

# ── Actionability ruleset ─────────────────────────────────────────────────────
#
# Rules are evaluated IN ORDER; first match wins.  Each rule needs:
#   label   human-readable segment name
#   action  recommended marketing action (single sentence)
#   why     brief rationale shown in the UI tooltip
#   color   hex for UI badges
#   conds   list of (feature, op, z_threshold)
#             feature      = key from FEATURE_COLS
#             op           = ">=" | "<="
#             z_threshold  = std-dev units; ±0.3 ≈ moderate, ±0.5 ≈ strong
#
# DESIGN PRINCIPLE: every rule must imply a DIFFERENT recommended action.
# If two rules would yield the same action, merge them.
#
# RECENCY NOTE: recency = days since last comment; high z = lapsed, low z = recent.
#   "recency <= 0.3"  → not meaningfully lapsed (for Advocates)
#   "recency <= -0.3" → significantly more recent than average (Rising Stars)
#   "recency >= 0.5"  → meaningfully lapsed (Lapsed Regulars)
#
_LABEL_RULES: list[dict] = [
    {
        # High influence + positive + still active → community amplifiers.
        # Comes first so a fan with high likes + positive sentiment is always
        # an Advocate, not caught by the broader Loyal Regulars rule.
        "label":  "Advocates",
        "action": "Ambassador outreach — DM for collab, feature their comments in social posts, offer early access",
        "why":    "High reach (liked comments), strongly positive, and still active: the community's natural amplifiers",
        "color":  "#27AE60",
        "conds":  [
            ("likes_earned",  ">=",  0.4),
            ("avg_sentiment", ">=",  0.3),
            ("recency",       "<=",  0.3),
        ],
    },
    {
        # High reach + negative sentiment → influential dissenter.
        # Must come before Loyal Regulars to avoid misclassifying critics
        # who also happen to be consistent commenters.
        "label":  "Engaged Critics",
        "action": "Route to feedback review immediately — reply personally, extract product/content insights; do not ignore",
        "why":    "Influential reach (liked comments) expressing negative sentiment: the community endorses their criticism",
        "color":  "#E74C3C",
        "conds":  [
            ("likes_earned",  ">=",  0.4),
            ("avg_sentiment", "<=", -0.2),
        ],
    },
    {
        # Consistent week-over-week presence, positive, multi-video.
        "label":  "Loyal Regulars",
        "action": "Maintain engagement — reply to their comments; use them as an informal sounding board for content ideas",
        "why":    "Consistent across many weeks, positive, spans multiple videos: the bedrock of the comment section",
        "color":  "#2980B9",
        # avg_sentiment threshold is -0.1 (not +0.2) because this fan base trends
        # uniformly positive; sentiment z-scores cluster near zero so requiring
        # "above average" sentiment would exclude otherwise-clear Loyal Regulars.
        # The defining signal is activity pattern (consistency + multi-video), not
        # whether they're slightly more or less positive than the channel average.
        "conds":  [
            ("consistency",   ">=",  0.3),
            ("avg_sentiment", ">=", -0.1),
            ("comment_count", ">=",  0.2),
        ],
    },
    {
        # Newly active fans: recent, above-average comments, but not yet influential.
        "label":  "Rising Stars",
        "action": "Spotlight and nurture — a public reply now converts casual interest into long-term loyalty before influence peaks",
        "why":    "Recently active, above-average comment frequency, positive, but likes haven't accumulated yet",
        "color":  "#F39C12",
        "conds":  [
            ("recency",       "<=", -0.3),
            ("comment_count", ">=",  0.2),
            ("likes_earned",  "<=",  0.3),
        ],
    },
    {
        # Once spanned many videos but haven't been heard from recently.
        "label":  "Lapsed Regulars",
        "action": "Re-engagement push — milestone content, throwback references, or a direct reply to their last comment",
        "why":    "Once active across many videos but now silent: high churn risk, still recoverable with the right trigger",
        "color":  "#8E44AD",
        "conds":  [
            ("recency",          ">=",  0.5),
            ("videos_commented", ">=",  0.2),
        ],
    },
    {
        # Low-activity, positive-or-neutral-ish fans — the silent majority.
        # Threshold is -0.3 (not 0.0) to robustly catch the typical mass cluster,
        # whose centroid sits near avg_sentiment_z ≈ 0 (population mean); a tiny
        # negative floating-point value would otherwise fall through to Needs Review.
        "label":  "Casual Fans",
        "action": "Nurture with accessible content — no immediate action; monitor for upticks that could indicate promotion to Loyal Regulars",
        "why":    "Low activity and influence but positive-or-neutral sentiment: enjoy the content without deep investment",
        "color":  "#95A5A6",
        "conds":  [
            ("comment_count", "<=",  0.2),
            ("avg_sentiment", ">=", -0.3),
        ],
    },
]
# Applied when NO rule above matches
_FALLBACK_RULE: dict = {
    "label":  "Needs Review",
    "action": "Review manually — this cluster's centroid did not match any action rule; extend the ruleset in fans.py",
    "color":  "#BDC3C7",
}

# ── Feature building ──────────────────────────────────────────────────────────

def build_fan_features(comments_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-commenter feature matrix from clean, scored comments.

    Parameters
    ----------
    comments_df :
        Non-spam, non-duplicate comments with columns:
        author, author_channel_id, video_id, like_count, text,
        published_at, and (optional) sentiment_score, sentiment_label.

    Returns
    -------
    DataFrame with one row per unique (author, author_channel_id) pair
    and columns: author, author_channel_id, + FEATURE_COLS.
    """
    if comments_df.empty:
        return pd.DataFrame()

    df = comments_df.copy()
    df["_pub_dt"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
    # ISO year-week string e.g. "2024-W42" — handles year-boundary correctly
    df["_week"] = df["_pub_dt"].dt.strftime("%G-W%V")

    # Reference date: latest comment in the dataset (stable across runs,
    # unlike using today's date which would make recency drift daily)
    ref_date = df["_pub_dt"].dropna().max()

    GROUP = ["author", "author_channel_id"]

    # Core activity aggregation
    activity = (
        df.groupby(GROUP, as_index=False)
        .agg(
            comment_count    = ("text",      "count"),
            videos_commented = ("video_id",  "nunique"),
            likes_earned     = ("like_count","sum"),
            _last_comment    = ("_pub_dt",   "max"),
            consistency      = ("_week",     "nunique"),
        )
    )
    activity["recency"] = (
        (ref_date - activity["_last_comment"])
        .dt.days
        .fillna(0)
        .clip(lower=0)
        .astype(int)
    )
    activity = activity.drop(columns=["_last_comment"])

    # Sentiment — Uncertain excluded so the average reflects confident signal
    if "sentiment_label" in df.columns and "sentiment_score" in df.columns:
        certain = df[df["sentiment_label"] != "Uncertain"]
        if not certain.empty:
            sent = (
                certain.groupby(GROUP, as_index=False)["sentiment_score"]
                .mean()
                .round(4)
                .rename(columns={"sentiment_score": "avg_sentiment"})
            )
            activity = activity.merge(sent, on=GROUP, how="left")
        else:
            activity["avg_sentiment"] = float("nan")
    else:
        activity["avg_sentiment"] = float("nan")

    # Fill NaN avg_sentiment with 0.0 — equivalent to neutral/missing;
    # after z-scoring this places them exactly at the population mean
    activity["avg_sentiment"] = activity["avg_sentiment"].fillna(0.0)

    return activity.reset_index(drop=True)


# ── K selection ───────────────────────────────────────────────────────────────

def _select_k(X_scaled: np.ndarray, n_fans: int) -> tuple[int, list[dict]]:
    """
    Empirically choose k by maximising silhouette coefficient.

    Returns (best_k, k_log) where k_log contains silhouette + inertia
    for every k evaluated so the caller can inspect the elbow curve.
    """
    k_max = min(max(K_RANGE), n_fans // MIN_FANS_PER_K)
    k_log: list[dict] = []
    best_k = 3
    best_sil = -1.0

    for k in K_RANGE:
        if k > k_max or k >= n_fans:
            break
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init="auto")
        labels = km.fit_predict(X_scaled)

        if len(set(labels)) < 2:
            continue

        sil = silhouette_score(X_scaled, labels)
        inertia = km.inertia_
        k_log.append({
            "k":          k,
            "silhouette": round(sil, 4),
            "inertia":    round(inertia, 2),
        })
        logger.info("fan segmentation  k=%d  silhouette=%.4f  inertia=%.1f", k, sil, inertia)

        if sil > best_sil:
            best_sil = sil
            best_k = k

    return best_k, k_log


# ── Cluster labelling ─────────────────────────────────────────────────────────

def _label_cluster(centroid_z: dict) -> dict:
    """
    Match a centroid's z-score profile to the first matching _LABEL_RULES entry.
    Returns the full rule dict (label, action, color) or _FALLBACK_RULE.
    """
    for rule in _LABEL_RULES:
        if all(
            (centroid_z.get(feat, 0.0) >= thr)
            if op == ">="
            else (centroid_z.get(feat, 0.0) <= thr)
            for feat, op, thr in rule["conds"]
        ):
            return rule
    return _FALLBACK_RULE


def _cluster_record(
    cluster_id: int,
    members: pd.DataFrame,
    centroid_orig: np.ndarray,
    centroid_z: np.ndarray,
    X_scaled_members: np.ndarray,
) -> dict:
    """
    Build the summary dict for a single cluster.

    Returns
    -------
    dict with keys: cluster_id, label, action, color, size,
                    mean_sentiment, centroid, centroid_z, examples
    """
    cent_z_dict = dict(zip(FEATURE_COLS, centroid_z))
    cent_orig_dict = dict(zip(FEATURE_COLS, centroid_orig))
    rule = _label_cluster(cent_z_dict)

    # Representative fans: closest to centroid in z-score space
    if len(X_scaled_members) > 0:
        dists = np.linalg.norm(X_scaled_members - centroid_z, axis=1)
        n_examples = min(5, len(members))
        closest_idx = np.argsort(dists)[:n_examples]
        examples = members.iloc[closest_idx][
            ["author", "comment_count", "videos_commented",
             "avg_sentiment", "likes_earned", "recency", "consistency"]
        ].to_dict("records")
    else:
        examples = []

    # Mean sentiment over cluster members who have a confident label
    mean_sent = float(members["avg_sentiment"].mean()) if not members.empty else float("nan")

    return {
        "cluster_id":    cluster_id,
        "label":         rule["label"],
        "action":        rule["action"],
        "color":         rule["color"],
        "size":          len(members),
        "mean_sentiment": round(mean_sent, 4),
        "centroid":      {k: round(float(v), 3) for k, v in cent_orig_dict.items()},
        "centroid_z":    {k: round(float(v), 3) for k, v in cent_z_dict.items()},
        "examples":      examples,
    }


# ── Public API ────────────────────────────────────────────────────────────────

def run_fan_segmentation(
    comments_df: pd.DataFrame,
    k_override: int | None = None,
) -> dict:
    """
    Build fan features, select k, cluster, label segments.

    This is designed to run during ingestion (inside load_data), not on
    page load — the returned dict is stored in session state and read by
    the Streamlit tab without re-running any computation.

    Parameters
    ----------
    comments_df  : non-spam, non-duplicate, scored comments
    k_override   : force a specific k; bypasses empirical selection

    Returns
    -------
    {
        "fans_df"   : DataFrame  (one row per commenter, + cluster_id, cluster_label)
        "clusters"  : list[dict] (one per cluster, ordered by size desc)
        "k_log"     : list[dict] (silhouette + inertia per k)
        "k_chosen"  : int
    }
    or {} on failure / insufficient data.
    """
    if not _SKLEARN_AVAILABLE:
        logger.warning("scikit-learn not installed — fan segmentation skipped")
        return {}

    fans_df = build_fan_features(comments_df)
    if fans_df.empty or len(fans_df) < MIN_FANS_PER_K * 2:
        logger.warning(
            "Too few unique commenters (%d) for segmentation — need ≥%d",
            len(fans_df), MIN_FANS_PER_K * 2,
        )
        return {}

    # Log-transform right-skewed features, then z-score everything.
    # See _LOG_FEATURES docstring for rationale.
    X = fans_df[FEATURE_COLS].to_numpy(dtype=float)
    for i, col in enumerate(FEATURE_COLS):
        if col in _LOG_FEATURES:
            X[:, i] = np.log1p(X[:, i])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # K selection
    if k_override is not None:
        k = int(k_override)
        k_log: list[dict] = []
        logger.info("fan segmentation  k=%d  (override)", k)
    else:
        k, k_log = _select_k(X_scaled, len(fans_df))
        chosen_entry = next((r for r in k_log if r["k"] == k), None)
        if chosen_entry:
            logger.info(
                "fan segmentation  chosen k=%d  silhouette=%.4f  inertia=%.1f",
                k, chosen_entry["silhouette"], chosen_entry["inertia"],
            )

    # Fit final model
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init="auto")
    fans_df = fans_df.copy()
    fans_df["cluster_id"] = km.fit_predict(X_scaled)

    # Centroids in z-score space and original units.
    # inverse_transform undoes StandardScaler → log-transformed space.
    # expm1 undoes log1p for the right-skewed features → original units.
    centers_z = km.cluster_centers_                             # shape (k, n_features)
    centers_log = scaler.inverse_transform(centers_z)          # log-space for log-features
    centers_orig = centers_log.copy()
    for i, col in enumerate(FEATURE_COLS):
        if col in _LOG_FEATURES:
            centers_orig[:, i] = np.expm1(np.clip(centers_orig[:, i], 0, None))

    # Build per-cluster summary records
    clusters: list[dict] = []
    for cid in range(k):
        mask = fans_df["cluster_id"] == cid
        members = fans_df[mask].reset_index(drop=True)
        X_members = X_scaled[mask.to_numpy()]
        rec = _cluster_record(
            cluster_id       = cid,
            members          = members,
            centroid_orig    = centers_orig[cid],
            centroid_z       = centers_z[cid],
            X_scaled_members = X_members,
        )
        clusters.append(rec)

    # Attach human-readable label to the fan DataFrame
    cid_to_label = {c["cluster_id"]: c["label"] for c in clusters}
    fans_df["cluster_label"] = fans_df["cluster_id"].map(cid_to_label)

    # Order clusters by size descending (largest segment first in UI)
    clusters.sort(key=lambda c: -c["size"])

    return {
        "fans_df":  fans_df,
        "clusters": clusters,
        "k_log":    k_log,
        "k_chosen": k,
    }
