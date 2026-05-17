"""
Statistically honest alerting with per-family multiple-comparisons correction.

Three alert families
--------------------
sentiment_spike   One-sample t-test (scipy) per video vs. channel-wide mean.
                  Baseline: flat mean over all certain (non-Uncertain) comments.
                  m = number of videos with >= MIN_VIDEO_COMMENTS certain comments.

velocity_anomaly  Z-score of each video's total YouTube comment count vs. the
                  channel distribution.  No formal p-value — comment counts are
                  right-skewed (Poisson-ish), so a z-score is more honest than
                  a t-test.  The correction translates into a stricter sigma
                  cutoff (see below).  Uses videos_df["comment_count"] (real
                  YouTube totals, not our fetched sample) so a viral video is
                  caught even if we only downloaded 80 of its 2 000 comments.
                  m = total videos in the corpus.

keyword_shift     One-sample t-test per keyword vs. channel-wide mean.
                  Keywords = top KEYWORD_TOP_N by TF-IDF/frequency from the
                  corpus.  Only keywords with >= MIN_KEYWORD_MENTIONS certain-
                  comment mentions enter the family; the family size m is the
                  number actually tested.

Multiple-comparisons correction
--------------------------------
BONFERRONI (default, CORRECTION_METHOD = "bonferroni")
    Per-test α = ALPHA / m.  Strictly controls FWER — the probability of any
    false positive in the family is ≤ ALPHA.
    Trade-off: can be too conservative when many real changes occur
    simultaneously (a busy match week would produce zero alerts even though
    genuine shifts are happening).

BH — Benjamini-Hochberg (CORRECTION_METHOD = "bh")
    Controls FDR ≤ ALPHA: the *expected fraction* of fired alerts that are
    false positives is ≤ ALPHA.  Catches more real signals.  Recommended when
    the marketing lead would rather catch most real shifts than almost none,
    and can tolerate ~ALPHA of alerts being spurious.
    Trade-off: on a day with no true change, up to ALPHA fraction of tests
    may still fire.

Velocity z-score → corrected threshold mapping
------------------------------------------------
Under a normal approximation, a two-tailed p-value maps to z via
    p = 2 · (1 − Φ(z))
Bonferroni equivalent: z* = Φ⁻¹(1 − α/(2m)).
BH equivalent: rank the z-derived normal-approx p-values and apply BH.
This approximation is conservative (comment-count tails are heavier than
normal), so the threshold is higher than naive — we tend to under-fire, not
over-fire, which is the safe direction for a false-positive-sensitive feed.

Alert contract
--------------
Every fired alert carries:
  family, title, magnitude, n (sample size), p_raw, corrected_threshold,
  correction_method, direction, action, severity.
No alert fires without an action string.  Effect-size filter (MIN_* constants)
is applied *after* statistical correction — never used to pre-select which tests
to run (that would inflate power dishonestly).
"""

from __future__ import annotations

import re
from typing import Literal

import numpy as np
import pandas as pd

try:
    from scipy import stats as _scipy_stats
    _SCIPY = True
except ImportError:
    _SCIPY = False

from src.analytics import get_trending_topics

# ── Tunable config ─────────────────────────────────────────────────────────────

ALPHA: float = 0.05

# "bonferroni": strict, controls FWER (default).
# "bh": less strict, controls FDR — fires more when many true signals exist.
CORRECTION_METHOD: Literal["bonferroni", "bh"] = "bonferroni"

MIN_VIDEO_COMMENTS: int = 30    # per-video sample guard (certain comments)
MIN_KEYWORD_MENTIONS: int = 15  # per-keyword sample guard
KEYWORD_TOP_N: int = 20         # keywords included in the family

# Minimum absolute effect size to surface an alert (applied after correction).
# Prevents trivially small but statistically significant differences from firing.
MIN_SENTIMENT_DELTA: float = 0.08   # |video_mean − channel_mean|
MIN_VELOCITY_Z: float = 2.5          # absolute z-score floor for velocity


# ── BH correction ──────────────────────────────────────────────────────────────

def _bh_mask(p_values: np.ndarray) -> np.ndarray:
    """
    Standard Benjamini-Hochberg procedure.
    Sort p-values ascending; find the largest rank k where p_(k) <= k·α/m;
    reject all ranks 1..k.  Returns boolean mask (True = reject H0).
    """
    n = len(p_values)
    if n == 0:
        return np.array([], dtype=bool)
    order = np.argsort(p_values)
    sorted_p = p_values[order]
    bh_levels = ALPHA * np.arange(1, n + 1) / n
    passing = sorted_p <= bh_levels
    if not passing.any():
        return np.zeros(n, dtype=bool)
    k = int(np.where(passing)[0][-1]) + 1   # number of rejections
    result = np.zeros(n, dtype=bool)
    result[order[:k]] = True
    return result


# ── Action templates ───────────────────────────────────────────────────────────

def _sent_action(delta: float, title: str) -> str:
    short = title[:55]
    if delta < 0:
        return (
            f'Sentiment for "{short}" is significantly below the channel average. '
            "Review the most critical comments on this video, identify the recurring "
            "concern, and address it in a community post or follow-up video."
        )
    return (
        f'Sentiment for "{short}" is significantly above the channel average. '
        "Identify the hook, tone, and topic that drove this reaction so you can "
        "replicate the format."
    )


def _velocity_action(n: int, title: str, baseline: float) -> str:
    short = title[:55]
    return (
        f'"{short}" has {n:,} total comments vs. a channel baseline of {baseline:.0f}. '
        "Pin a comment to guide the conversation and consider a follow-up post "
        "capitalising on the momentum."
    )


def _kw_action(delta: float, keyword: str) -> str:
    if delta < 0:
        return (
            f'Fans are using "{keyword}" in a more negative context than usual. '
            "Read the comments mentioning this keyword and draft a response or "
            "content clarification."
        )
    return (
        f'"{keyword}" is driving unusually strong positive engagement. '
        "Lean into it in upcoming content, captions, and titles."
    )


# ── Sample comment helper ─────────────────────────────────────────────────────

def _top_comments(df: pd.DataFrame, n: int = 3) -> list[dict]:
    """Return up to n most-liked comments as plain dicts for embedding in alerts."""
    if df.empty or "text" not in df.columns:
        return []
    cols = [c for c in ("author", "text", "like_count") if c in df.columns]
    rows = df.nlargest(n, "like_count") if "like_count" in df.columns else df.head(n)
    out = []
    for _, r in rows[cols].iterrows():
        out.append({
            "author":     str(r.get("author", "fan")),
            "text":       str(r.get("text", ""))[:280],
            "like_count": int(r.get("like_count", 0)),
        })
    return out


# ── Family 1: Sentiment spike ──────────────────────────────────────────────────

def _sentiment_spike_family(
    comments_df: pd.DataFrame,
    videos_df: pd.DataFrame,
    correction: str,
) -> tuple[list[dict], dict]:
    """One-sample t-test per video vs. channel mean. Requires scipy."""
    meta: dict = {"m": 0, "tested": 0, "naive_pass": 0, "corrected_pass": 0,
                  "description": "sentiment t-test per video vs. channel mean"}

    if not _SCIPY or comments_df.empty:
        return [], meta

    certain = (
        comments_df[comments_df["sentiment_label"] != "Uncertain"]
        if "sentiment_label" in comments_df.columns
        else comments_df
    )
    if certain.empty or "sentiment_score" not in certain.columns:
        return [], meta

    channel_mean = float(certain["sentiment_score"].mean())

    vid_title: dict = {}
    if not videos_df.empty and "video_id" in videos_df.columns and "title" in videos_df.columns:
        vid_title = dict(zip(videos_df["video_id"], videos_df["title"]))

    # Qualify videos by sample guard (no effect-size pre-filter here)
    groups = [
        (vid, grp)
        for vid, grp in certain.groupby("video_id")
        if len(grp) >= MIN_VIDEO_COMMENTS
    ]
    m = len(groups)
    meta["m"] = m
    meta["tested"] = m
    if m == 0:
        return [], meta

    # Run all t-tests; collect regardless of effect size
    records = []
    for vid, grp in groups:
        scores = grp["sentiment_score"].to_numpy(dtype=float)
        delta = float(np.mean(scores)) - channel_mean
        _, p_val = _scipy_stats.ttest_1samp(scores, popmean=channel_mean)
        records.append({
            "video_id": vid,
            "title": vid_title.get(vid, vid),
            "n": len(scores),
            "delta": round(delta, 4),
            "p_raw": p_val,
        })

    p_arr = np.array([r["p_raw"] for r in records])

    # Naive count (uncorrected α = 0.05)
    meta["naive_pass"] = int((p_arr < ALPHA).sum())

    # Apply correction
    if correction == "bonferroni":
        threshold = ALPHA / m
        pass_mask = p_arr < threshold
        # Bonferroni-adjusted p = min(p_raw * m, 1)
        adj_p = np.minimum(p_arr * m, 1.0)
    else:  # bh
        pass_mask = _bh_mask(p_arr)
        order = np.argsort(p_arr)
        ranks = np.empty(len(p_arr), dtype=int)
        ranks[order] = np.arange(1, len(p_arr) + 1)
        adj_p = np.minimum(p_arr * len(p_arr) / ranks, 1.0)
        threshold = ALPHA  # BH controls FDR at this level

    meta["corrected_pass"] = int(pass_mask.sum())

    alerts: list[dict] = []
    for i, (rec, passed) in enumerate(zip(records, pass_mask)):
        if not passed:
            continue
        # Effect-size gate applied after correction
        if abs(rec["delta"]) < MIN_SENTIMENT_DELTA:
            continue
        action = _sent_action(rec["delta"], rec["title"])
        # Pull the most-liked comments from this video that drove the signal.
        # For negative spikes use the lowest-scoring comments; for positive use highest.
        vid_certain = certain[certain["video_id"] == rec["video_id"]]
        if rec["delta"] < 0:
            sample_rows = vid_certain.nsmallest(3, "sentiment_score")
        else:
            sample_rows = vid_certain.nlargest(3, "like_count")
        alerts.append({
            "family":              "sentiment_spike",
            "title":               f"Sentiment {'drop' if rec['delta'] < 0 else 'spike'}: \"{rec['title'][:55]}\"",
            "video_id":            rec["video_id"],
            "magnitude":           rec["delta"],
            "magnitude_label":     f"{rec['delta']:+.3f} vs. channel mean ({channel_mean:+.3f})",
            "n":                   rec["n"],
            "p_raw":               round(rec["p_raw"], 6),
            "p_adj":               round(float(adj_p[i]), 6),
            "corrected_threshold": round(threshold, 6) if correction == "bonferroni" else ALPHA,
            "correction_method":   correction,
            "direction":           "negative" if rec["delta"] < 0 else "positive",
            "action":              action,
            "severity":            "warning" if rec["delta"] < 0 else "info",
            "sample_comments":     _top_comments(sample_rows),
        })

    return alerts, meta


# ── Family 2: Velocity anomaly ─────────────────────────────────────────────────

def _velocity_anomaly_family(
    comments_df: pd.DataFrame,
    videos_df: pd.DataFrame,
    correction: str,
) -> tuple[list[dict], dict]:
    """
    Z-score of each video's total YouTube comment count vs. channel distribution.

    Uses videos_df["comment_count"] (real YouTube totals) so a viral video is
    caught even if we only fetched a small sample of its comments.

    Bonferroni sigma mapping:
        z* = Φ⁻¹(1 − α/(2m))   [two-tailed, normal approximation]
    BH: rank normal-approx p-values derived from z-scores and apply BH.

    The normal approximation is conservative (heavier tails → we under-fire).
    """
    meta: dict = {"m": 0, "tested": 0, "naive_pass": 0, "corrected_pass": 0,
                  "description": "z-score of video comment count vs. channel distribution"}

    if videos_df.empty or "comment_count" not in videos_df.columns:
        return [], meta

    vid_title: dict = {}
    if "video_id" in videos_df.columns and "title" in videos_df.columns:
        vid_title = dict(zip(videos_df["video_id"], videos_df["title"]))

    counts = videos_df["comment_count"].fillna(0).astype(float)
    m = len(counts)
    meta["m"] = m
    meta["tested"] = m
    if m < 3:
        return [], meta

    mu = float(counts.mean())
    sigma = float(counts.std(ddof=1))
    if sigma == 0:
        return [], meta

    z_scores = (counts - mu) / sigma

    # Normal-approx two-tailed p-values for ranking / BH
    z_arr = z_scores.to_numpy()  # work in numpy throughout
    if _SCIPY:
        p_approx = 2.0 * _scipy_stats.norm.sf(np.abs(z_arr))
        bonf_sigma = float(_scipy_stats.norm.ppf(1.0 - ALPHA / (2.0 * m)))
    else:
        p_approx = 2.0 * np.exp(-0.5 * z_arr ** 2) / np.sqrt(2.0 * np.pi)
        bonf_sigma = 3.0  # conservative fallback

    # Naive threshold: |z| > 2.0 (conventional two-sigma)
    meta["naive_pass"] = int((np.abs(z_arr) > 2.0).sum())

    p_arr = p_approx

    if correction == "bonferroni":
        threshold_sigma = bonf_sigma
        pass_mask_arr = np.abs(z_arr) >= threshold_sigma
        threshold_label = f"|z| ≥ {threshold_sigma:.3f}σ (Bonferroni α/{m})"
    else:  # bh
        pass_mask_arr = _bh_mask(p_arr)
        threshold_sigma = bonf_sigma  # shown for reference
        threshold_label = f"BH FDR ≤ {ALPHA:.0%}"

    meta["corrected_pass"] = int(pass_mask_arr.sum())

    alerts: list[dict] = []
    for idx, (vid_row, z_val, p_val, passed) in enumerate(
        zip(videos_df.itertuples(index=False), z_arr, p_arr, pass_mask_arr)
    ):
        if not passed:
            continue
        if abs(z_val) < MIN_VELOCITY_Z:
            continue
        vid = vid_row.video_id if hasattr(vid_row, "video_id") else str(idx)
        n = int(getattr(vid_row, "comment_count", 0))
        title = vid_title.get(vid, vid)
        action = _velocity_action(n, title, mu)
        vid_comments = (
            comments_df[comments_df["video_id"] == vid]
            if not comments_df.empty and "video_id" in comments_df.columns
            else pd.DataFrame()
        )
        alerts.append({
            "family":              "velocity_anomaly",
            "title":               f"Comment surge: \"{title[:55]}\"",
            "video_id":            vid,
            "magnitude":           round(float(z_val), 2),
            "magnitude_label":     f"{n:,} comments ({z_val:+.2f}σ vs. channel mean {mu:.0f})",
            "n":                   n,
            "p_raw":               round(float(p_val), 6),
            "p_adj":               round(float(p_val) * m, 6) if correction == "bonferroni" else round(float(p_val), 6),
            "corrected_threshold": threshold_label,
            "correction_method":   f"{correction}_sigma",
            "direction":           "high" if z_val > 0 else "low",
            "action":              action,
            "severity":            "info",
            "sample_comments":     _top_comments(vid_comments),
        })

    return alerts, meta


# ── Family 3: Keyword sentiment shift ─────────────────────────────────────────

def _keyword_shift_family(
    comments_df: pd.DataFrame,
    correction: str,
) -> tuple[list[dict], dict]:
    """
    One-sample t-test per top keyword vs. channel-wide mean sentiment.
    Keywords = top KEYWORD_TOP_N by TF-IDF/frequency from the full corpus.
    Family size m = number of qualifying keywords (>= MIN_KEYWORD_MENTIONS).
    """
    meta: dict = {"m": 0, "tested": 0, "naive_pass": 0, "corrected_pass": 0,
                  "description": f"sentiment t-test for top-{KEYWORD_TOP_N} keywords vs. channel mean"}

    if not _SCIPY or comments_df.empty:
        return [], meta

    if "sentiment_score" not in comments_df.columns or "text" not in comments_df.columns:
        return [], meta

    certain = (
        comments_df[comments_df["sentiment_label"] != "Uncertain"]
        if "sentiment_label" in comments_df.columns
        else comments_df
    )
    if certain.empty:
        return [], meta

    channel_mean = float(certain["sentiment_score"].mean())

    kw_df = get_trending_topics(comments_df, top_n=KEYWORD_TOP_N)
    if kw_df.empty:
        return [], meta

    keywords = kw_df["word"].tolist()

    # Qualify keywords by sample guard (no effect-size pre-filter)
    records = []
    for kw in keywords:
        pattern = r"\b" + re.escape(kw) + r"\b"
        mask = certain["text"].str.lower().str.contains(pattern, regex=True, na=False)
        subset = certain[mask]
        if len(subset) < MIN_KEYWORD_MENTIONS:
            continue
        scores = subset["sentiment_score"].to_numpy(dtype=float)
        delta = float(np.mean(scores)) - channel_mean
        _, p_val = _scipy_stats.ttest_1samp(scores, popmean=channel_mean)
        records.append({
            "keyword": kw,
            "n": len(scores),
            "delta": round(delta, 4),
            "p_raw": p_val,
            "subset": subset,   # kept for sample comment extraction below
        })

    m = len(records)
    meta["m"] = m
    meta["tested"] = m
    if m == 0:
        return [], meta

    p_arr = np.array([r["p_raw"] for r in records])
    meta["naive_pass"] = int((p_arr < ALPHA).sum())

    if correction == "bonferroni":
        threshold = ALPHA / m
        pass_mask = p_arr < threshold
        adj_p = np.minimum(p_arr * m, 1.0)
    else:  # bh
        pass_mask = _bh_mask(p_arr)
        order = np.argsort(p_arr)
        ranks = np.empty(len(p_arr), dtype=int)
        ranks[order] = np.arange(1, len(p_arr) + 1)
        adj_p = np.minimum(p_arr * len(p_arr) / ranks, 1.0)
        threshold = ALPHA

    meta["corrected_pass"] = int(pass_mask.sum())

    alerts: list[dict] = []
    for i, (rec, passed) in enumerate(zip(records, pass_mask)):
        if not passed:
            continue
        # Effect-size gate after correction
        if abs(rec["delta"]) < MIN_SENTIMENT_DELTA:
            continue
        action = _kw_action(rec["delta"], rec["keyword"])
        subset = rec["subset"]
        # For negative keywords show the most critical; positive show most endorsed
        if rec["delta"] < 0:
            sample_rows = subset.nsmallest(3, "sentiment_score")
        else:
            sample_rows = subset.nlargest(3, "like_count")
        alerts.append({
            "family":              "keyword_shift",
            "title":               f"Keyword signal: \"{rec['keyword']}\"",
            "video_id":            None,
            "magnitude":           rec["delta"],
            "magnitude_label":     f"{rec['delta']:+.3f} vs. channel mean ({channel_mean:+.3f}), n={rec['n']} mentions",
            "n":                   rec["n"],
            "p_raw":               round(rec["p_raw"], 6),
            "p_adj":               round(float(adj_p[i]), 6),
            "corrected_threshold": round(threshold, 6) if correction == "bonferroni" else ALPHA,
            "correction_method":   correction,
            "direction":           "negative" if rec["delta"] < 0 else "positive",
            "action":              action,
            "severity":            "warning" if rec["delta"] < -0.10 else "info",
            "sample_comments":     _top_comments(sample_rows),
        })

    return alerts, meta


# ── Public API ─────────────────────────────────────────────────────────────────

def run_alerts(
    comments_df: pd.DataFrame,
    videos_df: pd.DataFrame,
    *,
    correction: str = CORRECTION_METHOD,
) -> dict:
    """
    Run all three alert families with multiple-comparisons correction.

    Parameters
    ----------
    comments_df : clean, scored comments (must have sentiment_label, sentiment_score)
    videos_df   : video metadata (must have video_id, title, comment_count)
    correction  : "bonferroni" (default) or "bh"

    Returns
    -------
    {
        "alerts":    list[dict]   — fired alerts; every entry has action, n, threshold
        "summary":   {
            "tested":      int  — total hypothesis tests run across all families
            "passed":      int  — alerts that cleared correction + effect-size gate
            "naive_count": int  — how many would have fired without correction (α=0.05 / |z|>2)
        }
        "families":  dict  — per-family metadata (m, tested, naive_pass, corrected_pass)
        "correction": str  — method used
    }
    """
    sent_alerts, sent_meta = _sentiment_spike_family(comments_df, videos_df, correction)
    vel_alerts,  vel_meta  = _velocity_anomaly_family(comments_df, videos_df, correction)
    kw_alerts,   kw_meta   = _keyword_shift_family(comments_df, correction)

    all_alerts = sent_alerts + vel_alerts + kw_alerts

    total_tested = sent_meta["m"] + vel_meta["m"] + kw_meta["m"]
    naive_total  = sent_meta["naive_pass"] + vel_meta["naive_pass"] + kw_meta["naive_pass"]

    return {
        "alerts":    all_alerts,
        "summary":   {
            "tested":      total_tested,
            "passed":      len(all_alerts),
            "naive_count": naive_total,
        },
        "families":  {
            "sentiment_spike":  sent_meta,
            "velocity_anomaly": vel_meta,
            "keyword_shift":    kw_meta,
        },
        "correction":  correction,
        "fetched_at":  pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
    }
