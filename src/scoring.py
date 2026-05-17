"""
Transformer-based comment sentiment + emotion scorer.

Sentiment routing (two models):
  EN / undetermined → cardiffnlp/twitter-roberta-base-sentiment-latest
  Non-EN            → cardiffnlp/twitter-xlm-roberta-base-sentiment

Emotion (one model, run on all comments in the same batched pass):
  j-hartmann/emotion-english-distilroberta-base
  7 classes: anger, disgust, fear, joy, neutral, sadness, surprise

Columns added per comment:
  sentiment_label   (str)   "Positive" | "Neutral" | "Negative" | "Uncertain"
  neg_prob          (float) probability for negative sentiment class
  neu_prob          (float) probability for neutral sentiment class
  pos_prob          (float) probability for positive sentiment class
  sentiment_score   (float) pos_prob - neg_prob, range [-1, +1]
  emotion           (str)   dominant emotion label (absent on VADER path)
  emotion_anger     (float) \
  emotion_disgust   (float)  |
  emotion_fear      (float)  | full 7-class emotion distribution
  emotion_joy       (float)  | (absent on VADER path)
  emotion_neutral   (float)  |
  emotion_sadness   (float)  |
  emotion_surprise  (float) /

Sentiment and emotion run in the same batched loop — the text list is
compiled and sorted once; for each batch we call both pipelines back-to-
back before advancing to the next batch.  This avoids a second O(N) pass
over the corpus and keeps model warm-up costs to a single amortised load.

VADER is retained behind use_vader=True for quick UI preview without
loading any transformer.  The same sentiment column schema is produced;
emotion columns are absent (NaN in the resulting DataFrame).
"""

from __future__ import annotations

ROBERTA_EN_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
ROBERTA_MULTI_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"

# Canonical label order used for emotion columns and UI colour maps
EMOTION_LABELS: tuple[str, ...] = (
    "anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"
)

BATCH_SIZE: int = 32

# If the winning class probability is below this, the comment is labelled
# "Uncertain" instead of Positive / Neutral / Negative.
#
# Why Uncertain is excluded from averages rather than treated as Neutral
# -----------------------------------------------------------------------
# A neutral label means the model is *confident* the comment carries no
# strong sentiment (e.g. high neu_prob, low pos/neg). Uncertain means the
# model's probability mass is split — it genuinely cannot tell. Collapsing
# uncertain into neutral causes signal compression: a comment the model
# cannot read is not the same as one it reads as neutral, and including its
# score (pos − neg ≈ 0) artificially drags the mean toward zero. A video
# with 30% uncertain comments would appear lukewarmly received when the
# signal is simply absent. Excluding uncertain preserves the integrity of
# the aggregate as a measure of what can be *confidently* said about how
# fans feel, rather than a diluted blend of sentiment and noise.
UNCERTAIN_THRESHOLD: float = 0.55

# Lazy singletons — keyed by model name; populated on first call
_pipelines: dict[str, object] = {}


def _get_pipeline(model_name: str):
    if model_name in _pipelines:
        return _pipelines[model_name]
    try:
        from transformers import pipeline as hf_pipeline
        pipe = hf_pipeline(
            "text-classification",
            model=model_name,
            top_k=None,       # return all three class probabilities
            truncation=True,
            max_length=512,
            device=-1,        # CPU; change to 0 for CUDA
        )
        _pipelines[model_name] = pipe
    except Exception:
        _pipelines[model_name] = None
    return _pipelines[model_name]


def _parse_probs(result: list[dict]) -> tuple[float, float, float]:
    """Return (neg, neu, pos) probabilities from a sentiment pipeline result."""
    scores = {r["label"].lower(): r["score"] for r in result}
    pos = scores.get("positive", 0.0)
    neg = scores.get("negative", 0.0)
    neu = scores.get("neutral", 0.0)
    return round(neg, 4), round(neu, 4), round(pos, 4)


def _parse_emotion(result: list[dict]) -> dict:
    """Extract dominant emotion label + full 7-class distribution."""
    scores = {r["label"].lower(): round(r["score"], 4) for r in result}
    dominant = max(scores, key=scores.__getitem__)
    return {
        "emotion": dominant,
        **{f"emotion_{label}": scores.get(label, 0.0) for label in EMOTION_LABELS},
    }


def _label(neg: float, neu: float, pos: float) -> str:
    """Assign a 4-class sentiment label from class probabilities."""
    if max(neg, neu, pos) < UNCERTAIN_THRESHOLD:
        return "Uncertain"
    if pos >= neg and pos >= neu:
        return "Positive"
    if neg >= pos and neg >= neu:
        return "Negative"
    return "Neutral"


def _score_batch_transformer(
    records: list[dict],
    text_col: str,
    batch_size: int,
    run_emotion: bool = True,
) -> list[dict]:
    """
    Route each record to the EN or MULTI sentiment model by language, sort
    within each group by token-length proxy to minimise padding, then run
    sentiment + emotion in the **same** batch loop — one text scan, two
    pipeline calls per batch.
    """
    # Partition by language first — only load multi model if actually needed
    en_indices: list[int] = []
    non_en_indices: list[int] = []
    for i, rec in enumerate(records):
        lang = rec.get("language", "und")
        if lang in ("en", "und"):
            en_indices.append(i)
        else:
            non_en_indices.append(i)

    en_pipe = _get_pipeline(ROBERTA_EN_MODEL)
    multi_pipe = _get_pipeline(ROBERTA_MULTI_MODEL) if non_en_indices else None
    # Emotion model loaded once, shared across both language groups
    emo_pipe = _get_pipeline(EMOTION_MODEL) if run_emotion else None

    # Re-route non-EN to EN if multi model failed to load
    multi_indices: list[int] = []
    for i in non_en_indices:
        if multi_pipe is None:
            en_indices.append(i)
        else:
            multi_indices.append(i)

    results: list[dict] = [{}] * len(records)

    def _run_group(indices: list[int], pipe, ep=emo_pipe) -> None:
        if not indices or pipe is None:
            _vader_fallback(records, indices, text_col, results)
            return

        # Sort by text length (shortest first) to minimise padding waste
        sorted_indices = sorted(
            indices, key=lambda i: len(str(records[i].get(text_col, "")))
        )
        texts = [str(records[i].get(text_col, "")) or " " for i in sorted_indices]

        raw_sent: list = []
        raw_emo: list = []
        # Single pass: sentiment + emotion called on the same batch each iteration
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            raw_sent.extend(pipe(batch, batch_size=batch_size))
            if ep is not None:
                raw_emo.extend(ep(batch, batch_size=batch_size))

        for local_idx, global_idx in enumerate(sorted_indices):
            neg, neu, pos = _parse_probs(raw_sent[local_idx])
            score = round(pos - neg, 4)
            rec_result = {
                "neg_prob": neg,
                "neu_prob": neu,
                "pos_prob": pos,
                "sentiment_score": score,
                "sentiment_label": _label(neg, neu, pos),
            }
            if raw_emo:
                rec_result.update(_parse_emotion(raw_emo[local_idx]))
            results[global_idx] = rec_result

    _run_group(en_indices, en_pipe)
    _run_group(multi_indices, multi_pipe)
    return results


def _vader_fallback(
    records: list[dict],
    indices: list[int],
    text_col: str,
    results: list[dict],
) -> None:
    """Fill results[i] using VADER for the given index subset."""
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    vader = SentimentIntensityAnalyzer()
    for i in indices:
        text = str(records[i].get(text_col, ""))
        s = vader.polarity_scores(text)
        neg, neu, pos = round(s["neg"], 4), round(s["neu"], 4), round(s["pos"], 4)
        score = round(s["compound"], 4)
        results[i] = {
            "neg_prob": neg,
            "neu_prob": neu,
            "pos_prob": pos,
            "sentiment_score": score,
            "sentiment_label": _label(neg, neu, pos),
        }


def _score_all_vader(records: list[dict], text_col: str) -> list[dict]:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    vader = SentimentIntensityAnalyzer()
    out = []
    for rec in records:
        text = str(rec.get(text_col, ""))
        s = vader.polarity_scores(text)
        neg, neu, pos = round(s["neg"], 4), round(s["neu"], 4), round(s["pos"], 4)
        score = round(s["compound"], 4)
        out.append({
            "neg_prob": neg,
            "neu_prob": neu,
            "pos_prob": pos,
            "sentiment_score": score,
            "sentiment_label": _label(neg, neu, pos),
        })
    return out


import re as _re

_SARCASM_RE = _re.compile(
    r"\byeah right\b"
    r"|\bnot\s+\w+\s+(good|great|amazing|best|elite|special|incredible)\b"
    r"|\boverrated+d*\b"
    r"|\blmao\b|\blmfao\b"
    r"|💀|🙄"
    r"|\bsooo+\s+(good|great|amazing|elite)\b",
    _re.IGNORECASE,
)


def count_likely_sarcastic(texts) -> int:
    """
    Count comments matching at least one sarcasm-signal pattern.
    Accepts a pandas Series or any iterable of strings.
    These patterns flag obvious signals only — the count is a lower bound.
    """
    try:
        import pandas as _pd
        s = texts if isinstance(texts, _pd.Series) else _pd.Series(list(texts))
        return int(s.astype(str).str.contains(_SARCASM_RE, regex=True, na=False).sum())
    except Exception:
        return sum(1 for t in texts if _SARCASM_RE.search(str(t)))


def score_comments(
    records: list[dict],
    already_scored_ids: set[str] | None = None,
    text_col: str = "cleaned_text",
    batch_size: int = BATCH_SIZE,
    use_vader: bool = False,
    run_emotion: bool = True,
) -> list[dict]:
    """
    Score comments for sentiment and emotion in a single batched pass.

    Parameters
    ----------
    records:
        Comment dicts. Each must have a ``text_col`` key. If ``language``
        is present, non-EN comments are routed to the multilingual sentiment
        model; all comments share the same English emotion model.
    already_scored_ids:
        Set of comment_id strings already present in the store; matching
        records are passed through unchanged (delta-scoring support).
    text_col:
        Column fed to the models. Default: ``cleaned_text``.
    batch_size:
        Inference batch size. Default 32.
    use_vader:
        Use VADER for sentiment instead of transformers. Emotion columns
        will be absent from the output when True.
    run_emotion:
        Load and run the emotion model alongside sentiment. Set False to
        skip emotion (e.g. warm-up pass, quota-limited environments).

    Returns
    -------
    Same records (new dicts) enriched with:
        sentiment_label, neg_prob, neu_prob, pos_prob, sentiment_score
        emotion, emotion_anger … emotion_surprise  (absent when use_vader=True)
    """
    already_scored_ids = already_scored_ids or set()

    to_score: list[int] = []
    out: list[dict] = [dict(r) for r in records]

    for i, rec in enumerate(out):
        cid = rec.get("comment_id", "")
        if cid and cid in already_scored_ids:
            continue
        to_score.append(i)

    if not to_score:
        return out

    subset = [out[i] for i in to_score]

    if use_vader:
        scored = _score_all_vader(subset, text_col)
    else:
        scored = _score_batch_transformer(
            subset, text_col, batch_size, run_emotion=run_emotion
        )

    for local_i, global_i in enumerate(to_score):
        out[global_i].update(scored[local_i])

    return out
