import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_vader = SentimentIntensityAnalyzer()

# RoBERTa state — loaded once per process, never reloaded on failure
_roberta_pipe = None
_roberta_attempted = False
ROBERTA_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"


def _load_roberta():
    global _roberta_pipe, _roberta_attempted
    if _roberta_attempted:
        return _roberta_pipe
    _roberta_attempted = True
    try:
        from transformers import pipeline as hf_pipeline
        _roberta_pipe = hf_pipeline(
            "text-classification",
            model=ROBERTA_MODEL,
            top_k=None,       # return all three class probabilities
            truncation=True,  # silently truncate at 512 tokens
            max_length=512,
            device=-1,        # CPU; set to 0 for CUDA
        )
    except Exception:
        _roberta_pipe = None
    return _roberta_pipe


def roberta_available() -> bool:
    return _load_roberta() is not None


def _label(compound: float) -> str:
    if compound >= 0.05:
        return "Positive"
    if compound <= -0.05:
        return "Negative"
    return "Neutral"


def _parse_roberta(result: list[dict]) -> dict:
    """Convert [{label, score}, ...] to {pos, neg, neu, compound}."""
    scores = {r["label"].lower(): r["score"] for r in result}
    pos = scores.get("positive", 0.0)
    neg = scores.get("negative", 0.0)
    neu = scores.get("neutral", 0.0)
    # compound maps naturally to [-1, +1]: strong positive → near +1, strong negative → near -1
    return {
        "pos": round(pos, 4),
        "neg": round(neg, 4),
        "neu": round(neu, 4),
        "compound": round(pos - neg, 4),
    }


def batch_analyze(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    df = df.copy()
    texts = df[text_col].astype(str).tolist()

    # VADER — always runs; cheap baseline and useful for disagreement analysis
    vader_scores = [_vader.polarity_scores(t) for t in texts]
    df["vader_compound"] = [s["compound"] for s in vader_scores]

    pipe = _load_roberta()
    if pipe is not None:
        # RoBERTa is primary: fine-tuned on Twitter text, handles slang and context
        raw = pipe(texts, batch_size=32)
        roberta = [_parse_roberta(r) for r in raw]

        df["roberta_compound"] = [r["compound"] for r in roberta]
        df["sentiment_score"] = df["roberta_compound"]
        df["sentiment_pos"] = [r["pos"] for r in roberta]
        df["sentiment_neg"] = [r["neg"] for r in roberta]
        df["sentiment_neu"] = [r["neu"] for r in roberta]
    else:
        # VADER fallback when transformer is unavailable
        df["sentiment_score"] = df["vader_compound"]
        df["sentiment_pos"] = [s["pos"] for s in vader_scores]
        df["sentiment_neg"] = [s["neg"] for s in vader_scores]
        df["sentiment_neu"] = [s["neu"] for s in vader_scores]

    df["sentiment_label"] = df["sentiment_score"].apply(_label)
    return df
