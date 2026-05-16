import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()


def analyze_sentiment(text: str) -> dict:
    scores = _analyzer.polarity_scores(str(text))
    compound = scores["compound"]
    if compound >= 0.05:
        label = "Positive"
    elif compound <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"
    return {
        "label": label,
        "compound": round(compound, 4),
        "pos": scores["pos"],
        "neg": scores["neg"],
        "neu": scores["neu"],
    }


def batch_analyze(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    results = df[text_col].apply(analyze_sentiment)
    df = df.copy()
    df["sentiment_label"] = results.apply(lambda r: r["label"])
    df["sentiment_score"] = results.apply(lambda r: r["compound"])
    df["sentiment_pos"] = results.apply(lambda r: r["pos"])
    df["sentiment_neg"] = results.apply(lambda r: r["neg"])
    df["sentiment_neu"] = results.apply(lambda r: r["neu"])
    return df
