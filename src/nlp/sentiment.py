from functools import lru_cache

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


@lru_cache(maxsize=1)
def _analyzer() -> SentimentIntensityAnalyzer:
    return SentimentIntensityAnalyzer()


def analyze_sentiment(text: str) -> tuple[str, float]:
    """Return (label, compound_score) for a piece of text using VADER.

    VADER is lexicon-based, so no model download or GPU is required and the
    pipeline can run fully offline out of the box.
    """
    score = _analyzer().polarity_scores(text)["compound"]
    if score >= 0.05:
        label = "positive"
    elif score <= -0.05:
        label = "negative"
    else:
        label = "neutral"
    return label, score
