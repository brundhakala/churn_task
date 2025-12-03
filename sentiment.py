from textblob import TextBlob

def analyze_sentiment(text: str) -> float:
    """
    Returns polarity between -1 (negative) and +1 (positive)
    """
    if not text:
        return 0.0
    tb = TextBlob(text)
    return tb.sentiment.polarity
