"""
sentiment_analysis.py
---------------------
Sentiment analysis using a pre-trained Hugging Face pipeline.

The pipeline loads 'distilbert-base-uncased-finetuned-sst-2-english' by default,
classifying text as POSITIVE or NEGATIVE with a confidence score.

Usage:
    python sentiment_analysis.py
"""

from transformers import pipeline


def run_sentiment_analysis():
    """
    Load a pre-trained sentiment-analysis pipeline and classify example texts.

    Prints the predicted label and confidence score for each input.
    """
    print("Loading sentiment-analysis model...")
    sentiment_analyzer = pipeline("sentiment-analysis")
    print("Model loaded successfully.\n")

    texts = [
        "I love this movie, it's absolutely fantastic!",
        "This is the worst experience I've ever had.",
        "The weather is okay today.",
        "PyTorch makes deep learning research incredibly accessible.",
        "I'm not sure whether I like this or not.",
    ]

    print("=" * 60)
    print("  Sentiment Analysis Results")
    print("=" * 60)

    for text in texts:
        result = sentiment_analyzer(text)[0]
        emoji = "😊" if result["label"] == "POSITIVE" else "😞"
        print(f'\n{emoji} "{text}"')
        print(f'   → Label: {result["label"]}  |  Confidence: {result["score"]:.4f}')

    print("\nDone.")


if __name__ == "__main__":
    run_sentiment_analysis()
