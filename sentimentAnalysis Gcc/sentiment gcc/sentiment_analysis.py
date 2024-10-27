# Install NLTK and download the VADER lexicon
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download the VADER lexicon (if not already downloaded)
nltk.download('vader_lexicon')

# Create a sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Define a function to determine sentiment
def analyze_sentiment(text):
    sentiment_scores = analyzer.polarity_scores(text)
    
    if sentiment_scores['compound'] >= 0.05:
        return 'Positive'
    elif sentiment_scores['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Test the sentiment analysis function
text = input("give a sentiment statement")
sentiment = analyze_sentiment(text)
print(f"Sentiment: {sentiment}")
