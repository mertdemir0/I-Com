import feedparser
import requests
from feel_it import SentimentClassifier
import sqlite3
import re

# Initialize sentiment classifier
sentiment_classifier = SentimentClassifier()

# Example RSS sources (Italian news sites)
RSS_SOURCES = [
    "https://www.repubblica.it/rss/homepage/rss2.0.xml",
    "https://www.corriere.it/rss/homepage.xml",
    "https://www.ansa.it/sito/notizie/topnews/topnews_rss.xml",
    "https://www.ilsole24ore.com/rss/italia.xml"
]

# Example keywords to search for
KEYWORDS = ["economia", "politica", "tecnologia", "salute"]

def add_rss_source(url):
    RSS_SOURCES.append(url)
    print(f"Added RSS source: {url}")

def fetch_news(rss_url):
    print(f"Fetching news from {rss_url}")
    feed = feedparser.parse(rss_url)
    return feed.entries

def search_keywords(news_item, keywords):
    title = news_item.get('title', '').lower()
    summary = news_item.get('summary', '').lower()
    content = title + ' ' + summary
    return any(keyword.lower() in content for keyword in keywords)

def analyze_sentiment(text):
    # Perform sentiment analysis
    sentiment = sentiment_classifier.predict([text])[0]
    # FEEL-IT returns 'positive' or 'negative', you might want to convert this to a numeric score
    return 1 if sentiment == 'positive' else -1

def create_database():
    conn = sqlite3.connect('news_sentiment.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS news
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  title TEXT,
                  summary TEXT,
                  link TEXT,
                  source TEXT,
                  sentiment REAL)''')
    conn.commit()
    conn.close()

def store_results(news_item, sentiment):
    conn = sqlite3.connect('news_sentiment.db')
    c = conn.cursor()
    c.execute("INSERT INTO news (title, summary, link, source, sentiment) VALUES (?, ?, ?, ?, ?)",
              (news_item.get('title', ''),
               news_item.get('summary', ''),
               news_item.get('link', ''),
               news_item.get('source', {}).get('title', ''),
               sentiment))
    conn.commit()
    conn.close()

def main():
    create_database()

    for source in RSS_SOURCES:
        news_items = fetch_news(source)
        for item in news_items:
            if search_keywords(item, KEYWORDS):
                # Use the summary for sentiment analysis if available, otherwise use the title
                text_for_sentiment = item.get('summary', item.get('title', ''))
                sentiment = analyze_sentiment(text_for_sentiment)
                store_results(item, sentiment)
                print(f"Processed: {item.get('title', 'No title')} | Sentiment: {sentiment}")

if __name__ == "__main__":
    main()