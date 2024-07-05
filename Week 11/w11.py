import asyncio
import aiohttp
import feedparser
import csv
import sqlite3
from datetime import datetime
from urllib.parse import urlparse
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize BERT model and tokenizer
model_name = "dbmdz/bert-base-italian-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# RSS feed URLs
RSS_SOURCES = [
    "https://www.repubblica.it/rss/homepage/rss2.0.xml",
    "https://www.corriere.it/rss/homepage.xml",
    # Add more RSS feeds as needed
]

# Keywords to search for
KEYWORDS = ["economia", "politica", "tecnologia", "salute", "nucleare", "covid"]

# Set to store processed links
processed_links = set()

def analyze_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)
    sentiment_score = probabilities[0, 1].item() - probabilities[0, 0].item()  # Positive - Negative
    
    return sentiment_score

def search_keywords(text):
    text_lower = text.lower()
    return [keyword for keyword in KEYWORDS if keyword in text_lower]

async def fetch_news_async(rss_url):
    async with aiohttp.ClientSession() as session:
        async with session.get(rss_url) as response:
            content = await response.text()
            return feedparser.parse(content)

def get_website_name(url):
    parsed_url = urlparse(url)
    return parsed_url.netloc

async def process_feeds():
    results = []
    for source in RSS_SOURCES:
        feed = await fetch_news_async(source)
        if feed:
            for item in feed.entries:
                link = item.get('link', '')
                if link not in processed_links:
                    full_text = item.get('title', '') + ' ' + item.get('summary', '')
                    sentiment = analyze_sentiment(full_text)
                    matched_keywords = search_keywords(full_text)
                    results.append({
                        'title': item.get('title', ''),
                        'full_text': full_text,
                        'link': link,
                        'source': get_website_name(source),
                        'sentiment': sentiment,
                        'pub_date': item.get('published', ''),
                        'author': item.get('author', ''),
                        'keywords': ', '.join(matched_keywords)
                    })
                    processed_links.add(link)
    return results

def save_to_csv(data, filename='news_sentiment.csv'):
    with open(filename, 'a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=data[0].keys())
        if file.tell() == 0:
            writer.writeheader()
        writer.writerows(data)

def save_to_database(data, db_name='news_sentiment.db'):
    conn = sqlite3.connect(db_name)
    df = pd.DataFrame(data)
    df.to_sql('news', conn, if_exists='append', index=False)
    conn.close()

async def main():
    results = await process_feeds()
    if results:
        save_to_csv(results)
        save_to_database(results)
        logging.info(f"Processed {len(results)} new news items.")
    else:
        logging.info("No new news items to process.")

def scheduled_job():
    asyncio.run(main())

if __name__ == "__main__":
    # Run the job immediately
    scheduled_job()
    
    # Set up the scheduler
    scheduler = AsyncIOScheduler()
    scheduler.add_job(scheduled_job, 'interval', hours=1)
    scheduler.start()
    logging.info("Scheduler started. Press Ctrl+C to exit.")

    try:
        asyncio.get_event_loop().run_forever()
    except (KeyboardInterrupt, SystemExit):
        pass