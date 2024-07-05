import asyncio
import aiohttp
import feedparser
import sqlite3
import logging
from nltk.stem import SnowballStemmer
from apscheduler.schedulers.background import BackgroundScheduler
import pandas as pd
from datetime import datetime
import warnings
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from urllib.parse import urlparse
import json
import threading

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize stemmer for Italian
stemmer = SnowballStemmer("italian")

# Load configuration
def load_config():
    with open('config.json', 'r') as f:
        return json.load(f)

CONFIG = load_config()

# Load RSS sources
def load_rss_sources():
    with open(CONFIG['rss_sources_file'], 'r') as f:
        return json.load(f)

RSS_SOURCES = load_rss_sources()

KEYWORDS = CONFIG['keywords']
BATCH_SIZE = CONFIG['batch_size']

async def fetch_news_async(rss_url):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(rss_url) as response:
                content = await response.text()
                return feedparser.parse(content)
    except Exception as e:
        logging.error(f"Error fetching news from {rss_url}: {str(e)}")
        return None

def search_keywords(news_item, keywords):
    content = news_item.get('title', '') + ' ' + news_item.get('summary', '')
    content_lower = content.lower()
    matched_keywords = [keyword for keyword in keywords if keyword.lower() in content_lower]
    return matched_keywords

def analyze_sentiment(text):
    try:
        from textblob import TextBlob
        blob = TextBlob(text)
        return blob.sentiment.polarity
    except Exception as e:
        logging.error(f"Error in sentiment analysis: {str(e)}")
        return 0.0  # Neutral sentiment in case of error

def create_database():
    conn = sqlite3.connect(CONFIG['database_file'])
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS news
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  title TEXT,
                  summary TEXT,
                  link TEXT,
                  source TEXT,
                  sentiment REAL,
                  pub_date TEXT,
                  author TEXT,
                  keyword TEXT)''')
    conn.commit()
    conn.close()

def get_website_name(url):
    parsed_url = urlparse(url)
    return parsed_url.netloc

def store_results(news_item, sentiment, source_url, keywords):
    conn = sqlite3.connect(CONFIG['database_file'])
    c = conn.cursor()
    
    pub_date = news_item.get('published', '')
    try:
        parsed_date = datetime.strptime(pub_date, "%a, %d %b %Y %H:%M:%S %z")
        formatted_date = parsed_date.strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        logging.warning(f"Could not parse date: {pub_date}. Using current time.")
        formatted_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    source = get_website_name(source_url)

    for keyword in keywords:
        c.execute("INSERT INTO news (title, summary, link, source, sentiment, pub_date, author, keyword) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                  (news_item.get('title', ''),
                   news_item.get('summary', ''),
                   news_item.get('link', ''),
                   source,
                   sentiment,
                   formatted_date,
                   news_item.get('author', ''),
                   keyword))
    conn.commit()
    conn.close()

async def process_source(source):
    feed = await fetch_news_async(source)
    if feed:
        for item in feed.entries:
            matched_keywords = search_keywords(item, KEYWORDS)
            if matched_keywords:
                text_for_sentiment = item.get('summary', item.get('title', ''))
                sentiment = analyze_sentiment(text_for_sentiment)
                store_results(item, sentiment, source, matched_keywords)
                logging.info(f"Processed: {item.get('title', 'No title')} | Source: {get_website_name(source)} | Sentiment: {sentiment} | Keywords: {matched_keywords}")

async def process_batch(batch):
    tasks = [process_source(source) for source in batch]
    await asyncio.gather(*tasks)

async def main():
    create_database()
    for i in range(0, len(RSS_SOURCES), BATCH_SIZE):
        batch = RSS_SOURCES[i:i+BATCH_SIZE]
        await process_batch(batch)

def scheduled_job():
    asyncio.run(main())

# Dashboard
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("News Sentiment Dashboard"),
    dcc.DatePickerRange(
        id='date-picker-range',
        start_date_placeholder_text="Start Date",
        end_date_placeholder_text="End Date",
        calendar_orientation='horizontal',
    ),
    dcc.Dropdown(
        id='keyword-dropdown',
        options=[{'label': keyword, 'value': keyword} for keyword in KEYWORDS],
        value=KEYWORDS[0],
        multi=False
    ),
    dcc.Graph(id='sentiment-trend'),
    dcc.Interval(
        id='interval-component',
        interval=CONFIG['update_interval']*1000,  # in milliseconds
        n_intervals=0
    )
])

@app.callback(
    Output('sentiment-trend', 'figure'),
    [Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date'),
     Input('keyword-dropdown', 'value'),
     Input('interval-component', 'n_intervals')]
)
def update_graph(start_date, end_date, selected_keyword, n):
    conn = sqlite3.connect(CONFIG['database_file'])
    df = pd.read_sql_query("SELECT * FROM news WHERE keyword = ?", conn, params=(selected_keyword,))
    conn.close()
    
    logging.info(f"Selected keyword: {selected_keyword}")
    logging.info(f"Date range: {start_date} to {end_date}")
    logging.info(f"Number of rows in df: {len(df)}")
    
    df['date'] = pd.to_datetime(df['pub_date'], errors='coerce')
    df = df.dropna(subset=['date'])

    if start_date is not None:
        df = df[df['date'] >= start_date]
    if end_date is not None:
        df = df[df['date'] <= end_date]

    daily_sentiment = df.groupby(df['date'].dt.date)['sentiment'].mean().reset_index()
    
    logging.info(f"Number of rows in daily_sentiment: {len(daily_sentiment)}")
    logging.info(f"Data for graph: {daily_sentiment.to_dict()}")

    fig = go.Figure(data=[go.Scatter(x=daily_sentiment['date'], y=daily_sentiment['sentiment'], mode='lines+markers')])
    
    fig.update_layout(
        title=f'Daily Average Sentiment for Keyword: {selected_keyword}',
        xaxis_title='Date',
        yaxis_title='Sentiment',
        template='plotly_white'
    )

    return fig

def run_scheduler():
    scheduler = BackgroundScheduler()
    scheduler.add_job(scheduled_job, 'interval', seconds=CONFIG['update_interval'])
    scheduler.start()
    logging.info("Scheduler started.")

if __name__ == "__main__":
    # Check database content
    conn = sqlite3.connect(CONFIG['database_file'])
    df = pd.read_sql_query("SELECT * FROM news", conn)
    logging.info(f"Database content:\n{df.head()}")
    logging.info(f"Available keywords: {df['keyword'].unique()}")
    conn.close()

    # Run the job immediately
    scheduled_job()
    
    # Start the scheduler in a separate thread
    scheduler_thread = threading.Thread(target=run_scheduler)
    scheduler_thread.start()
    
    # Run the Dash app
    app.run_server(debug=False, port=CONFIG['dashboard_port'])