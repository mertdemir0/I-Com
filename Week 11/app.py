import asyncio
import aiohttp
import feedparser
import sqlite3
import logging
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from apscheduler.schedulers.background import BackgroundScheduler
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

BATCH_SIZE = CONFIG['batch_size']

# Define specific topics and their keywords
SPECIFIC_TOPICS = {
    'Nucleare': ['nucleare', 'atomica', 'fissione', 'fusione', 'reattore'],
    'COVID-19': ['covid', 'coronavirus', 'pandemia', 'vaccino', 'lockdown'],
    'Clima': ['clima', 'riscaldamento globale', 'emissioni', 'sostenibilità'],
    # Add more specific topics as needed
}

# Initialize NMF model, vectorizer, and topic mapping
nmf_model = None
vectorizer = None
topic_mapping = {}

# Initialize BERT model and tokenizer
model_name = "dbmdz/bert-base-italian-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def analyze_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)
    sentiment_score = probabilities[0, 1].item() - probabilities[0, 0].item()  # Positive - Negative
    
    return sentiment_score

async def fetch_news_async(rss_url):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(rss_url) as response:
                content = await response.text()
                return feedparser.parse(content)
    except Exception as e:
        logging.error(f"Error fetching news from {rss_url}: {str(e)}")
        return None

def train_nmf_model(texts, n_topics=5):
    global nmf_model, vectorizer, topic_mapping
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words=stopwords.words('italian'))
    tfidf = vectorizer.fit_transform(texts)
    nmf_model = NMF(n_components=n_topics, random_state=1).fit(tfidf)
    topic_mapping = {i: i for i in range(n_topics)}

def get_topic_words(topic_idx, n_top_words=5):
    if nmf_model is None or vectorizer is None:
        return "Uncategorized"
    feature_names = vectorizer.get_feature_names_out()
    if topic_idx in topic_mapping:
        model_topic_idx = topic_mapping[topic_idx]
        topic_words = [feature_names[i] for i in nmf_model.components_[model_topic_idx].argsort()[:-n_top_words - 1:-1]]
        return ', '.join(topic_words)
    return "Uncategorized"

def get_specific_topics(text):
    text_lower = text.lower()
    return [topic for topic, keywords in SPECIFIC_TOPICS.items() if any(keyword in text_lower for keyword in keywords)]

def get_topic(text):
    global nmf_model, vectorizer, topic_mapping
    if nmf_model is None or vectorizer is None:
        return 0
    tfidf = vectorizer.transform([text])
    topic_distribution = nmf_model.transform(tfidf)
    topic = topic_distribution.argmax()
    if topic not in topic_mapping.values():
        new_topic_id = max(topic_mapping.keys()) + 1 if topic_mapping else 0
        topic_mapping[new_topic_id] = topic
        return new_topic_id
    return [k for k, v in topic_mapping.items() if v == topic][0]

def create_database():
    conn = sqlite3.connect(CONFIG['database_file'])
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS news
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  title TEXT,
                  full_text TEXT,
                  link TEXT,
                  source TEXT,
                  sentiment REAL,
                  pub_date TEXT,
                  author TEXT,
                  topic INTEGER,
                  specific_topics TEXT)''')
    conn.commit()
    conn.close()

def get_website_name(url):
    parsed_url = urlparse(url)
    return parsed_url.netloc

def store_results(news_item, sentiment, source_url, topic, specific_topics):
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
    full_text = news_item.get('title', '') + ' ' + news_item.get('summary', '')

    c.execute("""INSERT INTO news 
                 (title, full_text, link, source, sentiment, pub_date, author, topic, specific_topics) 
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
              (news_item.get('title', ''),
               full_text,
               news_item.get('link', ''),
               source,
               sentiment,
               formatted_date,
               news_item.get('author', ''),
               topic,
               json.dumps(specific_topics)))
    conn.commit()
    conn.close()

async def process_source(source):
    feed = await fetch_news_async(source)
    if feed:
        for item in feed.entries:
            full_text = item.get('title', '') + ' ' + item.get('summary', '')
            sentiment = analyze_sentiment(full_text)
            topic = get_topic(full_text)
            specific_topics = get_specific_topics(full_text)
            store_results(item, sentiment, source, topic, specific_topics)
            logging.info(f"Processed: {item.get('title', 'No title')} | Source: {get_website_name(source)} | Topic: {get_topic_words(topic)} | Specific Topics: {specific_topics} | Sentiment: {sentiment}")

async def process_batch(batch):
    tasks = [process_source(source) for source in batch]
    await asyncio.gather(*tasks)

async def main():
    create_database()
    
    # Fetch all texts for NMF model training
    all_texts = []
    for source in RSS_SOURCES:
        feed = await fetch_news_async(source)
        if feed:
            for item in feed.entries:
                all_texts.append(item.get('title', '') + ' ' + item.get('summary', ''))
    
    # Train NMF model
    train_nmf_model(all_texts)
    
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
        id='topic-dropdown',
        options=[],  # This will be populated dynamically
        value=None,
        multi=False
    ),
    dcc.Dropdown(
        id='specific-topic-dropdown',
        options=[{'label': topic, 'value': topic} for topic in SPECIFIC_TOPICS.keys()],
        value=None,
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
    Output('topic-dropdown', 'options'),
    Input('interval-component', 'n_intervals')
)
def update_dropdown_options(n):
    conn = sqlite3.connect(CONFIG['database_file'])
    df = pd.read_sql_query("SELECT DISTINCT topic FROM news", conn)
    conn.close()
    return [{'label': f"Topic {topic}: {get_topic_words(topic)}", 'value': topic} for topic in df['topic']]

@app.callback(
    Output('sentiment-trend', 'figure'),
    [Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date'),
     Input('topic-dropdown', 'value'),
     Input('specific-topic-dropdown', 'value'),
     Input('interval-component', 'n_intervals')]
)
def update_graph(start_date, end_date, selected_topic, selected_specific_topic, n):
    conn = sqlite3.connect(CONFIG['database_file'])
    query = "SELECT * FROM news"
    params = []
    
    if selected_topic is not None:
        query += " WHERE topic = ?"
        params.append(selected_topic)
    
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    
    if selected_specific_topic:
        df = df[df['specific_topics'].apply(lambda x: selected_specific_topic in json.loads(x))]
    
    logging.info(f"Selected topic: {get_topic_words(selected_topic) if selected_topic is not None else 'All'}")
    logging.info(f"Selected specific topic: {selected_specific_topic}")
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

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily_sentiment['date'], y=daily_sentiment['sentiment'], mode='lines+markers'))
    
    title = 'Daily Average Sentiment'
    if selected_topic is not None:
        title += f' for Topic: {get_topic_words(selected_topic)}'
    if selected_specific_topic:
        title += f' - Specific Topic: {selected_specific_topic}'
    
    fig.update_layout(
        title=title,
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
    # Run the job immediately
    scheduled_job()
    
    # Start the scheduler in a separate thread
    scheduler_thread = threading.Thread(target=run_scheduler)
    scheduler_thread.start()
    
    # Run the Dash app
    app.run_server(debug=False, port=CONFIG['dashboard_port'])