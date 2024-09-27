import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

# Download stopwords
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))

# Function to scrape news articles from a free news website
def scrape_news_articles(url='https://www.bbc.com/news'):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Example: Find all article links or summary content
    articles = []
    for item in soup.find_all('a', href=True):
        # Extracting article summaries or titles (depends on site structure)
        if item.text and item['href'].startswith('/news'):
            article_url = 'https://www.bbc.com' + item['href']
            article_response = requests.get(article_url)
            article_soup = BeautifulSoup(article_response.content, 'html.parser')
            paragraphs = article_soup.find_all('p')
            article_text = ' '.join([p.get_text() for p in paragraphs])
            articles.append(article_text)

    return pd.DataFrame({'text': articles, 'source': 'Scraped'})

# Function to load local CSV file
def load_local_data(file_path):
    return pd.read_csv(file_path)

# Function to clean the text
def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    text = text.lower()  # Lowercase text
    tokens = nltk.word_tokenize(text)  # Tokenize
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    lemmatizer = nltk.WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatize
    return ' '.join(tokens)

# Example of how we combine scraped and local data
def combine_data(url='https://www.bbc.com/news', csv_file=None):
    # Scrape news articles
    news_df = scrape_news_articles(url)

    if csv_file:
        # Load local CSV
        local_df = load_local_data(csv_file)
        local_df['source'] = 'Local'
        # Combine both
        combined_df = pd.concat([news_df, local_df], ignore_index=True)
    else:
        combined_df = news_df

    # Clean the text data
    combined_df['clean_text'] = combined_df['text'].apply(lambda x: clean_text(x) if pd.notnull(x) else '')

    return combined_df

# Function to split the dataset into train and test
def prepare_data(url='https://www.bbc.com/news', csv_file=None):
    df = combine_data(url, csv_file)
    df = df[df['clean_text'] != '']  # Filter out empty articles

    # For now, we generate dummy labels (this will change based on actual data)
    df['label'] = pd.cut(df.index, 3, labels=["negative", "neutral", "positive"])  # Placeholder for actual labels

    # Split into train and test
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    return train_df, test_df

# Example usage
train_data, test_data = prepare_data(url='https://www.bbc.com/news', csv_file=None)
print(train_data.head())
