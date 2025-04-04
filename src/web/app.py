import sys
import os
import pandas as pd
import numpy as np
import json
import threading
import subprocess
import requests
from datetime import datetime
import logging
import glob
import pickle
import re
import random
import feedparser
import time
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import sqlite3
from flask import Flask, render_template, request, redirect, url_for, jsonify

# Add proper paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(src_dir)

# Ensure all needed directories exist
os.makedirs(os.path.join(project_root, 'data', 'display'), exist_ok=True)
os.makedirs(os.path.join(project_root, 'data', 'scraped'), exist_ok=True)
os.makedirs(os.path.join(project_root, 'data', 'processed_scraped'), exist_ok=True)
os.makedirs(os.path.join(project_root, 'models'), exist_ok=True)
os.makedirs(os.path.join(project_root, 'logs'), exist_ok=True)
os.makedirs(os.path.join(current_dir, 'static', 'css'), exist_ok=True)
os.makedirs(os.path.join(current_dir, 'static', 'js'), exist_ok=True)

# Add paths to sys.path
sys.path.append(src_dir)
sys.path.append(os.path.join(src_dir, 'scripts'))
sys.path.append(project_root)

# Logger configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)

# Global variables
articles_df = None
category_model = None
sentiment_model = None
loaded_date = None
enhanced_models = False

# Save sentiment gauge CSS
sentiment_gauge_css = """
.sentiment-gauge-container {
    margin: 30px 0;
    background-color: #f8f9fa;
    border-radius: 15px;
    padding: 20px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}
.sentiment-gauge-title {
    text-align: center;
    font-size: 1.5rem;
    margin-bottom: 20px;
    font-weight: 600;
    color: #333;
}
.gauge-container {
    position: relative;
    height: 160px;
    margin-bottom: 20px;
}
.gauge-bg {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100px;
}
.gauge-half-circle {
    position: absolute;
    top: 0;
    left: 50%;
    transform: translateX(-50%);
    width: 220px;
    height: 110px;
    border-top-left-radius: 110px;
    border-top-right-radius: 110px;
    background: linear-gradient(90deg, 
        #d32f2f 0%, 
        #f44336 10%, 
        #ff9800 30%, 
        #ffeb3b 50%, 
        #8bc34a 70%, 
        #4caf50 90%, 
        #2e7d32 100%
    );
}
.gauge-mask {
    position: absolute;
    top: 10px;
    left: 50%;
    transform: translateX(-50%);
    width: 200px;
    height: 100px;
    border-top-left-radius: 100px;
    border-top-right-radius: 100px;
    background: #f8f9fa;
    z-index: 1;
}
.gauge-value {
    position: absolute;
    top: 95px;
    left: 50%;
    transform: translateX(-50%);
    text-align: center;
    z-index: 3;
}
.sentiment-score {
    font-size: 2.5rem;
    font-weight: bold;
    color: #333;
    transition: color 0.5s ease;
}
.sentiment-score.text-danger { color: #d32f2f; }
.sentiment-score.text-warning { color: #ff9800; }
.sentiment-score.text-success { color: #2e7d32; }
.gauge-label-text {
    font-size: 0.85rem;
    color: #666;
    margin-top: 5px;
}
.gauge-needle {
    position: absolute;
    top: 0;
    left: 50%;
    transform-origin: bottom center;
    transform: translateX(-50%) rotate(-90deg);
    height: 110px;
    width: 4px;
    background-color: #333;
    z-index: 2;
    border-radius: 2px;
}
.gauge-needle::after {
    content: '';
    position: absolute;
    bottom: -5px;
    left: -5px;
    width: 14px;
    height: 14px;
    background-color: #333;
    border-radius: 50%;
}
.gauge-labels {
    position: relative;
    top: 110px;
    width: 100%;
    display: flex;
    justify-content: space-between;
    padding: 0 10px;
    margin-top: 10px;
}
.gauge-label {
    font-size: 0.8rem;
    color: #777;
    text-align: center;
    max-width: 80px;
}
.word-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}
.word-tag {
    padding: 6px 12px;
    border-radius: 16px;
    font-size: 0.85rem;
    font-weight: 500;
}
.word-tag.positive {
    background-color: rgba(76, 175, 80, 0.15);
    color: #2e7d32;
}
.word-tag.negative {
    background-color: rgba(244, 67, 54, 0.15);
    color: #c62828;
}
"""

# Save sentiment gauge JS
sentiment_gauge_js = """
document.addEventListener('DOMContentLoaded', function() {
    // Initialize all sentiment gauges on the page
    initSentimentGauges();
});

function initSentimentGauges() {
    const gauges = document.querySelectorAll('.sentiment-gauge-container');
    
    gauges.forEach(function(gauge) {
        const gaugeNeedle = gauge.querySelector('.gauge-needle');
        const scoreElement = gauge.querySelector('.sentiment-score');
        
        if (gaugeNeedle && scoreElement) {
            const score = parseFloat(scoreElement.dataset.score || 5);
            animateNeedle(gaugeNeedle, score);
            setScoreColor(scoreElement, score);
        }
    });
}

function animateNeedle(needle, score) {
    needle.style.transform = 'translateX(-50%) rotate(-90deg)';
    const finalAngle = Math.min(Math.max(score, 0), 10) * 18 - 90;
    needle.style.transition = 'transform 1.5s cubic-bezier(0.34, 1.56, 0.64, 1)';
    
    setTimeout(function() {
        needle.style.transform = `translateX(-50%) rotate(${finalAngle}deg)`;
    }, 100);
}

function setScoreColor(element, score) {
    element.classList.remove('text-danger', 'text-warning', 'text-success');
    
    if (score < 4) {
        element.classList.add('text-danger');
    } else if (score < 7) {
        element.classList.add('text-warning');
    } else {
        element.classList.add('text-success');
    }
}
"""

# Create static CSS and JS files
with open(os.path.join(current_dir, 'static', 'css', 'sentiment-gauge.css'), 'w') as f:
    f.write(sentiment_gauge_css)

with open(os.path.join(current_dir, 'static', 'js', 'sentiment-gauge.js'), 'w') as f:
    f.write(sentiment_gauge_js)

# Simple wrapper class for the category classifier
class CategoryClassifier:
    def __init__(self, categories=None):
        self.categories = categories or ['Zprávy', 'Politika', 'Ekonomika', 'Sport', 'Kultura', 'Technika']
    
    def predict(self, texts):
        if not isinstance(texts, list):
            texts = [texts]
        # Simple keyword-based classification
        results = []
        for text in texts:
            text = text.lower() if isinstance(text, str) else ""
            if any(word in text for word in ['fotbal', 'hokej', 'zápas', 'utkání', 'liga', 'turnaj']):
                results.append('Sport')
            elif any(word in text for word in ['vláda', 'ministr', 'prezident', 'parlament', 'zákon']):
                results.append('Politika')
            elif any(word in text for word in ['ekonomika', 'finance', 'koruna', 'inflace', 'daň', 'cena']):
                results.append('Ekonomika')
            elif any(word in text for word in ['film', 'hudba', 'umění', 'divadlo', 'koncert']):
                results.append('Kultura')
            elif any(word in text for word in ['technologie', 'mobil', 'internet', 'počítač', 'aplikace']):
                results.append('Technika')
            else:
                results.append('Zprávy')
        return results

# Simple wrapper class for the sentiment analyzer
class SentimentAnalyzer:
    def __init__(self):
        self.labels = ['negative', 'neutral', 'positive']
        self.positive_words = self.load_words('positive_words.txt')
        self.negative_words = self.load_words('negative_words.txt')
    
    def load_words(self, filename):
        try:
            word_path = os.path.join(project_root, 'data', filename)
            if os.path.exists(word_path):
                with open(word_path, 'r', encoding='utf-8') as f:
                    return [line.strip() for line in f]
            return []
        except:
            # Default short lists if file not found
            if 'positive' in filename:
                return ['dobrý', 'skvělý', 'výborný', 'úspěch', 'radost', 'krásný', 'příjemný']
            else:
                return ['špatný', 'negativní', 'problém', 'tragédie', 'konflikt', 'krize', 'útok']
    
    def predict(self, texts):
        if not isinstance(texts, list):
            texts = [texts]
        results = []
        for text in texts:
            text = text.lower() if isinstance(text, str) else ""
            
            # Count positive and negative words
            pos_count = sum(1 for word in text.split() if word in self.positive_words)
            neg_count = sum(1 for word in text.split() if word in self.negative_words)
            
            # Calculate sentiment
            if neg_count > pos_count * 1.2:
                results.append(0)  # negative
            elif pos_count > neg_count * 1.2:
                results.append(2)  # positive
            else:
                results.append(1)  # neutral
        
        return results
    
    def extract_sentiment_features(self, texts):
        if not isinstance(texts, list):
            texts = [texts]
        
        features = pd.DataFrame()
        
        # Count positive and negative words
        features['positive_word_count'] = [
            sum(1 for word in text.lower().split() if word in self.positive_words) 
            for text in texts
        ]
        
        features['negative_word_count'] = [
            sum(1 for word in text.lower().split() if word in self.negative_words)
            for text in texts
        ]
        
        # Calculate sentiment ratio
        features['sentiment_ratio'] = (features['positive_word_count'] + 1) / (features['negative_word_count'] + 1)
        
        return features

# Simple Article Chatbot
class SimpleArticleChatbot:
    def __init__(self):
        self.responses = {
            'greeting': [
                "Dobrý den! Jak vám mohu pomoci s tímto článkem?",
                "Zdravím! O co byste se chtěli dozvědět více?",
                "Vítejte! Mohu vám pomoci s analýzou tohoto článku."
            ],
            'sentiment': [
                "Sentiment tohoto článku je {sentiment}. {reason}",
                "Článek má {sentiment} tón. {reason}",
                "Text vykazuje {sentiment} sentiment, protože {reason}"
            ],
            'category': [
                "Tento článek patří do kategorie {category}.",
                "Článek je zařazen do kategorie {category}.",
                "Kategorie tohoto článku je {category}."
            ],
            'unknown': [
                "Omlouvám se, ale nerozumím vaší otázce. Můžete se zeptat na sentiment článku nebo jeho kategorii?",
                "Nejsem si jistý, co přesně chcete vědět. Zkuste se mě zeptat na sentiment nebo téma článku.",
                "Zkuste, prosím, přeformulovat otázku. Mohu vám poskytnout informace o sentimentu nebo kategorii článku."
            ]
        }
    
    def respond_to_article_query(self, message, article):
        message_lower = message.lower()
        
        # Check for greetings
        if any(word in message_lower for word in ['ahoj', 'dobrý den', 'zdravím', 'čau']):
            return random.choice(self.responses['greeting'])
        
        # Check for sentiment questions
        if any(word in message_lower for word in ['sentiment', 'nálada', 'tón', 'pozitivní', 'negativní', 'neutrální']):
            if 'sentiment' in article and article['sentiment']:
                sentiment = article['sentiment']
                reason = article.get('sentiment_reason', "Důvod není znám.")
                response = random.choice(self.responses['sentiment'])
                return response.format(sentiment=sentiment, reason=reason)
            else:
                return "Omlouvám se, ale sentiment tohoto článku nebyl analyzován."
        
        # Check for category questions
        if any(word in message_lower for word in ['kategorie', 'téma', 'oblast', 'rubrika', 'zaměření']):
            if 'Category' in article and article['Category']:
                category = article['Category']
                response = random.choice(self.responses['category'])
                return response.format(category=category)
            elif 'predicted_category' in article and article['predicted_category']:
                category = article['predicted_category']
                response = random.choice(self.responses['category'])
                return response.format(category=category)
            else:
                return "Omlouvám se, ale kategorie tohoto článku není známa."
        
        # Default response for unknown questions
        return random.choice(self.responses['unknown'])

# Initialize chatbot
article_chatbot = SimpleArticleChatbot()

def scrape_newest_articles():
    """Scrape the newest articles from specified sources"""
    sources_to_scrape = {
        "novinky": "https://www.novinky.cz/rss",
        "seznamzpravy": "https://www.seznamzpravy.cz/rss",
        "zpravy.aktualne": "https://zpravy.aktualne.cz/rss/"
    }
    
    all_articles = []
    
    for source_name, rss_url in sources_to_scrape.items():
        try:
            logger.info(f"Scraping {source_name}...")
            feed = feedparser.parse(rss_url)
            
            # Take the 5 newest entries
            for i, entry in enumerate(feed.entries[:5]):
                if i >= 5:  # Just to be safe
                    break
                    
                title = entry.title if hasattr(entry, 'title') else "Untitled"
                url = entry.link if hasattr(entry, 'link') else ""
                
                if not url:
                    continue
                
                # Get publication date
                pub_date = None
                if hasattr(entry, 'published'):
                    pub_date = entry.published
                elif hasattr(entry, 'pubDate'):
                    pub_date = entry.pubDate
                elif hasattr(entry, 'updated'):
                    pub_date = entry.updated
                
                # Try to get category
                category = ""
                if hasattr(entry, 'tags') and entry.tags:
                    try:
                        category = entry.tags[0].term
                    except:
                        try:
                            category = entry.tags[0]['term']
                        except:
                            category = ""
                elif hasattr(entry, 'category'):
                    category = entry.category
                
                # Get article text
                article_text, char_count, word_count = extract_article_text(url, source_name)
                
                # Add to articles list
                article = {
                    "Title": title,
                    "Content": article_text,
                    "Source": source_name,
                    "Category": category,
                    "PublishDate": pub_date,
                    "ArticleUrl": url,
                    "ArticleLength": char_count,
                    "WordCount": word_count,
                    "ScrapedDate": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                all_articles.append(article)
                logger.info(f"Scraped: {title}")
                
                # Small delay to avoid overloading the server
                time.sleep(0.5)
            
        except Exception as e:
            logger.error(f"Error scraping {source_name}: {e}")
    
    # Save scraped articles to JSON file
    if all_articles:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_path = os.path.join(project_root, 'data', 'scraped', f'articles_{timestamp}.json')
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(all_articles, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved {len(all_articles)} articles to {file_path}")
        
        # Also update all_articles.json in display directory
        display_file = os.path.join(project_root, 'data', 'display', 'all_articles.json')
        
        try:
            # If file exists, load existing articles and merge
            if os.path.exists(display_file):
                with open(display_file, 'r', encoding='utf-8') as f:
                    existing_articles = json.load(f)
                
                # Add IDs to new articles starting from max existing ID + 1
                max_id = 0
                for article in existing_articles:
                    if 'Id' in article and article['Id'] > max_id:
                        max_id = article['Id']
                
                for i, article in enumerate(all_articles):
                    article['Id'] = max_id + i + 1
                
                # Combine existing and new articles
                combined_articles = existing_articles + all_articles
                
                # Remove duplicates based on URL
                unique_articles = []
                urls = set()
                
                for article in combined_articles:
                    if article['ArticleUrl'] not in urls:
                        urls.add(article['ArticleUrl'])
                        unique_articles.append(article)
                
                with open(display_file, 'w', encoding='utf-8') as f:
                    json.dump(unique_articles, f, ensure_ascii=False, indent=2)
                
                logger.info(f"Updated {display_file} with {len(all_articles)} new articles")
            else:
                # First time - add IDs starting from 1
                for i, article in enumerate(all_articles):
                    article['Id'] = i + 1
                
                # Save directly
                with open(display_file, 'w', encoding='utf-8') as f:
                    json.dump(all_articles, f, ensure_ascii=False, indent=2)
                
                logger.info(f"Created {display_file} with {len(all_articles)} articles")
        except Exception as e:
            logger.error(f"Error updating display file: {e}")
    
    return len(all_articles)

def extract_article_text(url, source):
    """Extract text from an article URL"""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "cs,en-US;q=0.7,en;q=0.3"
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
            element.decompose()
        
        # Try to find article content
        article_content = soup.find(['article', 'main']) or soup.find('div', class_=lambda c: c and any(x in str(c).lower() for x in ['article', 'content', 'text', 'body']))
        
        if article_content:
            # Extract paragraphs
            paragraphs = article_content.find_all('p')
            text = ' '.join([p.get_text().strip() for p in paragraphs])
        else:
            # Fallback: get all text from body
            text = soup.get_text(separator=' ', strip=True)
        
        # Clean the text
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Count chars and words
        char_count = len(text)
        word_count = len(text.split())
        
        return text, char_count, word_count
    except Exception as e:
        logger.error(f"Error extracting text from {url}: {e}")
        return f"Error extracting content: {e}", 0, 0

def run_daily_scraper():
    """Run scraper to collect latest news"""
    logger.info("Latest news scraper started - sbírám nejnovější zprávy")
    
    try:
        scrape_count = scrape_newest_articles()
        logger.info(f"Scraped {scrape_count} new articles")
    except Exception as e:
        logger.error(f"Error in scraper: {e}")
    
    # Reload data
    load_data()
    
    logger.info("Latest news scraper completed successfully")

def load_data():
    """Load articles data and models"""
    global articles_df, category_model, sentiment_model, loaded_date
    
    # Path to display data
    display_file = os.path.join(project_root, 'data', 'display', 'all_articles.json')
    
    # Load models
    try:
        category_model = CategoryClassifier()
        logger.info("Category classifier loaded successfully")
        
        sentiment_model = SentimentAnalyzer()
        logger.info("Sentiment analyzer loaded successfully")
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        
    # Load articles
    try:
        if os.path.exists(display_file):
            with open(display_file, 'r', encoding='utf-8') as f:
                articles = json.load(f)
            articles_df = pd.DataFrame(articles)
            logger.info(f"Loaded {len(articles_df)} articles from {display_file}")
        else:
            # If no display file, create empty DataFrame
            articles_df = pd.DataFrame(columns=[
                'Id', 'Title', 'Content', 'Source', 'Category', 'PublishDate', 
                'ArticleUrl', 'ArticleLength', 'WordCount'
            ])
            logger.warning("No articles data file found, created empty DataFrame")
    except Exception as e:
        logger.error(f"Failed to load articles data: {e}")
        articles_df = pd.DataFrame(columns=[
            'Id', 'Title', 'Content', 'Source', 'Category', 'PublishDate', 
            'ArticleUrl', 'ArticleLength', 'WordCount'
        ])
    
    # Ensure Id column exists
    if 'Id' not in articles_df.columns:
        articles_df['Id'] = range(1, len(articles_df) + 1)
    
    # Make sure all required columns exist
    for col in ['Title', 'Content', 'Source', 'Category', 'PublishDate', 'ArticleLength', 'WordCount']:
        if col not in articles_df.columns:
            if col == 'Source' and 'SourceName' in articles_df.columns:
                articles_df['Source'] = articles_df['SourceName']
            elif col == 'Content' and 'ArticleText' in articles_df.columns:
                articles_df['Content'] = articles_df['ArticleText']
            elif col == 'PublishDate' and 'PublicationDate' in articles_df.columns:
                articles_df['PublishDate'] = articles_df['PublicationDate']
            else:
                articles_df[col] = None
    
    loaded_date = datetime.now()
    app.config['articles_df'] = articles_df
    
    # Process articles with models if available
    if category_model is not None and sentiment_model is not None and articles_df is not None and len(articles_df) > 0:
        try:
            # Apply category model if needed
            if 'predicted_category' not in articles_df.columns:
                logger.info("Applying category model to articles...")
                texts = articles_df['Content'].fillna('').tolist()
                articles_df['predicted_category'] = category_model.predict(texts)
                logger.info("Category prediction completed")
            
            # Apply sentiment model if needed
            if 'sentiment' not in articles_df.columns:
                logger.info("Applying sentiment model to articles...")
                texts = articles_df['Content'].fillna('').tolist()
                sentiment_ids = sentiment_model.predict(texts)
                articles_df['sentiment'] = [sentiment_model.labels[sid] for sid in sentiment_ids]
                
                # Calculate sentiment scores
                logger.info("Calculating sentiment scores...")
                features = sentiment_model.extract_sentiment_features(texts)
                articles_df['sentiment_score'] = [
                    5.0 if sid == 1 else (8.0 if sid == 2 else 2.0)
                    for sid in sentiment_ids
                ]
                
                # Add sentiment features
                articles_df['positive_words'] = features['positive_word_count'].tolist()
                articles_df['negative_words'] = features['negative_word_count'].tolist()
                articles_df['sentiment_ratio'] = features['sentiment_ratio'].tolist()
                
                # Add sentiment reasons
                articles_df['sentiment_reason'] = articles_df.apply(
                    lambda row: f"Text obsahuje {row['positive_words']} pozitivních a {row['negative_words']} negativních slov."
                    if 'positive_words' in row and 'negative_words' in row else "",
                    axis=1
                )
                
                logger.info("Sentiment analysis completed")
            
            # Save processed data
            with open(display_file, 'w', encoding='utf-8') as f:
                json.dump(articles_df.to_dict('records'), f, ensure_ascii=False, indent=2)
            logger.info("Processed data saved")
        except Exception as e:
            logger.error(f"Error processing articles with models: {e}")

@app.route('/')
def index():
    """Home page showing article statistics and top categories"""
    # Load data if not loaded yet
    if articles_df is None:
        load_data()
    
    # Get article stats
    stats = {
        'total_articles': len(articles_df) if articles_df is not None else 0,
        'sources': articles_df['Source'].nunique() if articles_df is not None and len(articles_df) > 0 else 0,
        'categories': articles_df['Category'].nunique() if articles_df is not None and len(articles_df) > 0 else 0,
        'date_range': {
            'from': articles_df['PublishDate'].min() if articles_df is not None and len(articles_df) > 0 else None,
            'to': articles_df['PublishDate'].max() if articles_df is not None and len(articles_df) > 0 else None
        },
        'newest_articles': articles_df.sort_values('Id', ascending=False).head(5).to_dict('records') if articles_df is not None and len(articles_df) > 0 else [],
        'top_sources': articles_df['Source'].value_counts().head(5).to_dict() if articles_df is not None and len(articles_df) > 0 else {},
        'top_categories': articles_df['Category'].value_counts().head(5).to_dict() if articles_df is not None and len(articles_df) > 0 else {},
        'loaded_date': loaded_date,
        'enhanced_models': enhanced_models
    }
    
    return render_template('index.html', stats=stats)

@app.route('/articles')
def articles():
    """Page showing all articles with filters"""
    # Load data if not loaded yet
    if articles_df is None:
        load_data()
    
    # Get filter parameters
    category = request.args.get('category', None)
    source = request.args.get('source', None)
    sentiment = request.args.get('sentiment', None)
    search = request.args.get('search', None)
    
    # Apply filters
    if len(articles_df) > 0:
        filtered_df = articles_df.copy()
        
        if category and category != 'all':
            filtered_df = filtered_df[filtered_df['Category'] == category]
        
        if source and source != 'all':
            filtered_df = filtered_df[filtered_df['Source'] == source]
        
        if sentiment and sentiment != 'all' and 'sentiment' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['sentiment'] == sentiment]
        
        if search:
            filtered_df = filtered_df[
                filtered_df['Title'].str.contains(search, case=False, na=False) | 
                filtered_df['Content'].str.contains(search, case=False, na=False)
            ]
        
        # Pagination
        page = int(request.args.get('page', 1))
        per_page = 20
        total_pages = (len(filtered_df) + per_page - 1) // per_page
        
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        
        paged_df = filtered_df.sort_values('Id', ascending=False).iloc[start_idx:end_idx]
        
        # Prepare data for template
        articles_list = paged_df.to_dict('records')
        categories = articles_df['Category'].dropna().unique().tolist()
        sources = articles_df['Source'].dropna().unique().tolist()
        sentiments = articles_df['sentiment'].dropna().unique().tolist() if 'sentiment' in articles_df.columns else []
    else:
        # Empty data
        articles_list = []
        categories = []
        sources = []
        sentiments = []
        page = 1
        total_pages = 0
        filtered_df = pd.DataFrame()
    
    return render_template(
        'articles.html', 
        articles=articles_list,
        categories=sorted(categories),
        sources=sorted(sources),
        sentiments=sorted(sentiments) if sentiments else [],
        current_category=category,
        current_source=source,
        current_sentiment=sentiment,
        current_search=search,
        page=page,
        total_pages=total_pages,
        total_articles=len(filtered_df)
    )

@app.route('/article/<int:article_id>')
def article_detail(article_id):
    """Page showing details of a specific article"""
    # Load data if not loaded yet
    if articles_df is None:
        load_data()
    
    # Find article
    article = articles_df[articles_df['Id'] == article_id]
    
    if len(article) == 0:
        return render_template('error.html', message=f"Článek s ID {article_id} nebyl nalezen")
    
    article_data = article.iloc[0].to_dict()
    
    # Add sentiment features if missing
    if sentiment_model is not None and 'sentiment' in article_data and article_data['sentiment']:
        sentiment_label = article_data['sentiment']
        sentiment_id = sentiment_model.labels.index(sentiment_label) if sentiment_label in sentiment_model.labels else 1
        
        # Make sure sentiment features dictionary exists
        if 'sentiment_features' not in article_data:
            article_data['sentiment_features'] = {
                'positive_word_count': article_data.get('positive_words', 0),
                'negative_word_count': article_data.get('negative_words', 0),
                'sentiment_ratio': article_data.get('sentiment_ratio', 1.0)
            }
        
        # Add sentiment score if missing
        if 'sentiment_score' not in article_data:
            article_data['sentiment_score'] = 5.0 if sentiment_id == 1 else (8.0 if sentiment_id == 2 else 2.0)
        
        # Add reason if missing
        if 'sentiment_reason' not in article_data or not article_data['sentiment_reason']:
            positive_count = article_data.get('positive_words', 0)
            negative_count = article_data.get('negative_words', 0)
            
            if sentiment_id == 2:  # positive
                article_data['sentiment_reason'] = f"Text obsahuje více pozitivních slov ({positive_count}) než negativních slov ({negative_count})."
            elif sentiment_id == 0:  # negative
                article_data['sentiment_reason'] = f"Text obsahuje více negativních slov ({negative_count}) než pozitivních slov ({positive_count})."
            else:
                article_data['sentiment_reason'] = f"Text obsahuje vyváženou kombinaci pozitivních ({positive_count}) a negativních ({negative_count}) slov."
    
    return render_template('article_detail.html', article=article_data)

@app.route('/categories')
def categories():
    """Page showing article distribution across categories"""
    # Load data if not loaded yet
    if articles_df is None:
        load_data()
    
    # Get category counts
    if len(articles_df) > 0:
        category_counts = articles_df['Category'].value_counts().reset_index()
        category_counts.columns = ['category', 'count']
    else:
        category_counts = pd.DataFrame(columns=['category', 'count'])
    
    # Prepare data for template
    categories_list = category_counts.to_dict('records')
    
    return render_template('categories.html', categories=categories_list)

@app.route('/sources')
def sources():
    """Page showing article distribution across sources"""
    # Load data if not loaded yet
    if articles_df is None:
        load_data()
    
    # Get source counts
    if len(articles_df) > 0:
        source_counts = articles_df['Source'].value_counts().reset_index()
        source_counts.columns = ['source', 'count']
    else:
        source_counts = pd.DataFrame(columns=['source', 'count'])
    
    # Prepare data for template
    sources_list = source_counts.to_dict('records')
    
    return render_template('sources.html', sources=sources_list)

@app.route('/api/analyze', methods=['POST'])
def analyze_text():
    """API endpoint to analyze a text"""
    data = request.json
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    result = {
        'text': text,
        'length': len(text),
        'word_count': len(text.split())
    }
    
    # Get category prediction if model is available
    if category_model is not None:
        try:
            predicted_category = category_model.predict([text])[0]
            result['category'] = predicted_category
        except Exception as e:
            logger.error(f"Error predicting category: {e}")
            result['category'] = "Neznámá"
    
    # Get sentiment prediction if model is available
    if sentiment_model is not None:
        try:
            sentiment_id = sentiment_model.predict([text])[0]
            result['sentiment'] = sentiment_model.labels[sentiment_id]
            
            # Add sentiment features
            features = sentiment_model.extract_sentiment_features([text])
            result['sentiment_score'] = 5.0 if sentiment_id == 1 else (8.0 if sentiment_id == 2 else 2.0)
            result['sentiment_features'] = {
                'positive_word_count': features['positive_word_count'].iloc[0],
                'negative_word_count': features['negative_word_count'].iloc[0],
                'sentiment_ratio': features['sentiment_ratio'].iloc[0]
            }
        except Exception as e:
            logger.error(f"Error predicting sentiment: {e}")
            result['sentiment'] = "neutrální"
            result['sentiment_score'] = 5.0
    
    return jsonify(result)

@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    """Page with form to analyze custom text"""
    if request.method == 'POST':
        text = request.form.get('text', '')
        
        result = {
            'text': text,
            'length': len(text),
            'word_count': len(text.split())
        }
        
        # Get category prediction if model is available
        if category_model is not None:
            try:
                predicted_category = category_model.predict([text])[0]
                result['category'] = predicted_category
            except Exception as e:
                logger.error(f"Error predicting category: {e}")
                result['category'] = "Neznámá"
        
        # Get sentiment prediction if model is available
        if sentiment_model is not None:
            try:
                sentiment_id = sentiment_model.predict([text])[0]
                result['sentiment'] = sentiment_model.labels[sentiment_id]
                
                # Add sentiment features
                features = sentiment_model.extract_sentiment_features([text])
                result['sentiment_score'] = 5.0 if sentiment_id == 1 else (8.0 if sentiment_id == 2 else 2.0)
                result['sentiment_features'] = {
                    'positive_word_count': features['positive_word_count'].iloc[0],
                    'negative_word_count': features['negative_word_count'].iloc[0],
                    'sentiment_ratio': features['sentiment_ratio'].iloc[0]
                }
            except Exception as e:
                logger.error(f"Error predicting sentiment: {e}")
                result['sentiment'] = "neutrální"
                result['sentiment_score'] = 5.0
                result['sentiment_features'] = {
                    'positive_word_count': 0,
                    'negative_word_count': 0,
                    'sentiment_ratio': 1.0
                }
        
        return render_template('analyze.html', result=result)
    
    return render_template('analyze.html')

@app.route('/api/article_chatbot', methods=['POST'])
def article_chatbot_api():
    """API endpoint for chatbot interactions about articles"""
    try:
        data = request.json
        message = data.get('message', '')
        article_id = data.get('article_id')
        
        if not message:
            return jsonify({'response': 'Prosím, napište nějakou zprávu.'})
        
        if not article_id:
            return jsonify({'response': 'Chybí ID článku.'})
        
        # Get the article
        if articles_df is None:
            load_data()
            
        article = articles_df[articles_df['Id'] == int(article_id)]
        
        if len(article) == 0:
            return jsonify({'response': f'Článek s ID {article_id} nebyl nalezen.'})
        
        article_data = article.iloc[0].to_dict()
        
        # Generate response
        response = article_chatbot.respond_to_article_query(message, article_data)
        
        return jsonify({'response': response})
    except Exception as e:
        logger.error(f"Error in article chatbot API: {e}")
        return jsonify({'response': 'Omlouvám se, ale něco se pokazilo. Zkuste to prosím znovu.'})

@app.route('/reload_data')
def reload_data():
    """Endpoint to reload data and models"""
    # Run latest news scraper in background thread
    thread = threading.Thread(target=run_daily_scraper)
    thread.daemon = True
    thread.start()
    
    return redirect(url_for('index'))

@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors"""
    return render_template('error.html', message="Stránka nebyla nalezena"), 404

@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors"""
    return render_template('error.html', message="Chyba serveru"), 500

if __name__ == '__main__':
    # Load data and models
    load_data()
    
    # Run initial scraper to collect latest news
    logger.info("Running initial scraper to collect latest news (5 per source)...")
    scrape_newest_articles()
    
    # Process scraped data
    logger.info("Processing scraped data...")
    load_data()  # Reload with the new data
    
    # Run daily scraper in background thread
    try:
        daily_thread = threading.Thread(target=run_daily_scraper)
        daily_thread.daemon = True
        daily_thread.start()
        logger.info("Daily scraper thread started")
    except Exception as e:
        logger.error(f"Failed to start scraper thread: {e}")
    
    # Run app
    app.run(debug=True, host='0.0.0.0', port=5000)