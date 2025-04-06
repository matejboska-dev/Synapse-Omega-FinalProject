#!/usr/bin/env python
import sys
import os
import pandas as pd
import numpy as np
import json
import threading
import requests
import re
from datetime import datetime
import logging
import random
import feedparser
import time
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import urllib3
from flask import Flask, render_template, request, redirect, url_for, jsonify

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

###############################################################################
# Nastavení cest a složek
###############################################################################
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(src_dir)

os.makedirs(os.path.join(project_root, 'data', 'display'), exist_ok=True)
os.makedirs(os.path.join(project_root, 'data', 'scraped'), exist_ok=True)
os.makedirs(os.path.join(project_root, 'data', 'processed_scraped'), exist_ok=True)
os.makedirs(os.path.join(project_root, 'models'), exist_ok=True)
os.makedirs(os.path.join(project_root, 'logs'), exist_ok=True)
os.makedirs(os.path.join(current_dir, 'static', 'css'), exist_ok=True)
os.makedirs(os.path.join(current_dir, 'static', 'js'), exist_ok=True)

sys.path.append(src_dir)
sys.path.append(os.path.join(src_dir, 'scripts'))
sys.path.append(project_root)

###############################################################################
# Logging
###############################################################################
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

###############################################################################
# Flask aplikace
###############################################################################
app = Flask(__name__)

###############################################################################
# Globální proměnné
###############################################################################
articles_df = None
sentiment_model = None
loaded_date = None
enhanced_models = False

# Hard-coded cesta k sentiment analyzeru
HARDCODED_SENTIMENT_MODEL_PATH = r"D:\GitHub\Synapse-Omega-FinalProject\models\sentiment_analyzer"

###############################################################################
# CSS a JS pro sentiment gauge
###############################################################################
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
"""

sentiment_gauge_js = """
document.addEventListener('DOMContentLoaded', function() {
    initSentimentGauges();
});

function initSentimentGauges() {
    const gauges = document.querySelectorAll('.sentiment-gauge-container');
    
    gauges.forEach(function(gauge) {
        const gaugeNeedle = gauge.querySelector('.gauge-needle');
        const scoreElement = gauge.querySelector('.sentiment-score');
        
        if (gaugeNeedle && scoreElement) {
            let score = 5.0;
            
            if (scoreElement.dataset.score) {
                score = parseFloat(scoreElement.dataset.score);
            } else {
                score = parseFloat(scoreElement.textContent || 5.0);
            }
            
            if (Math.abs(score - 5.0) < 0.1) {
                score = score < 5.0 ? 4.8 : 5.2;
                scoreElement.textContent = score.toFixed(1);
            }
            
            animateNeedle(gaugeNeedle, score);
            setScoreColor(scoreElement, score);
            
            console.log("Sentiment gauge initialized with score: " + score);
        } else {
            console.warn("Missing gauge needle or score element");
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

with open(os.path.join(current_dir, 'static', 'css', 'sentiment-gauge.css'), 'w') as f:
    f.write(sentiment_gauge_css)

with open(os.path.join(current_dir, 'static', 'js', 'sentiment-gauge.js'), 'w') as f:
    f.write(sentiment_gauge_js)

###############################################################################
# SimpleArticleChatbot – pouze sentiment
###############################################################################
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
            'unknown': [
                "Omlouvám se, ale nerozumím vaší otázce. Můžete se zeptat na sentiment článku?",
                "Nejsem si jistý, co přesně chcete vědět. Zkuste se mě zeptat na sentiment článku.",
                "Zkuste, prosím, přeformulovat otázku. Mohu vám poskytnout informace o sentimentu článku."
            ]
        }

    def respond_to_article_query(self, message, article):
        message_lower = message.lower()
        if any(word in message_lower for word in ['ahoj', 'dobrý den', 'zdravím', 'čau']):
            return random.choice(self.responses['greeting'])
        if any(word in message_lower for word in ['sentiment', 'nálada', 'tón', 'pozitivní', 'negativní', 'neutrální']):
            if 'sentiment' in article and article['sentiment']:
                sentiment = article['sentiment']
                reason = article.get('sentiment_reason', "Důvod není znám.")
                response = random.choice(self.responses['sentiment'])
                return response.format(sentiment=sentiment, reason=reason)
            else:
                return "Omlouvám se, ale sentiment tohoto článku nebyl analyzován."
        return random.choice(self.responses['unknown'])

article_chatbot = SimpleArticleChatbot()

###############################################################################
# EnhancedSentimentAnalyzer – načítání modelu a analýza sentimentu
###############################################################################
class EnhancedSentimentAnalyzer:
    def __init__(self, pipeline=None, labels=None, lexicons=None):
        self.pipeline = pipeline
        self.labels = labels or ['negative', 'neutral', 'positive']
        if lexicons:
            self.positive_words = lexicons.get('positive_words', [])
            self.negative_words = lexicons.get('negative_words', [])
            self.critical_negative_words = lexicons.get('critical_negative_words', [])
        else:
            self.positive_words = []
            self.negative_words = []
            self.critical_negative_words = []

    def predict(self, texts):
        if not isinstance(texts, list):
            texts = [texts]
        try:
            if self.pipeline:
                predictions = []
                for text in texts:
                    text_lower = text.lower() if isinstance(text, str) else ""
                    if any(crit_word in text_lower for crit_word in self.critical_negative_words):
                        predictions.append(0)
                    else:
                        pred = self.pipeline.predict([text])[0]
                        predictions.append(pred)
                return predictions
            else:
                return [1] * len(texts)
        except Exception as e:
            logger.error(f"Error in sentiment prediction: {e}")
            return [1] * len(texts)

    def explain_prediction(self, text):
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        text_lower = text.lower()
        words = text_lower.split()
        positive_words_found = [w for w in words if w in self.positive_words]
        negative_words_found = [w for w in words if w in self.negative_words]
        critical_words_found = [w for w in words if any(cw in w for cw in self.critical_negative_words)]
        positive_count = len(positive_words_found)
        negative_count = len(negative_words_found)
        critical_count = len(critical_words_found)
        if critical_count > 0:
            sentiment_id = 0
            sentiment = self.labels[sentiment_id]
            reason = f"Text obsahuje silně negativní slova: {', '.join(critical_words_found[:5])}"
        else:
            sentiment_id = self.predict([text])[0]
            sentiment = self.labels[sentiment_id]
            if sentiment == 'positive':
                if positive_count > 0:
                    reason = f"Text obsahuje pozitivní slova jako: {', '.join(positive_words_found[:5])}"
                else:
                    reason = "Text má celkově pozitivní tón."
            elif sentiment == 'negative':
                if negative_count > 0:
                    reason = f"Text obsahuje negativní slova jako: {', '.join(negative_words_found[:5])}"
                else:
                    reason = "Text má celkově negativní tón."
            else:
                reason = "Text obsahuje vyváženou směs pozitivních a negativních slov."
        sentiment_ratio = (positive_count + 1) / (negative_count + 1)
        sentiment_score = 5.0 + (sentiment_ratio - 1) * 2
        return {
            'text': text,
            'predicted_sentiment': sentiment,
            'sentiment_score': sentiment_score,
            'positive_words': positive_words_found[:10],
            'negative_words': negative_words_found[:10],
            'positive_word_count': positive_count,
            'negative_word_count': negative_count,
            'sentiment_ratio': sentiment_ratio,
            'reason': reason
        }

    @classmethod
    def load_model(cls, model_dir):
        try:
            import pickle
            with open(os.path.join(model_dir, 'model_info.pkl'), 'rb') as f:
                model_info = pickle.load(f)
            with open(os.path.join(model_dir, 'pipeline.pkl'), 'rb') as f:
                pipeline = pickle.load(f)
            with open(os.path.join(model_dir, 'lexicons.pkl'), 'rb') as f:
                lexicons = pickle.load(f)
            instance = cls(
                pipeline=pipeline,
                labels=model_info.get('labels', ['negative', 'neutral', 'positive']),
                lexicons=lexicons
            )
            logger.info(f"Enhanced sentiment analyzer loaded successfully from {model_dir}")
            return instance
        except Exception as e:
            logger.error(f"Error loading enhanced sentiment analyzer: {e}")
            return None

###############################################################################
# Pomocná funkce pro analýzu sentimentu - OPRAVENÁ VERZE
###############################################################################
def analyze_sentiment(text):
    if not text or not isinstance(text, str):
        return {
            'sentiment': 'neutral',
            'sentiment_score': 5.0,
            'sentiment_features': {
                'positive_word_count': 0,
                'negative_word_count': 0,
                'sentiment_ratio': 1.0
            },
            'positive_words': [],  # Vždy seznam
            'negative_words': [],  # Vždy seznam
            'sentiment_reason': "No text provided"
        }
    try:
        if sentiment_model is None:
            return {
                'sentiment': 'neutral',
                'sentiment_score': 5.0,
                'sentiment_features': {
                    'positive_word_count': 0,
                    'negative_word_count': 0,
                    'sentiment_ratio': 1.0
                },
                'positive_words': [],
                'negative_words': [],
                'sentiment_reason': "Sentiment model not loaded"
            }
        result = sentiment_model.explain_prediction(text)
        
        # Ověření, že positive_words a negative_words jsou skutečně seznamy
        if not isinstance(result.get('positive_words', []), list):
            result['positive_words'] = []
        if not isinstance(result.get('negative_words', []), list):
            result['negative_words'] = []
            
        # Vytvoření slovníku sentiment_features
        sentiment_features = {
            'positive_word_count': result.get('positive_word_count', 0),
            'negative_word_count': result.get('negative_word_count', 0),
            'sentiment_ratio': result.get('sentiment_ratio', 1.0)
        }
        
        return {
            'sentiment': result.get('predicted_sentiment', 'neutral'),
            'sentiment_score': result.get('sentiment_score', 5.0),
            'sentiment_features': sentiment_features,
            'positive_words': result.get('positive_words', []),
            'negative_words': result.get('negative_words', []),
            'sentiment_reason': result.get('reason', "Žádný důvod nebyl uveden")
        }
    except Exception as e:
        logger.error(f"Chyba při analýze sentimentu: {e}")
        return {
            'sentiment': 'neutral',
            'sentiment_score': 5.0,
            'sentiment_features': {
                'positive_word_count': 0,
                'negative_word_count': 0,
                'sentiment_ratio': 1.0
            },
            'positive_words': [],
            'negative_words': [],
            'sentiment_reason': f"Analysis failed: {str(e)}"
        }

###############################################################################
# Funkce pro extrakci textu článku
###############################################################################
def extract_article_text(url, source):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "cs,en-US;q=0.7,en;q=0.3",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
            "DNT": "1",
            "Upgrade-Insecure-Requests": "1",
            "Referer": "https://www.google.com/"
        }
        domain = urlparse(url).netloc.lower()
        logger.info(f"Extracting text from {url} (domain: {domain})")
        session = requests.Session()
        response = session.get(url, headers=headers, timeout=15, verify=False)
        if response.encoding == 'ISO-8859-1':
            response.encoding = 'utf-8'
        soup = BeautifulSoup(response.content, "html.parser")
        for element in soup(["script", "style", "iframe", "noscript", "header", "footer", "aside", "nav", "form", "button", "figure"]):
            element.decompose()
        article_title = ""
        title_tag = soup.find('title')
        if title_tag:
            article_title = title_tag.get_text().strip()
        text = ""
        if "seznam" in domain:
            logger.info("Attempting Seznam Zpravy specific extraction")
            selectors = ['div.article-body', 'article.article']
            for selector in selectors:
                content = soup.select_one(selector)
                if content:
                    paragraphs = content.find_all(['p', 'h2', 'h3', 'h4', 'h5'])
                    if paragraphs:
                        text = ' '.join([p.get_text().strip() for p in paragraphs])
                        if len(text) > 100:
                            break
        elif "novinky.cz" in domain:
            logger.info("Attempting Novinky.cz specific extraction")
            selectors = ['div.articleBody', 'article.article']
            for selector in selectors:
                content = soup.select_one(selector)
                if content:
                    paragraphs = content.find_all(['p', 'h2', 'h3', 'h4'])
                    if paragraphs:
                        text = ' '.join([p.get_text().strip() for p in paragraphs])
                        if len(text) > 100:
                            break
        else:
            logger.info("Using generic extraction methods")
            content_element = soup.find('article') or soup.find('main')
            if content_element:
                paragraphs = content_element.find_all(['p', 'h2', 'h3', 'h4'])
                if paragraphs:
                    text = ' '.join([p.get_text().strip() for p in paragraphs])
            if not text or len(text) < 100:
                paragraphs = soup.find_all('p')
                if paragraphs and len(paragraphs) > 3:
                    text = ' '.join([p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 20])
            if not text or len(text) < 100:
                body = soup.find('body')
                if body:
                    text = body.get_text(separator=' ', strip=True)
                    text_parts = text.split('\n')
                    longest_part = max(text_parts, key=len, default='')
                    if len(longest_part) > 200:
                        text = longest_part
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'Cookies\s+?na\s+?tomto\s+?webu', '', text)
        text = re.sub(r'Nastavení\s+?souhlasu\s+?s\s+?personalizací', '', text)
        text = re.sub(r'Souhlasím\s+?se\s+?zpracováním\s+?osobních\s+?údajů', '', text)
        text = re.sub(r'Tento\s+?web\s+?používá\s+?k\s+?poskytování\s+?služeb.+?soubory\s+?cookie', '', text)
        text = re.sub(r'There\s+?was\s+?an\s+?error\s+?loading\s+?the\s+?script', '', text)
        if not text or len(text) < 50:
            if article_title:
                logger.warning(f"Using title as fallback for {url}")
                text = article_title
                meta_desc = soup.find('meta', attrs={'name': 'description'})
                if meta_desc and meta_desc.get('content'):
                    text += " " + meta_desc.get('content')
            else:
                logger.error(f"Failed to extract any text from {url}")
                text = f"Nepodařilo se načíst obsah článku. Zkuste otevřít původní článek: {url}"
        char_count = len(text)
        word_count = len(text.split())
        logger.info(f"Extracted {char_count} characters and {word_count} words from {url}")
        return text, char_count, word_count, None
    except Exception as e:
        logger.error(f"Error extracting text from {url}: {e}")
        return f"Chyba při načítání obsahu: {str(e)}", 0, 0, ""

###############################################################################
# Scraping nejnovějších článků
###############################################################################
def scrape_newest_articles():
    sources_to_scrape = {
        "novinky": "https://www.novinky.cz/rss",
        "seznamzpravy": "https://www.seznamzpravy.cz/rss/",
        "zpravy.aktualne": "https://zpravy.aktualne.cz/rss/",
        "idnes": "https://servis.idnes.cz/rss.aspx?c=zpravodaj",
        "ct24": "https://ct24.ceskatelevize.cz/rss/hlavni-zpravy"
    }
    all_articles = []
    for source_name, rss_url in sources_to_scrape.items():
        try:
            logger.info(f"Scraping {source_name}...")
            feed = feedparser.parse(rss_url)
            for i, entry in enumerate(feed.entries[:5]):
                if i >= 5:
                    break
                title = entry.title if hasattr(entry, 'title') else "Untitled"
                url = entry.link if hasattr(entry, 'link') else ""
                if not url:
                    continue
                pub_date = None
                if hasattr(entry, 'published'):
                    pub_date = entry.published
                elif hasattr(entry, 'pubDate'):
                    pub_date = entry.pubDate
                elif hasattr(entry, 'updated'):
                    pub_date = entry.updated
                article_text, char_count, word_count, _ = extract_article_text(url, source_name)
                if char_count < 50:
                    logger.warning(f"Skipping article with too little content: {title}")
                    continue
                article = {
                    "Title": title,
                    "Content": article_text,
                    "Source": source_name,
                    "Category": None,
                    "PublishDate": pub_date,
                    "ArticleUrl": url,
                    "ArticleLength": char_count,
                    "WordCount": word_count,
                    "ScrapedDate": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "positive_word_count": 0,
                    "negative_word_count": 0,
                    "sentiment_ratio": 1.0
                }
                all_articles.append(article)
                logger.info(f"Scraped: {title} ({char_count} chars, {word_count} words)")
                time.sleep(1.0)
        except Exception as e:
            logger.error(f"Error scraping {source_name}: {e}")
    if all_articles:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_path = os.path.join(project_root, 'data', 'scraped', f'articles_{timestamp}.json')
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(all_articles, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(all_articles)} articles to {file_path}")
        display_file = os.path.join(project_root, 'data', 'display', 'all_articles.json')
        try:
            if os.path.exists(display_file):
                with open(display_file, 'r', encoding='utf-8') as f:
                    existing_articles = json.load(f)
                max_id = 0
                for article in existing_articles:
                    if 'Id' in article and article['Id'] > max_id:
                        max_id = article['Id']
                for i, article in enumerate(all_articles):
                    article['Id'] = max_id + i + 1
                combined_articles = existing_articles + all_articles
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
                for i, article in enumerate(all_articles):
                    article['Id'] = i + 1
                with open(display_file, 'w', encoding='utf-8') as f:
                    json.dump(all_articles, f, ensure_ascii=False, indent=2)
                logger.info(f"Created {display_file} with {len(all_articles)} articles")
        except Exception as e:
            logger.error(f"Error updating display file: {e}")
    return len(all_articles)

###############################################################################
# Spuštění denního scraperu (reload data)
###############################################################################
def run_daily_scraper():
    logger.info("Latest news scraper started - sbírám nejnovější zprávy")
    try:
        scrape_count = scrape_newest_articles()
        logger.info(f"Scraped {scrape_count} new articles")
    except Exception as e:
        logger.error(f"Error in scraper: {e}")
    load_data()
    logger.info("Latest news scraper completed successfully")

###############################################################################
# Funkce load_data() – načte články a aplikuje sentiment analýzu
###############################################################################
def load_data():
    global articles_df, sentiment_model, loaded_date, enhanced_models
    logger.info(f"Hard-coded path for sentiment analyzer: {HARDCODED_SENTIMENT_MODEL_PATH}")
    if os.path.exists(HARDCODED_SENTIMENT_MODEL_PATH):
        loaded_model = EnhancedSentimentAnalyzer.load_model(HARDCODED_SENTIMENT_MODEL_PATH)
        if loaded_model:
            sentiment_model = loaded_model
            logger.info("Sentiment model successfully loaded from hard-coded path.")
            enhanced_models = True
        else:
            logger.error("Failed to load sentiment model from the hard-coded path.")
            sentiment_model = None
    else:
        logger.error(f"Sentiment analyzer model directory not found at {HARDCODED_SENTIMENT_MODEL_PATH}")
        sentiment_model = None

    display_file = os.path.join(project_root, 'data', 'display', 'all_articles.json')
    if os.path.exists(display_file):
        try:
            with open(display_file, 'r', encoding='utf-8') as f:
                articles = json.load(f)
            articles_df = pd.DataFrame(articles)
            logger.info(f"Loaded {len(articles_df)} articles from {display_file}")
        except Exception as e:
            logger.error(f"Failed to load articles data: {e}")
            articles_df = pd.DataFrame(columns=[
                'Id', 'Title', 'Content', 'Source', 'PublishDate',
                'ArticleUrl', 'ArticleLength', 'WordCount'
            ])
    else:
        logger.warning("No articles data file found, created empty DataFrame")
        articles_df = pd.DataFrame(columns=[
            'Id', 'Title', 'Content', 'Source', 'PublishDate',
            'ArticleUrl', 'ArticleLength', 'WordCount'
        ])

    if 'Id' not in articles_df.columns:
        articles_df['Id'] = range(1, len(articles_df) + 1)

    for col in ['Title', 'Content', 'Source', 'PublishDate', 'ArticleUrl', 'ArticleLength', 'WordCount']:
        if col not in articles_df.columns:
            articles_df[col] = None

    loaded_date = datetime.now()
    app.config['articles_df'] = articles_df

    if sentiment_model is not None and len(articles_df) > 0:
        if 'sentiment' not in articles_df.columns or articles_df['sentiment'].isna().any():
            logger.info("Applying sentiment model to articles...")
            texts = articles_df['Content'].fillna('').tolist()
            sentiment_results = []
            for t in texts:
                sentiment_result = analyze_sentiment(t)
                sentiment_results.append(sentiment_result)
            
            # Přidáme sloupce s výsledky sentiment analýzy
            articles_df['sentiment'] = [r['sentiment'] for r in sentiment_results]
            articles_df['sentiment_score'] = [r['sentiment_score'] for r in sentiment_results]
            articles_df['sentiment_reason'] = [r['sentiment_reason'] for r in sentiment_results]
            articles_df['positive_words'] = [r['positive_words'] for r in sentiment_results]
            articles_df['negative_words'] = [r['negative_words'] for r in sentiment_results]
            
            # Vytvoříme slovník sentiment_features
            sentiment_features = []
            for r in sentiment_results:
                sentiment_features.append(r['sentiment_features'])
            articles_df['sentiment_features'] = sentiment_features
            
            logger.info("Sentiment analysis completed.")
            try:
                with open(display_file, 'w', encoding='utf-8') as f:
                    json.dump(articles_df.to_dict('records'), f, ensure_ascii=False, indent=2)
                logger.info("Processed data saved.")
            except Exception as e:
                logger.error(f"Error saving updated articles: {e}")

###############################################################################
# Dummy route pro /categories – přesměruje na /articles
###############################################################################
@app.route('/categories')
def categories():
    return redirect(url_for('articles'))

###############################################################################
# Routes
###############################################################################
@app.route('/')
def index():
    if articles_df is None:
        load_data()
    stats = {
        'total_articles': len(articles_df) if articles_df is not None else 0,
        'sources': articles_df['Source'].nunique() if articles_df is not None and len(articles_df) > 0 else 0,
        'categories': 0,  # Dummy hodnota
        'date_range': {
            'from': articles_df['PublishDate'].min() if articles_df is not None and len(articles_df) > 0 else None,
            'to': articles_df['PublishDate'].max() if articles_df is not None and len(articles_df) > 0 else None
        },
        'newest_articles': articles_df.sort_values('Id', ascending=False).head(5).to_dict('records') if len(articles_df) > 0 else [],
        'top_sources': articles_df['Source'].value_counts().head(5).to_dict() if len(articles_df) > 0 else {},
        'top_categories': {},  # Dummy prázdný slovník
        'loaded_date': loaded_date,
        'enhanced_models': enhanced_models
    }
    return render_template('index.html', stats=stats)

@app.route('/articles')
def articles():
    if articles_df is None:
        load_data()
    source = request.args.get('source', None)
    sentiment = request.args.get('sentiment', None)
    search = request.args.get('search', None)

    if len(articles_df) == 0:
        return render_template('articles.html', articles=[], sources=[], sentiments=[], 
                               page=1, total_pages=0, total_articles=0)

    filtered_df = articles_df.copy()
    if source and source != 'all':
        filtered_df = filtered_df[filtered_df['Source'] == source]
    if sentiment and sentiment != 'all' and 'sentiment' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['sentiment'] == sentiment]
    if search:
        filtered_df = filtered_df[
            filtered_df['Title'].str.contains(search, case=False, na=False) |
            filtered_df['Content'].str.contains(search, case=False, na=False)
        ]

    page = int(request.args.get('page', 1))
    per_page = 20
    total_pages = (len(filtered_df) + per_page - 1) // per_page
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    paged_df = filtered_df.sort_values('Id', ascending=False).iloc[start_idx:end_idx]
    articles_list = paged_df.to_dict('records')
    sources = sorted(articles_df['Source'].dropna().unique().tolist())
    sentiments = sorted(articles_df['sentiment'].dropna().unique().tolist()) if 'sentiment' in articles_df.columns else []

    return render_template(
        'articles.html',
        articles=articles_list,
        sources=sources,
        sentiments=sentiments,
        current_source=source,
        current_sentiment=sentiment,
        current_search=search,
        page=page,
        total_pages=total_pages,
        total_articles=len(filtered_df)
    )

@app.route('/article/<int:article_id>')
def article_detail(article_id):
    if articles_df is None:
        load_data()
    article = articles_df[articles_df['Id'] == article_id]
    if len(article) == 0:
        return render_template('error.html', message=f"Článek s ID {article_id} nebyl nalezen")
    
    article_data = article.iloc[0].to_dict()
    content = article_data.get('Content', '')
    
    # Vždy znovu proveď analýzu sentimentu pro aktuální zobrazení
    sentiment_res = analyze_sentiment(content)
    
    # Aktualizace článku správnými datovými strukturami
    article_data['sentiment'] = sentiment_res['sentiment']
    article_data['sentiment_score'] = sentiment_res['sentiment_score']
    article_data['sentiment_reason'] = sentiment_res['sentiment_reason']
    article_data['positive_words'] = sentiment_res['positive_words']
    article_data['negative_words'] = sentiment_res['negative_words']
    article_data['sentiment_features'] = sentiment_res['sentiment_features']
    
    return render_template('article_detail.html', article=article_data)

@app.route('/sources')
def sources():
    if articles_df is None:
        load_data()
    if len(articles_df) > 0:
        source_counts = articles_df['Source'].value_counts().reset_index()
        source_counts.columns = ['source', 'count']
    else:
        source_counts = pd.DataFrame(columns=['source', 'count'])
    return render_template('sources.html', sources=source_counts.to_dict('records'))

###############################################################################
# API endpoints
###############################################################################
@app.route('/api/analyze', methods=['POST'])
def analyze_text():
    data = request.json
    text = data.get('text', '')
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    result = {
        'text': text,
        'length': len(text),
        'word_count': len(text.split())
    }
    sentiment_res = analyze_sentiment(text)
    result.update(sentiment_res)
    return jsonify(result)

@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    if request.method == 'POST':
        text = request.form.get('text', '')
        result = {
            'text': text,
            'length': len(text),
            'word_count': len(text.split())
        }
        sentiment_res = analyze_sentiment(text)
        result.update(sentiment_res)
        return render_template('analyze.html', result=result)
    return render_template('analyze.html')

@app.route('/api/article_chatbot', methods=['POST'])
def article_chatbot_api():
    try:
        data = request.json
        message = data.get('message', '')
        article_id = data.get('article_id')
        if not message:
            return jsonify({'response': 'Prosím, napište nějakou zprávu.'})
        if not article_id:
            return jsonify({'response': 'Chybí ID článku.'})
        if articles_df is None:
            load_data()
        article = articles_df[articles_df['Id'] == int(article_id)]
        if len(article) == 0:
            return jsonify({'response': f'Článek s ID {article_id} nebyl nalezen.'})
        article_data = article.iloc[0].to_dict()
        
        # Zajistit, že data článku mají potřebné atributy pro chatbota
        content = article_data.get('Content', '')
        sentiment_res = analyze_sentiment(content)
        article_data.update(sentiment_res)
        
        response = article_chatbot.respond_to_article_query(message, article_data)
        return jsonify({'response': response})
    except Exception as e:
        logger.error(f"Error in article chatbot API: {e}")
        return jsonify({'response': 'Omlouvám se, ale něco se pokazilo. Zkuste to prosím znovu.'})

@app.route('/reload_data')
def reload_data():
    thread = threading.Thread(target=load_data)
    thread.start()
    return redirect(url_for('index'))

###############################################################################
# Chybové stránky
###############################################################################
@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html', message="Stránka nebyla nalezena"), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('error.html', message="Chyba serveru"), 500

###############################################################################
# Spuštění aplikace
###############################################################################
if __name__ == "__main__":
    load_data()
    app.run(debug=True, host='0.0.0.0', port=5000)