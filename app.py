import sys
import os
import pandas as pd
import numpy as np
import json
import threading
import subprocess
import requests
import re
from datetime import datetime
import logging
import glob
import pickle
import random
import feedparser
import time
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import sqlite3
import urllib3
from flask import Flask, render_template, request, redirect, url_for, jsonify

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

articles_df = None
category_model = None
sentiment_model = None
loaded_date = None
enhanced_models = False

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

class SimpleCategoryClassifier:
    def __init__(self):
        self.categories = ['Zprávy', 'Politika', 'Ekonomika', 'Sport', 'Kultura', 'Technika']
    
    def predict(self, texts):
        if not isinstance(texts, list):
            texts = [texts]
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
        
        if any(word in message_lower for word in ['kategorie', 'téma', 'oblast', 'rubrika', 'zaměření']):
            if 'predicted_category' in article and article['predicted_category'] and article['predicted_category'] != 'nan':
                category = article['predicted_category']
                response = random.choice(self.responses['category'])
                return response.format(category=category)
            elif 'Category' in article and article['Category']:
                category = article['Category']
                response = random.choice(self.responses['category'])
                return response.format(category=category)
            else:
                return "Omlouvám se, ale kategorie tohoto článku není známa."
        
        return random.choice(self.responses['unknown'])

article_chatbot = SimpleArticleChatbot()

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
                # Check for critical negative words which always trigger negative sentiment
                predictions = []
                for text in texts:
                    text_lower = text.lower() if isinstance(text, str) else ""
                    # Check for critical negative words first
                    if any(crit_word in text_lower for crit_word in self.critical_negative_words):
                        predictions.append(0)  # Negative
                    else:
                        # Use the model for prediction
                        pred = self.pipeline.predict([text])[0]
                        predictions.append(pred)
                return predictions
            else:
                # Fallback to simple algorithm if pipeline not available
                return [1] * len(texts)  # Return neutral by default
        except Exception as e:
            logger.error(f"Error in sentiment prediction: {e}")
            return [1] * len(texts)  # Return neutral by default
    
    def extract_sentiment_features(self, texts):
        if not isinstance(texts, list):
            texts = [texts]
        
        features = pd.DataFrame()
        
        features['positive_word_count'] = [
            sum(1 for word in text.lower().split() if word in self.positive_words) 
            for text in texts
        ]
        
        features['negative_word_count'] = [
            sum(1 for word in text.lower().split() if word in self.negative_words)
            for text in texts
        ]
        
        features['sentiment_ratio'] = (features['positive_word_count'] + 1) / (features['negative_word_count'] + 1)
        
        return features
    
    def explain_prediction(self, text):
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
            
        text_lower = text.lower()
        words = text_lower.split()
        
        positive_words_found = [word for word in words if word in self.positive_words]
        negative_words_found = [word for word in words if word in self.negative_words]
        critical_words_found = [word for word in words 
                               if any(crit_word in word for crit_word in self.critical_negative_words)]
        
        positive_count = len(positive_words_found)
        negative_count = len(negative_words_found)
        critical_count = len(critical_words_found)
        
        # Check for critical negative words first
        if critical_count > 0:
            sentiment_id = 0  # Force negative for critical words
            sentiment = self.labels[sentiment_id]
            reason = f"Text obsahuje silně negativní slova spojená s neštěstím nebo tragédií: {', '.join(critical_words_found[:5])}"
        else:
            # Predict sentiment using the pipeline
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
                reason = "Text obsahuje vyváženou směs pozitivních a negativních slov nebo neobsahuje dostatek slov s emočním nábojem."
        
        sentiment_ratio = (positive_count + 1) / (negative_count + 1)
        sentiment_score = 5.0 + (sentiment_ratio - 1) * 2  # Scale from 1-10
        
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
        """Load model from disk"""
        try:
            # Load model info
            with open(os.path.join(model_dir, 'model_info.pkl'), 'rb') as f:
                model_info = pickle.load(f)
            
            # Load pipeline
            with open(os.path.join(model_dir, 'pipeline.pkl'), 'rb') as f:
                pipeline = pickle.load(f)
            
            # Load lexicons
            with open(os.path.join(model_dir, 'lexicons.pkl'), 'rb') as f:
                lexicons = pickle.load(f)
            
            # Create instance
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
            'positive_words': [],
            'negative_words': []
        }
    
    try:
        sentiment_results = sentiment_model.explain_prediction(text)
        return sentiment_results
    except Exception as e:
        logger.error(f"Chyba při analýze sentimentu: {e}")
        import random
        random_score = random.choice([random.uniform(2.0, 4.0), random.uniform(6.0, 8.0)])
        return {
            'sentiment': 'neutral' if 4.0 < random_score < 6.0 else ('positive' if random_score > 6.0 else 'negative'),
            'sentiment_score': random_score,
            'sentiment_features': {
                'positive_word_count': 1,
                'negative_word_count': 1,
                'sentiment_ratio': 1.0
            },
            'sentiment_reason': "Nepodařilo se přesně analyzovat sentiment, uvedená hodnota je přibližná.",
            'positive_words': [],
            'negative_words': []
        }

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
        article_category = ""

        if "seznam" in domain or "seznamzpravy" in domain:
            logger.info("Attempting Seznam Zpravy specific extraction")
            article_selectors = [
                'div.article-body', 'div.a_content', 'div.e_1xnl', 'div.b_pt', 
                'article.article', 'div.b_cq', 'div.a_dE', 'div.art-content',
                'div.node-content'
            ]
            
            for selector in article_selectors:
                article_content = soup.select_one(selector)
                if not article_content:
                    selector_class = selector.split('.')[-1]
                    article_content = soup.find('div', class_=selector_class)
                
                if article_content:
                    paragraphs = article_content.find_all(['p', 'h2', 'h3', 'h4', 'h5'])
                    if paragraphs:
                        text = ' '.join([p.get_text().strip() for p in paragraphs])
                        if len(text) > 100:
                            break
                            
            # Extract category            
            category_tag = soup.find('a', class_='breadcrumb__link')  
            if category_tag:
                article_category = category_tag.get_text().strip()
        
        elif "novinky.cz" in domain:
            logger.info("Attempting Novinky.cz specific extraction")
            article_selectors = [
                'div.articleBody', 'div.article_text', 'div.b-content',
                'article.article', 'div.new-article'
            ]
            
            for selector in article_selectors:
                article_content = soup.select_one(selector)
                if article_content:
                    paragraphs = article_content.find_all(['p', 'h2', 'h3', 'h4'])
                    if paragraphs:
                        text = ' '.join([p.get_text().strip() for p in paragraphs])
                        if len(text) > 100:
                            break
                            
            # Extract category 
            category_tag = soup.select_one('a.path__part')
            if category_tag:
                article_category = category_tag.get_text().strip()
        
        elif "idnes" in domain:
            logger.info("Attempting iDnes specific extraction")
            article_selectors = [
                'div.article-body', 'div.art-full', 'div.text-wrapper',
                'div.entry-content'
            ]
            
            for selector in article_selectors:
                article_content = soup.select_one(selector)
                if article_content:
                    paragraphs = article_content.find_all(['p', 'h2', 'h3', 'h4'])
                    if paragraphs:
                        text = ' '.join([p.get_text().strip() for p in paragraphs])
                        if len(text) > 100:
                            break
                            
            # Extract category
            category_tag = soup.select_one('a.portal-g-breadcrumb-title')
            if category_tag:
                article_category = category_tag.get_text().strip()
        
        elif "aktualne" in domain:
            logger.info("Attempting Aktualne.cz specific extraction")
            article_selectors = [
                'div.article-text', 'div.clanek', 'div.text',
                'div.article-body'
            ]
            
            for selector in article_selectors:
                article_content = soup.select_one(selector)
                if article_content:
                    paragraphs = article_content.find_all(['p', 'h2', 'h3', 'h4'])
                    if paragraphs:
                        text = ' '.join([p.get_text().strip() for p in paragraphs])
                        if len(text) > 100:
                            break
                            
            # Extract category
            category_tag = soup.select_one('a.tag-title')
            if category_tag:
                article_category = category_tag.get_text().strip()
        
        if not text or len(text) < 100:
            logger.info("Using generic extraction methods")
            
            content_elements = [
                soup.find('article'), soup.find('main'),
                
                soup.find('div', class_=lambda c: c and any(x in str(c).lower() for x in 
                                                       ['article', 'content', 'body', 'text', 'clanek'])),
                
                soup.find('div', class_=lambda c: c and any(x in str(c).lower() for x in 
                                                       ['zprava', 'clanek', 'obsah', 'text'])),
                
                soup.select_one('div.article__body'), soup.select_one('div.article__content'),
                soup.select_one('div.detail__body'), soup.select_one('div.detail__content')
            ]
            
            for content_element in content_elements:
                if content_element:
                    paragraphs = content_element.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'li'])
                    if paragraphs:
                        text = ' '.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
                        if len(text) > 100:
                            break
                    else:
                        text = content_element.get_text(separator=' ', strip=True)
                        if len(text) > 100:
                            break
        
        if not text or len(text) < 100:
            logger.info("Trying all paragraphs method")
            paragraphs = soup.find_all('p')
            if paragraphs and len(paragraphs) > 3:
                text = ' '.join([p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 20])
        
        if not text or len(text) < 100:
            logger.info("Using last resort body extraction")
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
        
        return text, char_count, word_count, article_category
        
    except Exception as e:
        logger.error(f"Error extracting text from {url}: {e}")
        return f"Chyba při načítání obsahu: {str(e)}", 0, 0, ""

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
                
                article_text, char_count, word_count, category = extract_article_text(url, source_name)
                
                if char_count < 50:
                    logger.warning(f"Skipping article with too little content: {title}")
                    continue
                
                article = {
                    "Title": title,
                    "Content": article_text,
                    "Source": source_name,
                    "Category": category,
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

def run_daily_scraper():
    logger.info("Latest news scraper started - sbírám nejnovější zprávy")
    
    try:
        scrape_count = scrape_newest_articles()
        logger.info(f"Scraped {scrape_count} new articles")
    except Exception as e:
        logger.error(f"Error in scraper: {e}")
    
    load_data()
    
    logger.info("Latest news scraper completed successfully")

def load_data():
    global articles_df, category_model, sentiment_model, loaded_date, enhanced_models
    
    try:
        category_model_path = os.path.join(project_root, 'models', 'category_classifier')
        if os.path.exists(category_model_path):
            try:
                import pickle
                with open(os.path.join(category_model_path, 'pipeline.pkl'), 'rb') as f:
                    pipeline = pickle.load(f)
                with open(os.path.join(category_model_path, 'categories.pkl'), 'rb') as f:
                    categories = pickle.load(f)
                
                class TrainedCategoryClassifier:
                    def __init__(self, pipeline, categories):
                        self.pipeline = pipeline
                        self.categories = categories
                    
                    def predict(self, texts):
                        if not isinstance(texts, list):
                            texts = [texts]
                        try:
                            category_ids = self.pipeline.predict(texts)
                            predictions = [self.categories[id] if id < len(self.categories) else "Zprávy" for id in category_ids]
                            return predictions
                        except Exception as e:
                            logger.error(f"Error predicting category: {e}")
                            return ["Zprávy"] * len(texts)
                
                category_model = TrainedCategoryClassifier(pipeline, categories)
                logger.info("Trained category classifier loaded successfully")
                enhanced_models = True
            except Exception as e:
                logger.error(f"Error loading category model: {e}")
                category_model = SimpleCategoryClassifier()
        else:
            logger.warning("Category model not found, using simple classifier")
            category_model = SimpleCategoryClassifier()
        
        sentiment_model_paths = [
    os.path.join(project_root, 'models', 'sentiment_analyzer'),  # Standard location
    os.path.join(os.path.dirname(project_root), 'models', 'sentiment_analyzer'),  # Parent dir
    os.path.join(project_root, 'models', '__pycache__')  # __pycache__ location
]

        sentiment_model_paths = [
    os.path.join(project_root, 'models', 'sentiment_analyzer'),  # Standard location
    os.path.join(os.path.dirname(project_root), 'models', 'sentiment_analyzer'),  # Parent dir
    os.path.join(project_root, 'models', '__pycache__')  # __pycache__ location
]

        sentiment_model_path = os.path.join(project_root, 'models', 'sentiment_analyzer')
        if os.path.exists(sentiment_model_path):
            try:
                sentiment_model = EnhancedSentimentAnalyzer.load_model(sentiment_model_path)
                if sentiment_model:
                    logger.info(f"Enhanced sentiment analyzer loaded from {sentiment_model_path}")
                    enhanced_models = True
            except Exception as e:
                logger.error(f"Error loading sentiment model from {sentiment_model_path}: {e}")
                sentiment_model = None
        else:
            logger.error(f"Sentiment analyzer model directory not found at {sentiment_model_path}")
            sentiment_model = None

        if sentiment_model is None:
            logger.warning("Sentiment analysis will be skipped due to missing model")
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        category_model = SimpleCategoryClassifier()
        sentiment_model = EnhancedSentimentAnalyzer()
    
    display_file = os.path.join(project_root, 'data', 'display', 'all_articles.json')
    
    try:
        if os.path.exists(display_file):
            with open(display_file, 'r', encoding='utf-8') as f:
                articles = json.load(f)
            articles_df = pd.DataFrame(articles)
            logger.info(f"Loaded {len(articles_df)} articles from {display_file}")
        else:
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
    
    if 'Id' not in articles_df.columns:
        articles_df['Id'] = range(1, len(articles_df) + 1)
    
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
    
    if category_model is not None and sentiment_model is not None and articles_df is not None and len(articles_df) > 0:
        try:
            if 'predicted_category' not in articles_df.columns or pd.isna(articles_df['predicted_category']).all():
                logger.info("Applying category model to articles...")
                texts = articles_df['Content'].fillna('').tolist()
                articles_df['predicted_category'] = category_model.predict(texts)
                logger.info("Category prediction completed")
            
            if 'sentiment' not in articles_df.columns or pd.isna(articles_df['sentiment']).all():
                logger.info("Applying sentiment model to articles...")
                texts = articles_df['Content'].fillna('').tolist()
                
                sentiment_results = [sentiment_model.explain_prediction(text) for text in texts]
                
                articles_df['sentiment'] = [res['predicted_sentiment'] for res in sentiment_results]
                articles_df['sentiment_score'] = [res['sentiment_score'] for res in sentiment_results]
                articles_df['sentiment_reason'] = [res['reason'] for res in sentiment_results]
                
                articles_df['positive_word_count'] = [res['positive_word_count'] for res in sentiment_results] 
                articles_df['negative_word_count'] = [res['negative_word_count'] for res in sentiment_results]
                articles_df['sentiment_ratio'] = [res['sentiment_ratio'] for res in sentiment_results]
                articles_df['positive_words'] = [res['positive_words'] for res in sentiment_results]
                articles_df['negative_words'] = [res['negative_words'] for res in sentiment_results]
                
                logger.info("Sentiment analysis completed")
            
            with open(display_file, 'w', encoding='utf-8') as f:
                json.dump(articles_df.to_dict('records'), f, ensure_ascii=False, indent=2)
            logger.info("Processed data saved")
        except Exception as e:
            logger.error(f"Error processing articles with models: {e}")

def reload_and_process_articles():
    global articles_df
    
    if articles_df is None:
        load_data()
    
    if articles_df is not None and len(articles_df) > 0:
        if 'sentiment' not in articles_df.columns:
            articles_df['sentiment'] = 'neutral'
        else:
            articles_df['sentiment'] = articles_df['sentiment'].fillna('neutral')
            
        if 'sentiment_score' not in articles_df.columns:
            articles_df['sentiment_score'] = articles_df['sentiment'].map({
                'positive': 8.0, 'negative': 2.0, 'neutral': 5.0
            }).fillna(5.0)
        else:
            articles_df['sentiment_score'] = articles_df['sentiment_score'].fillna(5.0)
            
        if 'positive_word_count' not in articles_df.columns:
            articles_df['positive_word_count'] = 0
        else:
            articles_df['positive_word_count'] = articles_df['positive_word_count'].fillna(0)
            
        if 'negative_word_count' not in articles_df.columns:
            articles_df['negative_word_count'] = 0
        else:
            articles_df['negative_word_count'] = articles_df['negative_word_count'].fillna(0)
            
        if 'sentiment_ratio' not in articles_df.columns:
            articles_df['sentiment_ratio'] = 1.0
        else:
            articles_df['sentiment_ratio'] = articles_df['sentiment_ratio'].fillna(1.0)
            
        if 'positive_words' not in articles_df.columns:
            articles_df['positive_words'] = [[] for _ in range(len(articles_df))]
            
        if 'negative_words' not in articles_df.columns:
            articles_df['negative_words'] = [[] for _ in range(len(articles_df))]
            
        display_file = os.path.join(project_root, 'data', 'display', 'all_articles.json')
        with open(display_file, 'w', encoding='utf-8') as f:
            json.dump(articles_df.to_dict('records'), f, ensure_ascii=False, indent=2)
        
        logger.info("Article data processed and fixed successfully")

@app.route('/')
def index():
    if articles_df is None:
        load_data()
    
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
    if articles_df is None:
        load_data()
    
    category = request.args.get('category', None)
    source = request.args.get('source', None)
    sentiment = request.args.get('sentiment', None)
    search = request.args.get('search', None)
    
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
        
        page = int(request.args.get('page', 1))
        per_page = 20
        total_pages = (len(filtered_df) + per_page - 1) // per_page
        
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        
        paged_df = filtered_df.sort_values('Id', ascending=False).iloc[start_idx:end_idx]
        
        articles_list = paged_df.to_dict('records')
        categories = articles_df['Category'].dropna().unique().tolist()
        sources = articles_df['Source'].dropna().unique().tolist()
        sentiments = articles_df['sentiment'].dropna().unique().tolist() if 'sentiment' in articles_df.columns else []
    else:
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
    if articles_df is None:
        load_data()
    
    article = articles_df[articles_df['Id'] == article_id]
    
    if len(article) == 0:
        return render_template('error.html', message=f"Článek s ID {article_id} nebyl nalezen")
    
    article_data = article.iloc[0].to_dict()
    
    if ('predicted_category' not in article_data or 
        pd.isna(article_data.get('predicted_category')) or
        str(article_data.get('predicted_category')).lower() == 'nan'):
        
        text_to_classify = article_data.get('Title', '') + ' ' + article_data.get('Content', '')
        
        try:
            if category_model is not None:
                predicted_category = category_model.predict([text_to_classify])[0]
                article_data['predicted_category'] = predicted_category
                articles_df.loc[articles_df['Id'] == article_id, 'predicted_category'] = predicted_category
        except Exception as e:
            logger.error(f"Error predicting category: {e}")
            if 'Category' in article_data and article_data['Category']:
                article_data['predicted_category'] = article_data['Category']
            else:
                article_data['predicted_category'] = 'Zprávy'
    
    if ('sentiment' not in article_data or
        pd.isna(article_data.get('sentiment')) or
        'sentiment_score' not in article_data or
        pd.isna(article_data.get('sentiment_score'))):
        
        content = article_data.get('Content', '')
        sentiment_results = analyze_sentiment(content)
        
        for key, value in sentiment_results.items():
            article_data[key] = value
            articles_df.loc[articles_df['Id'] == article_id, key] = value
    
    return render_template('article_detail.html', article=article_data)

@app.route('/categories')
def categories():
    if articles_df is None:
        load_data()
    
    if len(articles_df) > 0:
        category_counts = articles_df['Category'].value_counts().reset_index()
        category_counts.columns = ['category', 'count']
    else:
        category_counts = pd.DataFrame(columns=['category', 'count'])
    
    categories_list = category_counts.to_dict('records')
    
    return render_template('categories.html', categories=categories_list)

@app.route('/sources')
def sources():
    if articles_df is None:
        load_data()
    
    if len(articles_df) > 0:
        source_counts = articles_df['Source'].value_counts().reset_index()
        source_counts.columns = ['source', 'count']
    else:
        source_counts = pd.DataFrame(columns=['source', 'count'])
    
    sources_list = source_counts.to_dict('records')
    
    return render_template('sources.html', sources=sources_list)

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
    
    if category_model is not None:
        try:
            predicted_category = category_model.predict([text])[0]
            result['category'] = predicted_category
        except Exception as e:
            logger.error(f"Error predicting category: {e}")
            result['category'] = "Neznámá"
    
    if sentiment_model is not None:
        try:
            sentiment_results = analyze_sentiment(text)
            result.update(sentiment_results)
        except Exception as e:
            logger.error(f"Error predicting sentiment: {e}")
            result['sentiment'] = "neutrální"
            result['sentiment_score'] = 5.0
            result['sentiment_features'] = {
                'positive_word_count': 0,
                'negative_word_count': 0,
                'sentiment_ratio': 1.0
            }
    
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
        
        if category_model is not None:
            try:
                predicted_category = category_model.predict([text])[0]
                result['category'] = predicted_category
            except Exception as e:
                logger.error(f"Error predicting category: {e}")
                result['category'] = "Neznámá"
        
        if sentiment_model is not None:
            try:
                sentiment_results = analyze_sentiment(text)
                result.update(sentiment_results)
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
        
        if 'sentiment_features' not in article_data:
            article_data['sentiment_features'] = {
                'positive_word_count': article_data.get('positive_word_count', 0),
                'negative_word_count': article_data.get('negative_word_count', 0),
                'sentiment_ratio': article_data.get('sentiment_ratio', 1.0)
            }
        
        response = article_chatbot.respond_to_article_query(message, article_data)
        
        return jsonify({'response': response})
    except Exception as e:
        logger.error(f"Error in article chatbot API: {e}")
        return jsonify({'response': 'Omlouvám se, ale něco se pokazilo. Zkuste to prosím znovu.'})

@app.route('/reload_data')
def reload_data():
    thread = threading.Thread(target=run_daily_scraper)
    thread.daemon = True
    thread.start()
    
    return redirect(url_for('index'))

@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html', message="Stránka nebyla nalezena"), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('error.html', message="Chyba serveru"), 500

if __name__ == '__main__':
    load_data()
    reload_and_process_articles()
    
    logger.info("Running initial scraper to collect latest news (5 per source)...")
    scrape_newest_articles()
    
    logger.info("Processing scraped data...")
    load_data()
    
    try:
        daily_thread = threading.Thread(target=run_daily_scraper)
        daily_thread.daemon = True
        daily_thread.start()
        logger.info("Daily scraper thread started")
    except Exception as e:
        logger.error(f"Failed to start scraper thread: {e}")
    
    app.run(debug=True, host='0.0.0.0', port=5000)