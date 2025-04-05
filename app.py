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
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

articles_df = None
sentiment_model = None
loaded_date = None
enhanced_models = False

# Force the sentiment analyzer model path:
FORCED_SENTIMENT_MODEL_PATH = r"D:\GitHub\Synapse-Omega-FinalProject\models\sentiment_analyzer"

################################################################################
# (Existing CSS/JS code for the sentiment gauge is written to disk here)
################################################################################

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
    const needle = document.getElementById('tachometer-needle');
    const scoreDisplay = document.getElementById('tachometer-score-display');
    if (needle && scoreDisplay) {
        let score = parseFloat(scoreDisplay.textContent);
        if (isNaN(score)) score = 5.0;
        // Convert score (0-10) to an angle: -90deg for 0, +90deg for 10.
        const angle = Math.min(Math.max(score, 0), 10) * 18 - 90;
        setTimeout(() => {
            needle.style.transform = `rotate(${angle}deg)`;
        }, 300);
    }
});
"""

with open(os.path.join(current_dir, 'static', 'css', 'sentiment-gauge.css'), 'w') as f:
    f.write(sentiment_gauge_css)
with open(os.path.join(current_dir, 'static', 'js', 'sentiment-gauge.js'), 'w') as f:
    f.write(sentiment_gauge_js)

################################################################################
# SimpleArticleChatbot and EnhancedSentimentAnalyzer Classes
################################################################################

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
        sentiment_id = self.predict([text])[0]
        sentiment = self.labels[sentiment_id]
        if critical_count > 0:
            reason = f"Text obsahuje silně negativní slova: {', '.join(critical_words_found[:5])}"
        else:
            if sentiment == 'positive':
                reason = f"Text obsahuje pozitivní slova jako: {', '.join(positive_words_found[:5])}" if positive_count > 0 else "Text má celkově pozitivní tón."
            elif sentiment == 'negative':
                reason = f"Text obsahuje negativní slova jako: {', '.join(negative_words_found[:5])}" if negative_count > 0 else "Text má celkově negativní tón."
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

################################################################################
# Sentiment analysis helper
################################################################################

def analyze_sentiment(text):
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
            'sentiment_reason': "Model not loaded; skipping sentiment analysis."
        }
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
        result = sentiment_model.explain_prediction(text)
        return {
            'sentiment': result['predicted_sentiment'],
            'sentiment_score': result['sentiment_score'],
            'sentiment_features': {
                'positive_word_count': result['positive_word_count'],
                'negative_word_count': result['negative_word_count'],
                'sentiment_ratio': result['sentiment_ratio']
            },
            'positive_words': result['positive_words'],
            'negative_words': result['negative_words'],
            'sentiment_reason': result['reason']
        }
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {e}")
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
            'sentiment_reason': "Analysis failed."
        }

################################################################################
# load_data() and Routes
################################################################################

def load_data():
    global articles_df, sentiment_model, loaded_date, enhanced_models
    logger.info("Forcing load from " + FORCED_SENTIMENT_MODEL_PATH)
    if os.path.exists(FORCED_SENTIMENT_MODEL_PATH):
        try:
            loaded_model = EnhancedSentimentAnalyzer.load_model(FORCED_SENTIMENT_MODEL_PATH)
            if loaded_model:
                sentiment_model = loaded_model
                logger.info("Enhanced sentiment analyzer loaded from forced path.")
                enhanced_models = True
            else:
                logger.warning("Failed to load sentiment model from forced path, skipping.")
                sentiment_model = None
        except Exception as e:
            logger.error(f"Error loading sentiment model from forced path: {e}")
            sentiment_model = None
    else:
        logger.error(f"Sentiment analyzer model directory not found at {FORCED_SENTIMENT_MODEL_PATH}")
        sentiment_model = None

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
            articles_df[col] = None
    loaded_date = datetime.now()
    app.config['articles_df'] = articles_df
    logger.info("load_data() completed. No bulk sentiment pass here – each article is re-analyzed individually.")

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
    if articles_df is None or len(articles_df) == 0:
        return render_template('articles.html', articles=[], categories=[], sources=[], sentiments=[], page=1, total_pages=0, total_articles=0)
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
    categories = sorted(articles_df['Category'].dropna().unique().tolist())
    sources = sorted(articles_df['Source'].dropna().unique().tolist())
    sentiments = sorted(articles_df['sentiment'].dropna().unique().tolist()) if 'sentiment' in articles_df.columns else []
    return render_template('articles.html',
                           articles=articles_list,
                           categories=categories,
                           sources=sources,
                           sentiments=sentiments,
                           current_category=category,
                           current_source=source,
                           current_sentiment=sentiment,
                           current_search=search,
                           page=page,
                           total_pages=total_pages,
                           total_articles=len(filtered_df))

@app.route('/article/<int:article_id>')
def article_detail(article_id):
    """
    Updated article_detail route to use the unified AI commentary,
    tachometer gauge, and include a political view.
    """
    if articles_df is None:
        load_data()
    article = articles_df[articles_df['Id'] == article_id]
    if len(article) == 0:
        return render_template('error.html', message=f"Článek s ID {article_id} nebyl nalezen")
    article_data = article.iloc[0].to_dict()

    # Always perform fresh sentiment analysis for this article.
    content = article_data.get('Content', '')
    sentiment_results = analyze_sentiment(content)
    # Update article_data and in-memory DataFrame using .at for the specific row.
    row_index = articles_df.index[articles_df['Id'] == article_id][0]
    for key, value in sentiment_results.items():
        article_data[key] = value
        articles_df.at[row_index, key] = value

    # If no predicted_category, use scraped Category.
    if ('predicted_category' not in article_data or 
        pd.isna(article_data.get('predicted_category')) or
        str(article_data.get('predicted_category')).lower() == 'nan'):
        if 'Category' in article_data and article_data['Category']:
            article_data['predicted_category'] = article_data['Category']
        else:
            article_data['predicted_category'] = 'Zprávy'

    # Ensure sentiment_features exists.
    if 'sentiment_features' not in article_data or not isinstance(article_data.get('sentiment_features'), dict):
        article_data['sentiment_features'] = {
            'positive_word_count': article_data.get('positive_word_count', 0),
            'negative_word_count': article_data.get('negative_word_count', 0),
            'sentiment_ratio': article_data.get('sentiment_ratio', 1.0)
        }

    # Add a political view (placeholder; replace with your model output if available)
    if 'political_view' not in article_data:
        article_data['political_view'] = "Neutral"

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
    return render_template('categories.html', categories=category_counts.to_dict('records'))

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

@app.route('/api/analyze', methods=['POST'])
def analyze_text():
    data = request.json
    text = data.get('text', '')
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    results = analyze_sentiment(text)
    return jsonify({
        'text': text,
        'length': len(text),
        'word_count': len(text.split()),
        **results
    })

@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    if request.method == 'POST':
        text = request.form.get('text', '')
        result = {
            'text': text,
            'length': len(text),
            'word_count': len(text.split())
        }
        analysis = analyze_sentiment(text)
        result.update(analysis)
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

if __name__ == "__main__":
    app.run(debug=True)
