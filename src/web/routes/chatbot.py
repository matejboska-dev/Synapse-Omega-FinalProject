import json
import logging
import os
import sys
import pandas as pd
from flask import Blueprint, request, jsonify, render_template, current_app
import random

# Add parent directory to system path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Imports from our modules
from models.enhanced_category_classifier import EnhancedCategoryClassifier
from models.enhanced_sentiment_analyzer import EnhancedSentimentAnalyzer
from data.text_preprocessor import TextPreprocessor

# Logger configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create blueprint
chatbot_bp = Blueprint('chatbot', __name__)

# Initialize models
category_model = None
sentiment_model = None
text_preprocessor = None
article_stats = None

def load_models():
    """Load machine learning models"""
    global category_model, sentiment_model, text_preprocessor, article_stats
    
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        
        # Load category model
        category_model_path = os.path.join(project_root, 'models', 'enhanced_category_classifier')
        if os.path.exists(category_model_path):
            category_model = EnhancedCategoryClassifier.load_model(category_model_path)
            logger.info("Enhanced category classifier loaded successfully")
        else:
            logger.warning(f"Enhanced category model not found at {category_model_path}")
        
        # Load sentiment model
        sentiment_model_path = os.path.join(project_root, 'models', 'enhanced_sentiment_analyzer')
        if os.path.exists(sentiment_model_path):
            sentiment_model = EnhancedSentimentAnalyzer.load_model(sentiment_model_path)
            logger.info("Enhanced sentiment analyzer loaded successfully")
        else:
            logger.warning(f"Enhanced sentiment model not found at {sentiment_model_path}")
        
        # Initialize text preprocessor
        text_preprocessor = TextPreprocessor(language='czech')
        
        # Compute article statistics
        compute_article_stats()
        
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")

def compute_article_stats():
    """Compute statistics about articles for the chatbot to use"""
    global article_stats
    
    try:
        # Get application context
        if 'articles_df' in current_app.config and current_app.config['articles_df'] is not None:
            df = current_app.config['articles_df']
            
            article_stats = {
                'total_articles': len(df),
                'sources': df['Source'].nunique(),
                'categories': df['Category'].nunique(),
                'top_sources': df['Source'].value_counts().head(5).to_dict(),
                'top_categories': df['Category'].value_counts().head(5).to_dict(),
                'sentiment_distribution': df['sentiment'].value_counts().to_dict() if 'sentiment' in df.columns else {},
                'avg_article_length': int(df['ArticleLength'].mean()) if 'ArticleLength' in df.columns else 0,
                'avg_word_count': int(df['WordCount'].mean()) if 'WordCount' in df.columns else 0,
            }
            
            # Additional stats if sentiment is available
            if 'sentiment' in df.columns:
                # Sources with most negative articles
                negative_by_source = df[df['sentiment'] == 'negative'].groupby('Source').size()
                article_stats['negative_sources'] = negative_by_source.sort_values(ascending=False).head(3).to_dict()
                
                # Sources with most positive articles
                positive_by_source = df[df['sentiment'] == 'positive'].groupby('Source').size()
                article_stats['positive_sources'] = positive_by_source.sort_values(ascending=False).head(3).to_dict()
            
            logger.info("Article statistics computed successfully")
        else:
            article_stats = {"error": "No article data available"}
            logger.warning("No article data available for computing statistics")
    
    except Exception as e:
        article_stats = {"error": str(e)}
        logger.error(f"Error computing article statistics: {str(e)}")

@chatbot_bp.route('/chatbot')
def chatbot_page():
    """Render chatbot page"""
    return render_template('chatbot.html')

@chatbot_bp.route('/api/chatbot', methods=['POST'])
def chatbot_api():
    """API endpoint for chatbot interactions"""
    if category_model is None or sentiment_model is None:
        load_models()
    
    # Get message from request
    data = request.json
    message = data.get('message', '')
    
    if not message:
        return jsonify({'response': 'Prosím, napište nějakou zprávu.'})
    
    # Process the message
    response = process_message(message)
    
    return jsonify({'response': response})

@chatbot_bp.route('/api/article_chatbot', methods=['POST'])
def article_chatbot_api():
    """API endpoint for chatbot interactions about specific articles"""

    __all__ = ['chatbot_bp', 'article_chatbot_api']


    if category_model is None or sentiment_model is None:
        load_models()
    
    # Get message from request
    data = request.json
    message = data.get('message', '')
    article_id = data.get('article_id')
    
    if not message:
        return jsonify({'response': 'Prosím, napište nějakou zprávu o článku.'})
    
    if not article_id:
        return jsonify({'response': 'Chybí ID článku.'})
    
    # Get the article from database or DataFrame
    article = get_article_by_id(article_id)
    
    if article is None:
        return jsonify({'response': f'Článek s ID {article_id} nebyl nalezen.'})
    
    # Process the message in context of the article
    response = process_article_message(message, article)
    
    return jsonify({'response': response})

def get_article_by_id(article_id):
    """Get article by ID from the DataFrame"""
    try:
        # Convert to int if it's a string
        article_id = int(article_id)
        
        # Get application context
        if 'articles_df' in current_app.config and current_app.config['articles_df'] is not None:
            df = current_app.config['articles_df']
            
            # Find article by ID
            article = df[df['Id'] == article_id]
            
            if len(article) == 0:
                return None
            
            return article.iloc[0].to_dict()
        
        return None
    except Exception as e:
        logger.error(f"Error getting article by ID: {str(e)}")
        return None

def process_article_message(message, article):
    """Process user message about a specific article and generate response"""
    message_lower = message.lower()
    
    # Check for sentiment explanation question
    if any(word in message_lower for word in ['proč', 'jaký', 'jak']) and any(word in message_lower for word in ['negativní', 'pozitivní', 'neutrální', 'sentimen']):
        return explain_article_sentiment(article)
    
    # Check for category explanation
    elif any(word in message_lower for word in ['kategor', 'téma', 'zaměření']) and not any(word in message_lower for word in ['politické', 'politický', 'politika', 'politicky']):
        return explain_article_category(article)
    
    # Check for political bias or support question
    elif any(word in message_lower for word in ['politik', 'politický', 'politické', 'podporuje', 'bias', 'straní', 'stranit', 'zaměření']):
        return analyze_political_bias(article)
    
    # Check for key information question
    elif any(word in message_lower for word in ['shrň', 'shrnutí', 'klíčové', 'důležité', 'hlavní']):
        return summarize_article(article)
    
    # General article analysis
    elif any(word in message_lower for word in ['analy', 'hodnoť', 'rozbor']):
        return general_article_analysis(article)
    
    # Default response for unrecognized questions about article
    else:
        return generate_article_response(message, article)

def explain_article_sentiment(article):
    """Explain the sentiment analysis of the article"""
    # Get sentiment from article if available
    sentiment = article.get('sentiment', None)
    title = article.get('Title', 'tohoto článku')
    content = article.get('Content', '')
    
    # Pokud sentiment chybí, ale máme obsah, analyzujeme ho přímo
    if not sentiment and content:
        try:
            if sentiment_model is None:
                load_models()
                
            processed_text = text_preprocessor.preprocess_text(content)
            sentiment_id = sentiment_model.predict([processed_text])[0]
            sentiment = sentiment_model.labels[sentiment_id]
            explanation = sentiment_model.explain_prediction(processed_text)
            
            # Nyní můžeme použít získaný sentiment a vysvětlení
            response = f"<p>Článek <strong>\"{title}\"</strong> byl právě analyzován jako <strong>{sentiment}</strong>.</p>"
            
            if sentiment == 'positive':
                response += "<p>Důvody pro pozitivní klasifikaci:</p><ul>"
                if explanation['positive_words']:
                    response += f"<li>Článek obsahuje pozitivní slova: <strong>{', '.join(explanation['positive_words'])}</strong></li>"
                response += f"<li>Poměr pozitivních ku negativním slovům: <strong>{explanation['sentiment_ratio']:.2f}</strong></li>"
                response += "</ul>"
                
                response += "<p>Pozitivní články jsou často zaměřeny na úspěchy, pokroky a pozitivní změny ve společnosti, ekonomice nebo politice.</p>"
            
            elif sentiment == 'negative':
                response += "<p>Důvody pro negativní klasifikaci:</p><ul>"
                if explanation['negative_words']:
                    response += f"<li>Článek obsahuje negativní slova: <strong>{', '.join(explanation['negative_words'])}</strong></li>"
                response += f"<li>Poměr pozitivních ku negativním slovům: <strong>{explanation['sentiment_ratio']:.2f}</strong></li>"
                response += "</ul>"
                
                response += "<p>Negativní články často informují o problémech, konfliktech nebo negativních událostech. To je běžné u zpravodajských médií, která mají tendenci častěji informovat o negativních zprávách.</p>"
            
            else:  # neutral
                response += "<p>Článek byl klasifikován jako neutrální, protože:</p><ul>"
                response += f"<li>Počet pozitivních slov: <strong>{explanation['positive_word_count']}</strong></li>"
                response += f"<li>Počet negativních slov: <strong>{explanation['negative_word_count']}</strong></li>"
                response += "<li>Tyto hodnoty jsou vyrovnané nebo článek neobsahuje dostatek emočně zabarvených slov.</li>"
                response += "</ul>"
                
                response += "<p>Neutrální články často obsahují faktické informace bez výrazného emočního zabarvení, což je typické pro objektivní zpravodajství.</p>"
            
            return response
        except Exception as e:
            logger.error(f"Error in real-time sentiment analysis: {str(e)}")
            return f"Omlouvám se, ale došlo k chybě při analýze sentimentu článku: {str(e)}"
    
    # Pokud sentiment chybí a nemáme obsah
    if not sentiment:
        return "Sentiment článku nebyl analyzován a text není k dispozici pro analýzu. Zkuste prosím aktualizovat data článků."
    
    # Zbytek funkce zůstává stejný jako předtím...
    # Get sentiment features if available
    sentiment_features = article.get('sentiment_features', None)
    
    # If no sentiment features, analyze the content
    if not sentiment_features and content:
        processed_text = text_preprocessor.preprocess_text(content)
        explanation = sentiment_model.explain_prediction(processed_text)
    else:
        # Use existing sentiment features
        explanation = {
            'positive_word_count': sentiment_features.get('positive_word_count', 0) if sentiment_features else 0,
            'negative_word_count': sentiment_features.get('negative_word_count', 0) if sentiment_features else 0,
            'sentiment_ratio': sentiment_features.get('sentiment_ratio', 1.0) if sentiment_features else 1.0,
            'positive_words': sentiment_features.get('positive_words', []) if sentiment_features else [],
            'negative_words': sentiment_features.get('negative_words', []) if sentiment_features else []
        }
    
    # Create explanation response
    response = f"<p>Článek <strong>\"{title}\"</strong> byl vyhodnocen jako <strong>{sentiment}</strong>.</p>"
    
    if sentiment == 'positive':
        response += "<p>Důvody pro pozitivní klasifikaci:</p><ul>"
        if explanation.get('positive_words', []):
            response += f"<li>Článek obsahuje pozitivní slova: <strong>{', '.join(explanation.get('positive_words', []))}</strong></li>"
        response += f"<li>Poměr pozitivních ku negativním slovům: <strong>{explanation.get('sentiment_ratio', 1.0):.2f}</strong></li>"
        response += "</ul>"
        
        response += "<p>Pozitivní články jsou často zaměřeny na úspěchy, pokroky a pozitivní změny ve společnosti, ekonomice nebo politice.</p>"
    
    elif sentiment == 'negative':
        response += "<p>Důvody pro negativní klasifikaci:</p><ul>"
        if explanation.get('negative_words', []):
            response += f"<li>Článek obsahuje negativní slova: <strong>{', '.join(explanation.get('negative_words', []))}</strong></li>"
        response += f"<li>Poměr pozitivních ku negativním slovům: <strong>{explanation.get('sentiment_ratio', 1.0):.2f}</strong></li>"
        response += "</ul>"
        
        response += "<p>Negativní články často informují o problémech, konfliktech nebo negativních událostech. To je běžné u zpravodajských médií, která mají tendenci častěji informovat o negativních zprávách.</p>"
    
    else:  # neutral
        response += "<p>Článek byl klasifikován jako neutrální, protože:</p><ul>"
        response += f"<li>Počet pozitivních slov: <strong>{explanation.get('positive_word_count', 0)}</strong></li>"
        response += f"<li>Počet negativních slov: <strong>{explanation.get('negative_word_count', 0)}</strong></li>"
        response += "<li>Tyto hodnoty jsou vyrovnané nebo text neobsahuje dostatek emočně zabarvených slov pro jednoznačnou klasifikaci.</li>"
        response += "</ul>"
        
        response += "<p>Neutrální články často obsahují faktické informace bez výrazného emočního zabarvení, což je typické pro objektivní zpravodajství.</p>"
    
    return response

def explain_article_category(article):
    """Explain why article was classified to a particular category"""
    # Get predicted category if available
    predicted_category = article.get('predicted_category', None)
    original_category = article.get('Category', None)
    
    title = article.get('Title', '')
    content = article.get('Content', '')
    
    # Pokud kategorie chybí, ale máme obsah, analyzujeme ho přímo
    if not predicted_category and content:
        try:
            if category_model is None:
                load_models()
                
            processed_text = text_preprocessor.preprocess_text(content)
            predicted_category = category_model.predict([processed_text])[0]
            
            # Pokud je kategorie "Uncategorized", zkusíme zjistit téma podle klíčových slov
            if predicted_category == "Uncategorized":
                # Klíčová slova pro různé kategorie
                keywords = {
                    "Zahraniční": ["válka", "konflikt", "armáda", "útok", "hranice", "dohoda", "prezident", "premiér", "země", "stát", "mezinárodní"],
                    "Válka": ["útok", "bomba", "armáda", "vojenský", "obrana", "letectvo", "válka", "konflikt", "zbraně", "bojovat", "bitva"],
                    "Politika": ["vláda", "premiér", "parlament", "poslanec", "politik", "strana", "volby", "zákon", "novela", "ministr"],
                    "Ekonomika": ["ekonomika", "finance", "daně", "rozpočet", "inflace", "koruna", "euro", "banka", "trh", "cena", "peníze"],
                    "Sport": ["sport", "zápas", "hráč", "fotbal", "hokej", "tenis", "olympiáda", "turnaj", "mistrovství", "vítěz"],
                    "Kultura": ["film", "divadlo", "hudba", "koncert", "umění", "festival", "kniha", "výstava", "premiéra", "herec"]
                }
                
                # Počítání klíčových slov pro každou kategorii
                category_scores = {}
                for category, words in keywords.items():
                    score = sum(content.lower().count(word) for word in words)
                    category_scores[category] = score
                
                # Najít kategorii s nejvyšším skóre
                if category_scores:
                    max_category = max(category_scores.items(), key=lambda x: x[1])
                    if max_category[1] > 0:  # Pokud jsme našli alespoň jedno klíčové slovo
                        predicted_category = max_category[0]
            
            # Nyní můžeme použít získanou kategorii
            response = f"<p>Článek <strong>\"{title}\"</strong> byl právě klasifikován do kategorie <strong>{predicted_category}</strong>.</p>"
            
            # Vysvětlení kategorizace
            response += "<p>Kategorizace probíhá automaticky na základě obsahu článku pomocí těchto kroků:</p>"
            response += "<ol>"
            response += "<li>Extrakce klíčových slov a frází z textu článku</li>"
            response += "<li>Analýza četnosti a významu těchto slov v různých kategoriích</li>"
            response += "<li>Porovnání s natrénovaným modelem pro klasifikaci kategorií</li>"
            response += "</ol>"
            
            # Kategorie-specifické vysvětlení
            if predicted_category.lower() in ["válka", "konflikt", "ukrajina", "zahraniční"]:
                response += "<p>Článek obsahuje mnoho zmínek o válečných operacích, zemích a vojenských tématech, což je typické pro kategorii zahraničních zpráv o konfliktech.</p>"
            
            # Porovnání s původní kategorií, pokud existuje
            if original_category and original_category != predicted_category:
                response += f"<p><strong>Poznámka:</strong> Původní kategorie článku byla <strong>{original_category}</strong>, ale náš model ji klasifikoval jako <strong>{predicted_category}</strong>.</p>"
            
            return response
            
        except Exception as e:
            logger.error(f"Error in real-time category prediction: {str(e)}")
            return f"Omlouvám se, ale došlo k chybě při analýze kategorie článku: {str(e)}"
    
    # Pokud kategorie chybí a nemáme obsah
    if not predicted_category:
        return "Kategorie článku nebyla předpovězena a text není k dispozici pro analýzu. Zkuste prosím aktualizovat data článků."
    
    # Původní kód pro případ, že kategorie už existuje
    # Create explanation response
    response = f"<p>Článek <strong>\"{title}\"</strong> byl zařazen do kategorie <strong>{predicted_category}</strong>.</p>"
    
    # Explain the category classification
    response += "<p>Kategorizace probíhá automaticky na základě obsahu článku pomocí těchto kroků:</p>"
    response += "<ol>"
    response += "<li>Extrakce klíčových slov a frází z textu článku</li>"
    response += "<li>Analýza četnosti a významu těchto slov v různých kategoriích</li>"
    response += "<li>Porovnání s natrénovaným modelem pro klasifikaci kategorií</li>"
    response += "</ol>"
    
    # Add category-specific explanations
    if predicted_category.lower() in ["politika", "zprávy z politiky", "domácí politika"]:
        response += "<p>Článek pravděpodobně obsahuje slova a fráze typicky spojené s politikou, jako jsou jména politiků, politických stran, volby, vláda, parlament, atd.</p>"
    
    elif predicted_category.lower() in ["ekonomika", "finance", "byznys"]:
        response += "<p>Článek pravděpodobně obsahuje ekonomické termíny jako HDP, inflace, trh, akcie, banka, ekonomický růst, atd.</p>"
    
    elif predicted_category.lower() in ["sport", "sportovní zprávy"]:
        response += "<p>Článek pravděpodobně obsahuje slova spojená se sportem jako zápas, utkání, hráč, turnaj, liga, skóre, atd.</p>"
    
    elif predicted_category.lower() in ["kultura", "umění", "zábava"]:
        response += "<p>Článek pravděpodobně obsahuje slova spojená s kulturou jako film, koncert, divadlo, umění, hudba, výstava, atd.</p>"
    
    elif predicted_category.lower() in ["zahraničí", "zahraniční zprávy", "svět", "válka", "konflikt"]:
        response += "<p>Článek pravděpodobně obsahuje zmínky o zahraničních státech, městech, politicích, ozbrojených konfliktech nebo mezinárodních událostech.</p>"
    
    # Compare with original category if different
    if original_category and original_category != predicted_category:
        response += f"<p><strong>Poznámka:</strong> Původní kategorie článku byla <strong>{original_category}</strong>, ale náš model ji předpověděl jako <strong>{predicted_category}</strong>. To může být způsobeno tím, že článek obsahuje prvky obou kategorií nebo že model není dokonale natrénován.</p>"
    
    return response

def analyze_political_bias(article):
    """Analyze potential political bias in the article"""
    title = article.get('Title', '')
    content = article.get('Content', '')
    
    if not content:
        return "Nelze analyzovat politické zaměření - text článku není k dispozici. Zkuste prosím aktualizovat data článku."
    
    # Keywords associated with different political orientations in Czech context
    left_wing_terms = ['sociální', 'odbory', 'rovnost', 'spravedlnost', 'práva', 'pracující', 'solidarita', 
                       'spravedlivý', 'chudý', 'nezaměstnaný', 'socialistický', 'ksčm', 'čssd']
    
    right_wing_terms = ['trh', 'podnikání', 'ekonomický', 'svoboda', 'daně', 'snížení', 'rozpočet', 
                        'podnikatel', 'ods', 'top 09', 'konkurence', 'soukromý', 'liberální']
    
    populist_terms = ['elity', 'proti', 'slibovat', 'nepřizpůsobivý', 'migranti', 'oligarcha', 
                      'sliby', 'spd', 'okamura', 'babiš', 'ano', 'trikolóra']
    
    # Pro-Ukraine vs Pro-Russia terms for war/conflict articles
    pro_ukraine_terms = ['ukrajinská armáda', 'ukrajinci', 'ukrajinského', 'naši', 'ukrajině', 'ukrajina', 
                         'osvobození území', 'kyjev', 'ukrajinský', 'zelenskyj', 'obrana ukrajiny']
    
    pro_russia_terms = ['ruské síly', 'ruská armáda', 'osvobození', 'moskva', 'putin', 'ruské', 'rusky', 
                        'rusko', 'donbas', 'doněck', 'luhansk', 'speciální operace', 'demilitarizace']
    
    # Convert to lowercase for case-insensitive matching
    content_lower = content.lower()
    
    # Count occurrences of terms
    left_count = sum(content_lower.count(term) for term in left_wing_terms)
    right_count = sum(content_lower.count(term) for term in right_wing_terms)
    populist_count = sum(content_lower.count(term) for term in populist_terms)
    
    # Count war/conflict bias terms if relevant
    ukraine_count = sum(content_lower.count(term) for term in pro_ukraine_terms)
    russia_count = sum(content_lower.count(term) for term in pro_russia_terms)
    
    # Determine general political bias
    if left_count > right_count + 1 and left_count > populist_count:
        bias_result = "levicově orientovaný"
        bias_explanation = f"obsahuje {left_count} levicově orientovaných výrazů"
    elif right_count > left_count + 1 and right_count > populist_count:
        bias_result = "pravicově orientovaný"
        bias_explanation = f"obsahuje {right_count} pravicově orientovaných výrazů"
    elif populist_count > left_count and populist_count > right_count:
        bias_result = "populisticky orientovaný"
        bias_explanation = f"obsahuje {populist_count} populisticky orientovaných výrazů"
    else:
        bias_result = "politicky neutrální"
        bias_explanation = f"neobsahuje významný počet politicky zabarvených výrazů (levice: {left_count}, pravice: {right_count}, populismus: {populist_count})"
    
    # Check for war/conflict bias if relevant
    conflict_bias = ""
    if ukraine_count > 0 or russia_count > 0:
        if ukraine_count > russia_count + 1:
            conflict_bias = f" a jeví se jako <strong>nakloněný ukrajinské straně</strong> konfliktu (zmínky Ukrajina: {ukraine_count}, Rusko: {russia_count})"
        elif russia_count > ukraine_count + 1:
            conflict_bias = f" a jeví se jako <strong>nakloněný ruské straně</strong> konfliktu (zmínky Rusko: {russia_count}, Ukrajina: {ukraine_count})"
        else:
            conflict_bias = f" a pokrývá konflikt <strong>relativně neutrálně</strong> (zmínky Ukrajina: {ukraine_count}, Rusko: {russia_count})"
    
    # Create response
    response = f"<p>Článek <strong>\"{title}\"</strong> se jeví jako <strong>{bias_result}</strong>, protože {bias_explanation}{conflict_bias}.</p>"
    
    # Add disclaimer
    response += "<p><strong>Poznámka:</strong> Tato analýza je založena na jednoduchém počítání klíčových slov a poskytuje pouze orientační pohled. Pro skutečně kvalitní analýzu politické orientace textu by byl potřeba pokročilejší lingvistický rozbor a kontext.</p>"
    
    # Add source information if available
    source = article.get('Source', None)
    if source:
        response += f"<p>Zdroj článku <strong>{source}</strong> může také ovlivňovat jeho politickou orientaci.</p>"
    
    return response

def summarize_article(article):
    """Provide a summary of the article"""
    title = article.get('Title', '')
    content = article.get('Content', '')
    source = article.get('Source', 'neznámého zdroje')
    publish_date = article.get('PublishDate', '')
    
    # Very simple summarization - take first few sentences
    # In a real app, you would use a more sophisticated summarization algorithm
    sentences = content.split('.')
    summary_sentences = sentences[:3]  # First 3 sentences
    summary = '. '.join(summary_sentences) + '.'
    
    response = f"<p><strong>Shrnutí článku \"{title}\"</strong></p>"
    response += f"<p>Článek byl publikován {publish_date} na {source}.</p>"
    response += f"<p>{summary}</p>"
    
    # Add topic information if available
    category = article.get('Category', article.get('predicted_category', None))
    if category:
        response += f"<p>Hlavní téma článku: <strong>{category}</strong></p>"
    
    # Add sentiment information if available
    sentiment = article.get('sentiment', None)
    if sentiment:
        response += f"<p>Tón článku: <strong>{sentiment}</strong></p>"
    
    return response

def general_article_analysis(article):
    """Provide a general analysis of the article"""
    title = article.get('Title', '')
    source = article.get('Source', 'neznámý')
    publish_date = article.get('PublishDate', '')
    word_count = article.get('WordCount', 0)
    length = article.get('ArticleLength', 0)
    category = article.get('Category', article.get('predicted_category', 'neznámá'))
    sentiment = article.get('sentiment', 'neutrální')
    
    # Analyze content stats
    avg_word_length = length / max(word_count, 1)
    
    # Compare to average article stats
    avg_length = article_stats.get('avg_article_length', 0) if article_stats else 0
    avg_words = article_stats.get('avg_word_count', 0) if article_stats else 0
    
    length_comparison = "průměrné délky" if abs(length - avg_length) < avg_length * 0.2 else (
        "nadprůměrně dlouhý" if length > avg_length else "podprůměrně krátký"
    )
    
    # Create analysis response
    response = f"<p><strong>Analýza článku \"{title}\"</strong></p>"
    response += "<ul>"
    response += f"<li><strong>Zdroj:</strong> {source}</li>"
    response += f"<li><strong>Datum publikace:</strong> {publish_date}</li>"
    response += f"<li><strong>Kategorie:</strong> {category}</li>"
    response += f"<li><strong>Sentiment:</strong> {sentiment}</li>"
    response += f"<li><strong>Počet slov:</strong> {word_count} (článek je {length_comparison})</li>"
    response += f"<li><strong>Průměrná délka slova:</strong> {avg_word_length:.1f} znaků</li>"
    response += "</ul>"
    
    # Add source analysis if we have stats
    if article_stats and 'top_sources' in article_stats:
        if source in article_stats['top_sources']:
            response += f"<p>Zdroj {source} patří mezi 5 nejčastějších zdrojů v naší databázi.</p>"
    
    # Add sentiment distribution context if available
    if article_stats and 'sentiment_distribution' in article_stats and sentiment in article_stats['sentiment_distribution']:
        total = sum(article_stats['sentiment_distribution'].values())
        percent = article_stats['sentiment_distribution'].get(sentiment, 0) / total * 100 if total > 0 else 0
        response += f"<p>Články s {sentiment} sentimentem tvoří přibližně {percent:.1f}% všech článků v naší databázi.</p>"
    
    return response

def generate_article_response(message, article):
    """Generate a general response about the article for unrecognized questions"""
    title = article.get('Title', '')
    
    # List of possible responses
    responses = [
        f"<p>Nejsem si jistý, na co přesně se ptáte ohledně článku \"{title}\". Můžete se zeptat na:</p><ul><li>Proč má článek pozitivní/negativní sentiment?</li><li>Proč byl článek zařazen do určité kategorie?</li><li>Jaké je politické zaměření článku?</li><li>Shrnutí hlavních bodů článku</li></ul>",
        
        f"<p>Omlouvám se, ale nerozumím přesně vaší otázce o článku \"{title}\". Zkuste se zeptat konkrétněji, například na sentiment článku, jeho kategorii nebo politické zaměření.</p>",
        
        f"<p>Pro článek \"{title}\" mohu poskytnout analýzu sentimentu, kategorie, politického zaměření nebo shrnout hlavní body. Na co konkrétně se chcete zeptat?</p>"
    ]
    
    return random.choice(responses)

def process_message(message):
    """
    Process user message and generate response
    """
    message_lower = message.lower()
    
    # Check for sentiment explanation question
    if 'proč' in message_lower and ('negativní' in message_lower or 'pozitivní' in message_lower or 'neutrální' in message_lower):
        return explain_sentiment_classification(message)
    
    # Check for category explanation
    elif 'kategor' in message_lower and ('jak' in message_lower or 'proč' in message_lower):
        return explain_category_classification()
    
    # Check for sentiment algorithm question
    elif 'jak' in message_lower and 'sentiment' in message_lower:
        return explain_sentiment_algorithm()
    
    # Check for negative sources question
    elif 'negativ' in message_lower and 'zdroj' in message_lower:
        return get_negative_sources()
    
    # Check for positive sources question
    elif 'pozitiv' in message_lower and 'zdroj' in message_lower:
        return get_positive_sources()
    
    # Check for stats question
    elif 'statisti' in message_lower or 'kolik' in message_lower:
        return get_article_statistics(message)
    
    # Check if user wants to analyze text
    elif 'analyz' in message_lower or 'klasifik' in message_lower:
        return analyze_text(message)
    
    # Default response for unrecognized questions
    else:
        return generate_default_response(message)

def explain_sentiment_classification(message):
    """Explain why a text was classified with a particular sentiment"""
    try:
        # Extract the actual text to analyze
        if 'proč je' in message.lower():
            text_start = message.lower().find('proč je') + 7
            text = message[text_start:].strip()
            
            # Preprocess text
            processed_text = text_preprocessor.preprocess_text(text)
            
            # Get explanation
            explanation = sentiment_model.explain_prediction(processed_text)
            
            response = f"<p>Analyzoval jsem text: <em>\"{text}\"</em></p>"
            response += f"<p>Sentiment textu jsem vyhodnotil jako <strong>{explanation['predicted_sentiment']}</strong>.</p>"
            
            if explanation['predicted_sentiment'] == 'positive':
                response += "<p>Důvody pro pozitivní klasifikaci:</p><ul>"
                if explanation['positive_words']:
                    response += f"<li>Nalezl jsem pozitivní slova: <strong>{', '.join(explanation['positive_words'])}</strong></li>"
                response += f"<li>Poměr pozitivních ku negativním slovům: <strong>{explanation['sentiment_ratio']:.2f}</strong></li>"
                response += "</ul>"
            
            elif explanation['predicted_sentiment'] == 'negative':
                response += "<p>Důvody pro negativní klasifikaci:</p><ul>"
                if explanation['negative_words']:
                    response += f"<li>Nalezl jsem negativní slova: <strong>{', '.join(explanation['negative_words'])}</strong></li>"
                response += f"<li>Poměr pozitivních ku negativním slovům: <strong>{explanation['sentiment_ratio']:.2f}</strong></li>"
                response += "</ul>"
            
            else:
                response += "<p>Text byl klasifikován jako neutrální, protože:</p><ul>"
                response += f"<li>Počet pozitivních slov: <strong>{explanation['positive_word_count']}</strong></li>"
                response += f"<li>Počet negativních slov: <strong>{explanation['negative_word_count']}</strong></li>"
                response += f"<li>Tyto hodnoty jsou vyrovnané nebo příliš nízké pro jednoznačnou klasifikaci.</li>"
                response += "</ul>"
            
            return response
        else:
            return "Prosím, uveďte konkrétní text, který chcete analyzovat. Například: \"Proč je tento článek negativní?\" nebo \"Proč je zpráva o zvýšení daní negativní?\""
    
    except Exception as e:
        logger.error(f"Error explaining sentiment: {str(e)}")
        return "Omlouvám se, ale při analýze sentimentu došlo k chybě. Zkuste to prosím znovu s jiným textem."

def explain_category_classification():
    """Explain how category classification works"""
    response = """
    <p>Klasifikace kategorií článků funguje na základě strojového učení s využitím několika technik:</p>
    <ol>
        <li><strong>Předzpracování textu</strong> - odstraňuji diakritiku, stopslova (jako "a", "ale", "je") a převádím slova na základní tvary</li>
        <li><strong>Extrakce příznaků</strong> - vytvářím vektory příznaků z textu pomocí:
            <ul>
                <li>TF-IDF pro slova (jak důležité jsou slova v článku)</li>
                <li>N-gramy znaků (zachycují části slov a jejich kombinace)</li>
            </ul>
        </li>
        <li><strong>Trénování klasifikátoru</strong> - používám algoritmus, který se naučil vzorce z tisíců článků s již přiřazenými kategoriemi</li>
    </ol>
    
    <p>Při klasifikaci nového článku proces zahrnuje:</p>
    <ol>
        <li>Předzpracování textu článku stejným způsobem</li>
        <li>Extrakci stejných typů příznaků</li>
        <li>Použití natrénovaného modelu pro predikci nejpravděpodobnější kategorie</li>
    </ol>
    
    <p>Kategorie jsou určeny na základě specifických slov a frází, které model identifikoval jako charakteristické pro danou kategorii. Například články o sportu často obsahují slova jako "zápas", "hráč", "skóre", zatímco články o ekonomice typicky zmiňují "inflace", "trh", "investice" apod.</p>
    """
    return response

def explain_sentiment_algorithm():
    """Explain how sentiment analysis works"""
    response = """
    <p>Analýza sentimentu článků probíhá v několika krocích:</p>
    
    <ol>
        <li><strong>Lexikální analýza</strong> - používám rozsáhlý slovník pozitivních a negativních slov v češtině</li>
        <li><strong>Komplexní předzpracování textu</strong> - text očistím od diakritiky, odstraním stopslova a převedu na základní tvary</li>
        <li><strong>Extrakce příznaků</strong> - analyzuji:
            <ul>
                <li>Počet pozitivních a negativních slov</li>
                <li>Poměr pozitivních ku negativním slovům</li>
                <li>Speciální příznaky jako vykřičníky, otazníky, slova psaná velkými písmeny</li>
                <li>TF-IDF příznaky slov i znaků</li>
            </ul>
        </li>
        <li><strong>Klasifikace</strong> - kombinuji všechny tyto příznaky v modelu, který byl natrénován na velkém množství článků</li>
    </ol>
    
    <p>Výsledky klasifikace jsou ve třech kategoriích:</p>
    <ul>
        <li><strong>Pozitivní</strong> - článek obsahuje převážně pozitivní slova a fráze</li>
        <li><strong>Neutrální</strong> - článek je vyvážený nebo neobsahuje dostatek emočně zabarvených slov</li>
        <li><strong>Negativní</strong> - článek obsahuje převážně negativní slova a fráze</li>
    </ul>
    
    <p>Tento přístup dosahuje přesnosti přes 90% na testovacích datech, což je významné zlepšení oproti předchozím modelům.</p>
    """
    return response

def get_negative_sources():
    """Get information about sources with most negative articles"""
    if article_stats and 'negative_sources' in article_stats:
        response = "<p>Zdroje s nejvyšším počtem negativních článků:</p><ul>"
        
        for source, count in article_stats['negative_sources'].items():
            response += f"<li><strong>{source}</strong>: {count} negativních článků</li>"
        
        response += "</ul>"
        
        response += "<p>Je důležité poznamenat, že toto nemusí nutně znamenat, že tyto zdroje jsou více pesimistické - může to být ovlivněno i tím, o jakých tématech častěji informují.</p>"
        
        return response
    else:
        return "Omlouvám se, ale nemám dostupná data o negativních zdrojích. Je možné, že analýza sentimentu nebyla provedena nebo nemáme dostatek článků."

def get_positive_sources():
    """Get information about sources with most positive articles"""
    if article_stats and 'positive_sources' in article_stats:
        response = "<p>Zdroje s nejvyšším počtem pozitivních článků:</p><ul>"
        
        for source, count in article_stats['positive_sources'].items():
            response += f"<li><strong>{source}</strong>: {count} pozitivních článků</li>"
        
        response += "</ul>"
        
        response += "<p>Je dobré vědět, že tyto zdroje častěji publikují pozitivní zprávy, ale pamatujte, že to může souviset i s tématy, kterým se věnují.</p>"
        
        return response
    else:
        return "Omlouvám se, ale nemám dostupná data o pozitivních zdrojích. Je možné, že analýza sentimentu nebyla provedena nebo nemáme dostatek článků."

def get_article_statistics(message):
    """Get statistics about articles"""
    if not article_stats or 'error' in article_stats:
        return "Omlouvám se, ale nemám dostupná statistická data o článcích."
    
    message_lower = message.lower()
    
    # Check what kind of statistics the user wants
    if 'zdroj' in message_lower:
        response = "<p>Statistiky o zdrojích:</p><ul>"
        response += f"<li>Celkový počet zdrojů: <strong>{article_stats['sources']}</strong></li>"
        
        response += "<li>Nejčastější zdroje:</li><ul>"
        for source, count in article_stats['top_sources'].items():
            response += f"<li><strong>{source}</strong>: {count} článků</li>"
        response += "</ul>"
        
        response += "</ul>"
        
        return response
    
    elif 'kategor' in message_lower:
        response = "<p>Statistiky o kategoriích:</p><ul>"
        response += f"<li>Celkový počet kategorií: <strong>{article_stats['categories']}</strong></li>"
        
        response += "<li>Nejčastější kategorie:</li><ul>"
        for category, count in article_stats['top_categories'].items():
            response += f"<li><strong>{category}</strong>: {count} článků</li>"
        response += "</ul>"
        
        response += "</ul>"
        
        return response
    
    elif 'sentiment' in message_lower:
        if 'sentiment_distribution' in article_stats and article_stats['sentiment_distribution']:
            response = "<p>Distribuce sentimentu v článcích:</p><ul>"
            
            for sentiment, count in article_stats['sentiment_distribution'].items():
                percentage = count / article_stats['total_articles'] * 100
                response += f"<li><strong>{sentiment}</strong>: {count} článků ({percentage:.1f}%)</li>"
            
            response += "</ul>"
            
            return response
        else:
            return "Omlouvám se, ale nemám dostupná data o distribuci sentimentu článků."
    
    # Default to general statistics
    response = "<p>Obecné statistiky o článcích:</p><ul>"
    response += f"<li>Celkový počet článků: <strong>{article_stats['total_articles']}</strong></li>"
    response += f"<li>Počet zdrojů: <strong>{article_stats['sources']}</strong></li>"
    response += f"<li>Počet kategorií: <strong>{article_stats['categories']}</strong></li>"
    response += f"<li>Průměrná délka článku: <strong>{article_stats['avg_article_length']}</strong> znaků</li>"
    response += f"<li>Průměrný počet slov: <strong>{article_stats['avg_word_count']}</strong></li>"
    
    if 'sentiment_distribution' in article_stats and article_stats['sentiment_distribution']:
        response += "<li>Distribuce sentimentu:</li><ul>"
        for sentiment, count in article_stats['sentiment_distribution'].items():
            percentage = count / article_stats['total_articles'] * 100
            response += f"<li><strong>{sentiment}</strong>: {percentage:.1f}%</li>"
        response += "</ul>"
    
    response += "</ul>"
    
    return response

def analyze_text(message):
    """Analyze text provided by the user"""
    # Try to extract text to analyze
    text_to_analyze = ""
    
    if "analyzuj" in message.lower() or "klasifikuj" in message.lower():
        parts = message.split('"')
        if len(parts) >= 3:  # Text is enclosed in quotes
            text_to_analyze = parts[1]
        else:
            # Check if there's text after the keywords
            for keyword in ["analyzuj", "klasifikuj", "analýza", "klasifikace"]:
                if keyword in message.lower():
                    idx = message.lower().find(keyword) + len(keyword)
                    text_to_analyze = message[idx:].strip()
                    if text_to_analyze and text_to_analyze[0] in [',', ':', ' ']:
                        text_to_analyze = text_to_analyze[1:].strip()
                    break
    
    if not text_to_analyze:
        return "Abych mohl analyzovat text, prosím, napište jej ve vašem dotazu. Například: \"Analyzuj text: Vláda schválila nový zákon o daních.\" nebo \"Klasifikuj: Fotbalisté Sparty postoupili do finále.\""
    
    # Preprocess text
    processed_text = text_preprocessor.preprocess_text(text_to_analyze)
    
    # Get category prediction
    predicted_category = category_model.predict([processed_text])[0]
    
    # Get sentiment prediction
    sentiment_id = sentiment_model.predict([processed_text])[0]
    sentiment = sentiment_model.labels[sentiment_id]
    
    # Get sentiment explanation
    explanation = sentiment_model.explain_prediction(processed_text)
    
    # Create response
    response = f"<p>Analýza textu: <em>\"{text_to_analyze}\"</em></p>"
    
    response += "<p><strong>Výsledky:</strong></p>"
    response += f"<p>Kategorie: <strong>{predicted_category}</strong></p>"
    response += f"<p>Sentiment: <strong>{sentiment}</strong></p>"
    
    # Add explanation
    response += "<p><strong>Vysvětlení sentimentu:</strong></p>"
    
    if sentiment == 'positive':
        response += "<ul>"
        if explanation['positive_words']:
            response += f"<li>Text obsahuje pozitivní slova: <strong>{', '.join(explanation['positive_words'])}</strong></li>"
        response += f"<li>Poměr pozitivních ku negativním slovům: <strong>{explanation['sentiment_ratio']:.2f}</strong></li>"
        response += "</ul>"
    
    elif sentiment == 'negative':
        response += "<ul>"
        if explanation['negative_words']:
            response += f"<li>Text obsahuje negativní slova: <strong>{', '.join(explanation['negative_words'])}</strong></li>"
        response += f"<li>Poměr pozitivních ku negativním slovům: <strong>{explanation['sentiment_ratio']:.2f}</strong></li>"
        response += "</ul>"
    
    else:  # neutral
        response += "<ul>"
        response += f"<li>Počet pozitivních slov: <strong>{explanation['positive_word_count']}</strong></li>"
        response += f"<li>Počet negativních slov: <strong>{explanation['negative_word_count']}</strong></li>"
        response += "<li>Poměr je vyrovnaný nebo text neobsahuje dostatek emočně zabarvených slov pro jednoznačnou klasifikaci.</li>"
        response += "</ul>"
    
    return response

def generate_default_response(message):
    """Generate default response for unrecognized questions"""
    default_responses = [
        "Můžete se mě zeptat na analýzu článků, vysvětlení sentimentu nebo kategorií, a také na statistiky o zpravodajských zdrojích.",
        "Nejsem si jistý, co máte na mysli. Zkuste se mě zeptat na analýzu textu, vysvětlení klasifikace nebo statistiky o článcích.",
        "Specializuji se na analýzu zpravodajských článků. Mohu vám pomoci s analýzou textu, vysvětlením sentimentu nebo informacemi o zdrojích.",
        "Nepochopil jsem váš dotaz. Mohu vám pomoci s analýzou sentimentu textu, kategorií článků nebo statistikami zdrojů."
    ]
    
    return random.choice(default_responses)

__all__ = ['chatbot_bp', 'article_chatbot_api']

# Register blueprint in app.py with:
# from routes.chatbot import chatbot_bp, article_chatbot_api
# app.register_blueprint(chatbot_bp)
# app.add_url_rule('/api/article_chatbot', view_func=article_chatbot_api, methods=['POST'])