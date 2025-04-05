import os
import sys
import logging
import pandas as pd
import numpy as np
import json
import pickle
from datetime import datetime
import pyodbc
import re
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset

# Fix encoding issues
try:
    sys.stdout.reconfigure(encoding='utf-8')
except:  # Fallback for older Python versions
    pass

# Add path to project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(project_root)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(project_root, 'logs', f"train_sentiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"), encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("sentiment_analyzer")

# Database connection parameters
DB_SERVER = "193.85.203.188"
DB_NAME = "boska"
DB_USER = "boska"
DB_PASSWORD = "123456"

def connect_to_database():
    """Connect to SQL Server database"""
    try:
        # Try to load config from file
        config_path = os.path.join(project_root, 'config', 'database.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                server = config.get('server', DB_SERVER)
                database = config.get('database', DB_NAME)
                username = config.get('username', DB_USER)
                password = config.get('password', DB_PASSWORD)
            logger.info("Using database configuration from config file")
        else:
            server = DB_SERVER
            database = DB_NAME
            username = DB_USER
            password = DB_PASSWORD
        
        # Connect
        conn_str = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}"
        conn = pyodbc.connect(conn_str)
        logger.info(f"Connected to database {database} on {server}")
        return conn
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        # Try to load alternative data
        logger.info("Will try to load data from local files instead")
        return None

def load_articles_from_db(conn, limit=None):
    """
    Load articles from database
    
    Args:
        conn: Database connection
        limit: Maximum number of articles to load
    
    Returns:
        pd.DataFrame: DataFrame with articles
    """
    try:
        # Query to get articles with text, title and category
        query = """
        SELECT 
            ArticleText, 
            Title, 
            Category, 
            SourceName, 
            PublicationDate
        FROM Articles
        WHERE ArticleText IS NOT NULL 
          AND LEN(ArticleText) > 100
        ORDER BY PublicationDate DESC
        """
        
        if limit:
            query = f"SELECT TOP {limit} " + query.split("SELECT ", 1)[1]
        
        df = pd.read_sql(query, conn)
        logger.info(f"Loaded {len(df)} articles from database")
        return df
    except Exception as e:
        logger.error(f"Error loading articles from database: {e}")
        return None

def load_local_articles():
    """
    Load articles from local files if database connection fails
    
    Returns:
        pd.DataFrame: DataFrame with articles
    """
    # Look for data files in project
    data_paths = [
        os.path.join(project_root, 'data', 'display', 'all_articles.json'),
        os.path.join(project_root, 'data', 'processed', 'articles_processed.json'),
        os.path.join(project_root, 'data', 'processed_scraped')
    ]
    
    # Try each path
    for path in data_paths:
        try:
            if os.path.isfile(path) and path.endswith('.json'):
                # Load single file
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                df = pd.DataFrame(data)
                logger.info(f"Loaded {len(df)} articles from {path}")
                return df
            elif os.path.isdir(path):
                # Try to find json files in directory
                json_files = [f for f in os.listdir(path) if f.endswith('.json')]
                if json_files:
                    newest_file = sorted(json_files)[-1]  # Get newest file
                    with open(os.path.join(path, newest_file), 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    df = pd.DataFrame(data)
                    logger.info(f"Loaded {len(df)} articles from {os.path.join(path, newest_file)}")
                    return df
        except Exception as e:
            logger.warning(f"Failed to load from {path}: {e}")
    
    logger.error("Could not load articles from any local path")
    return None

def preprocess_articles(df):
    """
    Preprocess articles for training
    
    Args:
        df: DataFrame with articles
    
    Returns:
        pd.DataFrame: Preprocessed DataFrame
    """
    # Check and rename columns if needed
    column_mapping = {
        'ArticleText': 'text',
        'Content': 'text',
        'SourceName': 'source',
        'Source': 'source'
    }
    
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns and new_col not in df.columns:
            df[new_col] = df[old_col]
    
    # Combine title and content for better analysis
    if 'Title' in df.columns and 'text' in df.columns:
        df['text'] = df['Title'] + ' ' + df['text']
    
    # Filter out rows with empty text
    df = df[df['text'].notna() & (df['text'].str.len() > 50)]
    logger.info(f"After filtering, {len(df)} articles remain")
    
    return df

def generate_czech_sentiment_lexicons():
    """Generate Czech sentiment lexicons for positive and negative words"""
    logger.info("Generating Czech sentiment lexicons")
    
    # Define basic Czech positive words
    positive_words = [
        "dobrý", "skvělý", "výborný", "pozitivní", "úspěch", "radost", "krásný", "příjemný",
        "štěstí", "spokojený", "výhra", "zisk", "růst", "lepší", "nejlepší", "zlepšení",
        "výhoda", "prospěch", "podpora", "rozvoj", "pokrok", "úspěšný", "optimistický",
        "šťastný", "veselý", "bezpečný", "klidný", "prospěšný", "úžasný", "perfektní",
        "vynikající", "senzační", "fantastický", "neuvěřitelný", "báječný", "nádherný",
        "velkolepý", "luxusní", "přátelský", "laskavý", "milý", "ochotný", "talentovaný",
        "nadaný", "inovativní", "kreativní", "silný", "výkonný", "efektivní", "užitečný",
        "cenný", "důležitý", "ohromující", "fascinující", "zajímavý", "pozoruhodný",
        "inspirativní", "motivující", "povzbuzující", "osvěžující", "uvolňující",
        "uklidňující", "příznivý", "konstruktivní", "produktivní", "perspektivní",
        "slibný", "nadějný", "obohacující", "vzrušující", "úchvatný", "impozantní",
        "působivý", "přesvědčivý", "vítaný", "populární", "oblíbený", "milovaný",
        "oceňovaný", "oslavovaný", "vyzdvihovaný", "vyžadovaný", "potřebný", "žádoucí",
        "velmi", "skvěle", "nadšení", "nadšený", "radostný", "vylepšený", "přelomový",
        "úžasně", "nadmíru", "mimořádně", "výjimečně", "srdečně", "ideální", "dobře",
        "pomoc", "pomáhat", "pomohl", "pomohli", "získat", "získal", "získali", "podpořit",
        "podpořil", "podpořili", "zlepšit", "zlepšil", "zlepšili", "ocenit", "ocenil",
        "ocenili", "zajistit", "zajistil", "zajistili", "zvýšit", "zvýšil", "zvýšili",
        "přínos", "přínosný", "výborně", "výtečně", "vylepšení", "zdokonalení"
    ]
    
    # Define basic Czech negative words
    negative_words = [
        "špatný", "negativní", "problém", "potíž", "selhání", "prohra", "ztráta", "pokles",
        "krize", "konflikt", "smrt", "válka", "nehoda", "tragédie", "nebezpečí", "zhoršení",
        "škoda", "nízký", "horší", "nejhorší", "slabý", "nepříznivý", "riziko", "hrozba",
        "kritický", "závažný", "obtížný", "těžký", "násilí", "strach", "obavy", "útok",
        "katastrofa", "pohroma", "neštěstí", "destrukce", "zničení", "zkáza", "porážka",
        "kolaps", "pád", "děsivý", "hrozný", "strašný", "příšerný", "otřesný", "hrozivý",
        "znepokojivý", "alarmující", "ohavný", "odpudivý", "nechutný", "odporný", "krutý",
        "brutální", "agresivní", "surový", "barbarský", "divoký", "vražedný", "smrtící",
        "jedovatý", "toxický", "škodlivý", "ničivý", "zničující", "fatální", "smrtelný",
        "zoufalý", "beznadějný", "bezmocný", "deprimující", "skličující", "depresivní",
        "smutný", "bolestný", "trýznivý", "traumatický", "poškozený", "rozbitý", "zlomený",
        "naštvaný", "rozzlobený", "rozzuřený", "rozhořčený", "nenávistný", "nepřátelský",
        "odmítavý", "podvodný", "klamavý", "lživý", "falešný", "neetický", "nemorální",
        "zkorumpovaný", "zkažený", "prohnilý", "bezcenný", "zbytečný", "marný", "bídný",
        "ubohý", "žalostný", "nedostatečný", "průměrný", "nudný", "nezajímavý", "nezáživný",
        "bohužel", "žel", "naneštěstí", "nešťastný", "narušený", "znechucený", "zraněný",
        "zraněno", "utrpení", "trápení", "vážné", "vážně", "kriticky", "drasticky",
        "hrozně", "selhal", "selhala", "nepovedlo", "nefunguje", "chyba", "nefunkční",
        "rozpadlý", "zhroutil", "zhroutila", "zničil", "zničila", "zaútočil", "zaútočila",
        "zabít", "zabil", "zabila", "zemřít", "zemřel", "zemřela", "upadl", "upadla",
        "obává", "obával", "obávali", "přestal", "přestala", "přestali", "zbankrotoval",
        "zbankrotovala", "nemoc", "nemocný", "dluh", "dluhy", "zadlužený", "nezaměstnaný",
        "nezaměstnanost", "chudoba", "chudobný", "omezení", "omezil", "omezili"
    ]
    
    # Save lexicons
    data_dir = os.path.join(project_root, 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    with open(os.path.join(data_dir, 'positive_words.txt'), 'w', encoding='utf-8') as f:
        for word in positive_words:
            f.write(f"{word}\n")
    
    with open(os.path.join(data_dir, 'negative_words.txt'), 'w', encoding='utf-8') as f:
        for word in negative_words:
            f.write(f"{word}\n")
    
    logger.info(f"Czech sentiment lexicons created: {len(positive_words)} positive, {len(negative_words)} negative words")
    return positive_words, negative_words

class EnhancedSentimentAnalyzer:
    """
    Enhanced Sentiment Analyzer for Czech news articles
    Implementation inspired by CZE-NEC paper (https://arxiv.org/abs/2307.10666)
    """
    
    def __init__(self, max_features=15000):
        """
        Initialize the sentiment analyzer
        
        Args:
            max_features (int): Maximum number of features for TF-IDF
        """
        self.max_features = max_features
        self.model = None
        self.pipeline = None
        self.labels = ['negative', 'neutral', 'positive']
        
        # Load lexicons
        self.positive_words = self.load_words('positive_words.txt')
        self.negative_words = self.load_words('negative_words.txt')
        self.critical_negative_words = self.get_critical_negative_words()
    
    def get_critical_negative_words(self):
        """Get a subset of highly negative words for special handling"""
        critical_words = {
            'smrt', 'úmrtí', 'zemřel', 'zemřela', 'zahynul', 'mrtvý', 'tragick', 'tragédie', 
            'katastrofa', 'neštěstí', 'oběť', 'obětí', 'nehoda', 'smrtelný', 'zraněn', 
            'vyžádala', 'zahynul', 'usmrcen', 'zabil', 'zabit', 'utonul', 'zastřelen',
            'válka', 'útok', 'konflikt', 'krize', 'hrozba', 'nebezpečí', 'násilí',
            'zbankrotoval', 'zadlužený', 'chudoba', 'nezaměstnanost', 'kolaps'
        }
        return critical_words
    
    def load_words(self, filename):
        """
        Load word list from file
        
        Args:
            filename (str): Name of the file containing words
        
        Returns:
            list: List of words
        """
        try:
            word_path = os.path.join(project_root, 'data', filename)
            if os.path.exists(word_path):
                with open(word_path, 'r', encoding='utf-8') as f:
                    return [line.strip() for line in f if line.strip()]
            return []
        except:
            logger.warning(f"Could not load {filename}, using empty list")
            return []
    
    def build_model(self):
        """
        Build a RandomForest model with TF-IDF features
        Enhanced with CZE-NEC insights for Czech sentiment analysis
        """
        # Create a pipeline with TF-IDF and RandomForest with more trees for better accuracy
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=self.max_features, 
                                     ngram_range=(1, 3),  # Include bigrams and trigrams for better context
                                     min_df=2)),
            ('clf', RandomForestClassifier(n_estimators=150, 
                                          min_samples_leaf=2,
                                          random_state=42, 
                                          n_jobs=-1,
                                          class_weight='balanced'))  # Balance class weights
        ])
        
        return self.pipeline
    
    def extract_sentiment_features(self, texts):
        """
        Extract sentiment features from texts
        
        Args:
            texts (list): List of texts
            
        Returns:
            pd.DataFrame: DataFrame with sentiment features
        """
        features = pd.DataFrame()
        
        # Count positive and negative words
        features['positive_word_count'] = [
            sum(1 for word in text.lower().split() if word in self.positive_words) 
            for text in tqdm(texts, desc="Counting positive words")
        ]
        
        features['negative_word_count'] = [
            sum(1 for word in text.lower().split() if word in self.negative_words)
            for text in tqdm(texts, desc="Counting negative words")
        ]
        
        # Count critical negative words (accidents, deaths, etc.)
        features['critical_negative_count'] = [
            sum(1 for word in text.lower().split() if any(crit_word in word for crit_word in self.critical_negative_words))
            for text in tqdm(texts, desc="Counting critical negative words")
        ]
        
        # Calculate sentiment ratio
        features['sentiment_ratio'] = (features['positive_word_count'] + 1) / (features['negative_word_count'] + 1)
        
        # Add text length features
        features['text_length'] = [len(text) for text in texts]
        features['word_count'] = [len(text.split()) for text in texts]
        
        return features
    
    def auto_label_data(self, texts):
        """
        Automatically generate sentiment labels based on word analysis with improved logic
        Inspired by CZE-NEC approach
        
        Args:
            texts (list): List of texts
            
        Returns:
            list: Sentiment labels (0=negative, 1=neutral, 2=positive)
        """
        features = self.extract_sentiment_features(texts)
        
        # Determine sentiment with improved rules for Czech language
        sentiments = []
        for i, row in features.iterrows():
            pos_count = row['positive_word_count']
            neg_count = row['negative_word_count']
            critical_count = row['critical_negative_count']
            ratio = row['sentiment_ratio']
            
            # If any critical negative words are present, classify as negative
            if critical_count > 0:
                sentiments.append(0)  # negative
            # Strong positive sentiment
            elif ratio > 1.5 or (pos_count >= 3 and neg_count == 0):
                sentiments.append(2)  # positive
            # Strong negative sentiment (adjusted threshold)
            elif ratio < 0.8 or (neg_count >= 2 and pos_count == 0):
                sentiments.append(0)  # negative
            # Neutral
            else:
                sentiments.append(1)  # neutral
                
        # Log distribution
        counts = np.bincount(sentiments, minlength=3)
        logger.info(f"Auto-labeled data: Negative={counts[0]}, Neutral={counts[1]}, Positive={counts[2]}")
        
        return sentiments
    
    def fit(self, texts, labels, validation_split=0.1):
        """
        Train the sentiment analyzer
        
        Args:
            texts (list): List of texts
            labels (list): List of sentiment labels
            validation_split (float): Proportion of data for validation
            
        Returns:
            dict: Training results
        """
        # Ensure labels are numpy array
        labels = np.array(labels)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            texts, labels, 
            test_size=validation_split,
            random_state=42,
            stratify=labels
        )
        
        # Build model if not already built
        if self.pipeline is None:
            self.build_model()
            logger.info("Model pipeline created")
        
        # Train model
        logger.info(f"Training model on {len(X_train)} samples, validating on {len(X_val)} samples...")
        self.pipeline.fit(X_train, y_train)
        
        # Evaluate model
        logger.info("Evaluating model on validation set...")
        val_acc = self.pipeline.score(X_val, y_val)
        
        # Get predictions for confusion matrix
        y_pred = self.pipeline.predict(X_val)
        cm = confusion_matrix(y_val, y_pred)
        
        # Generate classification report
        class_report = classification_report(
            y_val, y_pred, 
            target_names=self.labels,
            output_dict=True
        )
        
        logger.info(f"Validation accuracy: {val_acc:.4f}")
        logger.info(f"Confusion matrix:\n{cm}")
        logger.info("Classification report:")
        for label in self.labels:
            metrics = class_report[label]
            logger.info(f"  {label}: precision={metrics['precision']:.2f}, recall={metrics['recall']:.2f}, f1-score={metrics['f1-score']:.2f}")
        
        # Return results
        return {
            'accuracy': val_acc,
            'confusion_matrix': cm.tolist(),
            'classification_report': class_report
        }
    
    def predict(self, texts):
        """
        Predict sentiment for texts
        
        Args:
            texts (list): List of texts
            
        Returns:
            list: Predicted sentiment labels
        """
        if self.pipeline is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Check for critical negative words which always trigger negative sentiment
        predictions = []
        for text in texts:
            # Check for critical negative words first
            if any(crit_word in text.lower() for crit_word in self.critical_negative_words):
                predictions.append(0)  # Negative
            else:
                # Use the model for prediction
                pred = self.pipeline.predict([text])[0]
                predictions.append(pred)
                
        return predictions
    
    def explain_prediction(self, text):
        """
        Provide explanation for sentiment prediction
        
        Args:
            text (str): Input text
            
        Returns:
            dict: Explanation details
        """
        # Check for critical negative words first
        critical_words_found = [word for word in text.lower().split() 
                              if any(crit_word in word for crit_word in self.critical_negative_words)]
        
        if critical_words_found:
            sentiment_id = 0  # Negative
            sentiment = self.labels[sentiment_id]
        else:
            # Predict sentiment using the pipeline
            sentiment_id = self.pipeline.predict([text])[0]
            sentiment = self.labels[sentiment_id]
        
        # Count positive and negative words
        positive_words_found = [word for word in text.lower().split() if word in self.positive_words]
        negative_words_found = [word for word in text.lower().split() if word in self.negative_words]
        
        positive_count = len(positive_words_found)
        negative_count = len(negative_words_found)
        critical_count = len(critical_words_found)
        
        # Calculate sentiment ratio
        sentiment_ratio = (positive_count + 1) / (negative_count + 1)
        
        # Create explanation
        if sentiment == 'positive':
            if positive_count > 0:
                reason = f"Text obsahuje pozitivní slova jako: {', '.join(positive_words_found[:5])}"
            else:
                reason = "Text má celkově pozitivní tón, i když neobsahuje konkrétní pozitivní slova z našeho slovníku."
        elif sentiment == 'negative':
            if critical_count > 0:
                reason = f"Text obsahuje silně negativní slova spojená s neštěstím nebo tragédií: {', '.join(critical_words_found[:5])}"
            elif negative_count > 0:
                reason = f"Text obsahuje negativní slova jako: {', '.join(negative_words_found[:5])}"
            else:
                reason = "Text má celkově negativní tón, i když neobsahuje konkrétní negativní slova z našeho slovníku."
        else:
            reason = "Text obsahuje vyváženou směs pozitivních a negativních slov nebo neobsahuje dostatek slov s emočním nábojem."
        
        return {
            'text': text,
            'predicted_sentiment': sentiment,
            'positive_words': positive_words_found[:10],
            'negative_words': negative_words_found[:10],
            'critical_words': critical_words_found[:10],
            'positive_word_count': positive_count,
            'negative_word_count': negative_count,
            'critical_word_count': critical_count,
            'sentiment_ratio': sentiment_ratio,
            'reason': reason
        }
    
    def save_model(self, model_dir):
        """
        Save model to disk
        
        Args:
            model_dir (str): Directory to save the model
        """
        if self.pipeline is None:
            raise ValueError("No model to save. Train the model first.")
        
        # Create directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Save pipeline
        with open(os.path.join(model_dir, 'pipeline.pkl'), 'wb') as f:
            pickle.dump(self.pipeline, f)
        logger.info(f"Model pipeline saved to {os.path.join(model_dir, 'pipeline.pkl')}")
        
        # Save lexicons
        with open(os.path.join(model_dir, 'lexicons.pkl'), 'wb') as f:
            pickle.dump({
                'positive_words': self.positive_words,
                'negative_words': self.negative_words,
                'critical_negative_words': self.critical_negative_words
            }, f)
        
        # Save model info
        model_info = {
            'max_features': self.max_features,
            'labels': self.labels
        }
        
        with open(os.path.join(model_dir, 'model_info.pkl'), 'wb') as f:
            pickle.dump(model_info, f)
        
        logger.info(f"Model fully saved to {model_dir}")
    
    @classmethod
    def load_model(cls, model_dir):
        """
        Load model from disk
        
        Args:
            model_dir (str): Directory containing the saved model
            
        Returns:
            EnhancedSentimentAnalyzer: Loaded model
        """
        # Load model info
        with open(os.path.join(model_dir, 'model_info.pkl'), 'rb') as f:
            model_info = pickle.load(f)
        
        # Create instance
        instance = cls(
            max_features=model_info['max_features']
        )
        
        # Load pipeline
        with open(os.path.join(model_dir, 'pipeline.pkl'), 'rb') as f:
            instance.pipeline = pickle.load(f)
        
        # Load lexicons
        with open(os.path.join(model_dir, 'lexicons.pkl'), 'rb') as f:
            lexicons = pickle.load(f)
            instance.positive_words = lexicons['positive_words']
            instance.negative_words = lexicons['negative_words']
            if 'critical_negative_words' in lexicons:
                instance.critical_negative_words = lexicons['critical_negative_words']
        
        # Set labels
        instance.labels = model_info['labels']
        
        logger.info(f"Model loaded from {model_dir}")
        return instance

def save_word_lists():
    """Save the positive and negative word lists to files"""
    generate_czech_sentiment_lexicons()

def main():
    """Main function for training sentiment analyzer"""
    # Create necessary directories
    os.makedirs(os.path.join(project_root, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(project_root, 'models'), exist_ok=True)
    os.makedirs(os.path.join(project_root, 'data'), exist_ok=True)
    
    # Write word lists to files
    save_word_lists()
    
    # Set model output directory
    model_dir = os.path.join(project_root, 'models', 'sentiment_analyzer')
    os.makedirs(model_dir, exist_ok=True)
    
    # Always train model from scratch to incorporate CZE-NEC insights
    logger.info("Starting model training")
    
    # Connect to database
    conn = connect_to_database()
    
    # Load articles (get 3000 to ensure we have enough after filtering)
    if conn:
        df = load_articles_from_db(conn, limit=3000)
        conn.close()
    else:
        df = load_local_articles()
    
    if df is None or len(df) < 1500:
        logger.error(f"Insufficient data for training. Need at least 1500 articles, got {len(df) if df is not None else 0}")
        return
    
    # Preprocess articles
    df = preprocess_articles(df)
    
    # Create analyzer with optimized parameters for Czech
    analyzer = EnhancedSentimentAnalyzer(max_features=15000)
    
    # Auto-label the data
    logger.info("Auto-labeling data based on sentiment lexicons...")
    df['sentiment'] = analyzer.auto_label_data(df['text'].tolist())
    
    # Create balanced dataset with exactly 1500 samples for training (500 per class)
    logger.info("Creating balanced dataset with exactly 1500 samples...")
    samples_per_class = 500  # 500 per class = 1500 total
    balanced_df = pd.DataFrame()
    
    for sentiment in [0, 1, 2]:  # negative, neutral, positive
        class_df = df[df['sentiment'] == sentiment]
        
        if len(class_df) >= samples_per_class:
            # If we have enough, sample without replacement
            class_df = class_df.sample(samples_per_class, random_state=42)
        else:
            # If we don't have enough, oversample with replacement
            class_df = class_df.sample(samples_per_class, replace=True, random_state=42)
        
        balanced_df = pd.concat([balanced_df, class_df])
    
    # Shuffle the balanced dataset
    train_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Create test set from remaining articles
    remaining_indices = df.index.difference(train_df.index)
    test_df = df.loc[remaining_indices].sample(min(len(remaining_indices), 500), random_state=42)
    
    logger.info(f"Train set: {len(train_df)} samples, Test set: {len(test_df)} samples")
    
    # Train the model
    analyzer.fit(
        texts=train_df['text'].tolist(),
        labels=train_df['sentiment'].tolist(),
        validation_split=0.1
    )
    
    # Save the model
    analyzer.save_model(model_dir)
    logger.info("Model training and saving completed")
    
    # Test on sample sentences
    test_sentences = [
        "Vláda schválila nový zákon o podpoře podnikání, který přinese firmám úlevy na daních.",
        "Tragická nehoda na dálnici D1 si vyžádala tři lidské životy.",
        "Spotřebitelské ceny zůstaly v březnu stabilní, inflace se udržela na nízké úrovni."
    ]
    
    logger.info("Testing model on sample sentences:")
    for sentence in test_sentences:
        sentiment_id = analyzer.predict([sentence])[0]
        sentiment = analyzer.labels[sentiment_id]
        explanation = analyzer.explain_prediction(sentence)
        logger.info(f"Text: {sentence}")
        logger.info(f"Predicted sentiment: {sentiment}")
        logger.info(f"Reason: {explanation['reason']}")
        logger.info("-" * 50)

if __name__ == "__main__":
    main()