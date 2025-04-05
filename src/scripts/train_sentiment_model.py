import os
import sys
import logging
import pandas as pd
import numpy as np
import json
import pickle
import subprocess
import pyodbc
from datetime import datetime
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# Add path to project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(project_root)

# Setup logging
os.makedirs(os.path.join(project_root, 'logs'), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(project_root, 'logs', f"train_sentiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"), encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("sentiment_analyzer")

# Install datasets library if not already installed
try:
    import datasets
except ImportError:
    logger.info("Installing datasets library...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets"])
    import datasets
    logger.info("Datasets library installed successfully")

class SentimentAnalyzer:
    """Sentiment analyzer for Czech news articles"""
    
    def __init__(self, max_features=15000):
        """Initialize the sentiment analyzer"""
        self.max_features = max_features
        self.pipeline = None
        self.labels = ['negative', 'neutral', 'positive']
        
        # Load lexicons
        self.positive_words = []
        self.negative_words = []
        self.critical_negative_words = set([
            'smrt', 'úmrtí', 'zemřel', 'zemřela', 'zahynul', 'mrtvý', 'tragick', 'tragédie', 
            'katastrofa', 'neštěstí', 'oběť', 'obětí', 'nehoda', 'smrtelný', 'zraněn',
            'válka', 'útok', 'konflikt', 'krize'
        ])
    
    def load_words(self, word_lists):
        """Load word lists from provided dictionaries"""
        self.positive_words = word_lists.get('positive_words', [])
        self.negative_words = word_lists.get('negative_words', [])
        if 'critical_negative_words' in word_lists:
            self.critical_negative_words = word_lists['critical_negative_words']
        return self
    
    def extract_sentiment_features(self, texts):
        """Extract sentiment features from texts"""
        if not isinstance(texts, list):
            texts = [texts]
        
        features = pd.DataFrame()
        
        # Count positive and negative words
        features['positive_word_count'] = [
            sum(1 for word in str(text).lower().split() if word in self.positive_words) 
            for text in texts
        ]
        
        features['negative_word_count'] = [
            sum(1 for word in str(text).lower().split() if word in self.negative_words)
            for text in texts
        ]
        
        # Calculate sentiment ratio
        features['sentiment_ratio'] = (features['positive_word_count'] + 1) / (features['negative_word_count'] + 1)
        
        return features
    
    def predict(self, texts):
        """Predict sentiment for texts"""
        if not isinstance(texts, list):
            texts = [texts]
        
        if self.pipeline is None:
            raise ValueError("Model not trained. Must load model first.")
        
        # Check for critical negative words which trigger negative sentiment
        predictions = []
        for text in texts:
            text_lower = str(text).lower()
            
            # Check for critical negative words first
            if any(crit_word in text_lower for crit_word in self.critical_negative_words):
                predictions.append(0)  # Negative
            else:
                # Use the model for prediction
                try:
                    pred = self.pipeline.predict([text])[0]
                    predictions.append(pred)
                except:
                    # Fallback to lexicon-based prediction if model fails
                    pos_count = sum(1 for word in text_lower.split() if word in self.positive_words)
                    neg_count = sum(1 for word in text_lower.split() if word in self.negative_words)
                    
                    if neg_count > pos_count * 1.2:
                        predictions.append(0)  # negative
                    elif pos_count > neg_count * 1.2:
                        predictions.append(2)  # positive
                    else:
                        predictions.append(1)  # neutral
        
        return predictions
    
    def explain_prediction(self, text):
        """Provide explanation for sentiment prediction"""
        text_lower = str(text).lower()
        
        # Count positive and negative words
        positive_words_found = [word for word in text_lower.split() if word in self.positive_words]
        negative_words_found = [word for word in text_lower.split() if word in self.negative_words]
        critical_words_found = [word for word in text_lower.split() 
                               if any(crit_word in word for crit_word in self.critical_negative_words)]
        
        positive_count = len(positive_words_found)
        negative_count = len(negative_words_found)
        critical_count = len(critical_words_found)
        
        # Determine sentiment
        sentiment_id = self.predict([text])[0]
        sentiment = self.labels[sentiment_id]
        
        # Calculate sentiment ratio
        sentiment_ratio = (positive_count + 1) / (negative_count + 1)
        
        # Create explanation
        reason = ""
        if sentiment == 'positive':
            if positive_count > 0:
                reason = f"Text obsahuje pozitivní slova jako: {', '.join(positive_words_found[:5])}"
            else:
                reason = "Text má celkově pozitivní tón."
        elif sentiment == 'negative':
            if critical_count > 0:
                reason = f"Text obsahuje silně negativní slova: {', '.join(critical_words_found[:5])}"
            elif negative_count > 0:
                reason = f"Text obsahuje negativní slova jako: {', '.join(negative_words_found[:5])}"
            else:
                reason = "Text má celkově negativní tón."
        else:
            reason = "Text obsahuje vyváženou směs pozitivních a negativních slov."
        
        return {
            'text': text,
            'predicted_sentiment': sentiment,
            'positive_words': positive_words_found[:10],
            'negative_words': negative_words_found[:10],
            'positive_word_count': positive_count,
            'negative_word_count': negative_count,
            'sentiment_ratio': sentiment_ratio,
            'reason': reason
        }
    
    def save_model(self, model_dir):
        """Save model to disk"""
        if self.pipeline is None:
            raise ValueError("No model to save.")
        
        os.makedirs(model_dir, exist_ok=True)
        
        # Save pipeline
        with open(os.path.join(model_dir, 'pipeline.pkl'), 'wb') as f:
            pickle.dump(self.pipeline, f)
        
        # Save lexicons
        with open(os.path.join(model_dir, 'lexicons.pkl'), 'wb') as f:
            pickle.dump({
                'positive_words': self.positive_words,
                'negative_words': self.negative_words,
                'critical_negative_words': self.critical_negative_words
            }, f)
        
        # Save model info
        with open(os.path.join(model_dir, 'model_info.pkl'), 'wb') as f:
            pickle.dump({
                'max_features': self.max_features,
                'labels': self.labels
            }, f)
        
        # Save the class implementation itself
        with open(os.path.join(model_dir, 'sentiment_analyzer.py'), 'w', encoding='utf-8') as f:
            import inspect
            f.write(inspect.getsource(self.__class__))
            
        logger.info(f"Model saved to {model_dir}")
    
    @classmethod
    def load_model(cls, model_dir):
        """Load model from disk"""
        instance = cls()
        
        try:
            # Load model info
            with open(os.path.join(model_dir, 'model_info.pkl'), 'rb') as f:
                model_info = pickle.load(f)
                instance.max_features = model_info.get('max_features', 15000)
                instance.labels = model_info.get('labels', ['negative', 'neutral', 'positive'])
            
            # Load pipeline
            with open(os.path.join(model_dir, 'pipeline.pkl'), 'rb') as f:
                instance.pipeline = pickle.load(f)
            
            # Load lexicons
            with open(os.path.join(model_dir, 'lexicons.pkl'), 'rb') as f:
                lexicons = pickle.load(f)
                instance.load_words(lexicons)
            
            return instance
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return instance

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
    
    # Define critical negative words
    critical_negative_words = {
        'smrt', 'úmrtí', 'zemřel', 'zemřela', 'zahynul', 'mrtvý', 'tragick', 'tragédie', 
        'katastrofa', 'neštěstí', 'oběť', 'obětí', 'nehoda', 'smrtelný', 'zraněn', 
        'vyžádala', 'zahynul', 'usmrcen', 'zabil', 'zabit', 'utonul', 'zastřelen',
        'válka', 'útok', 'konflikt', 'krize', 'hrozba', 'nebezpečí', 'násilí',
        'zbankrotoval', 'zadlužený', 'chudoba', 'nezaměstnanost', 'kolaps'
    }
    
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
    return positive_words, negative_words, critical_negative_words

def connect_to_database():
    """Connect to SQL Server database using the provided configuration"""
    try:
        # Load configuration
        config_path = os.path.join(project_root, 'config', 'database.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                server = config.get('server', "193.85.203.188")
                database = config.get('database', "boska")
                username = config.get('username', "boska")
                password = config.get('password', "123456")
                driver = config.get('driver', "{ODBC Driver 17 for SQL Server}")
        else:
            # Use default values if config file doesn't exist
            server = "193.85.203.188"
            database = "boska"
            username = "boska"
            password = "123456"
            driver = "{ODBC Driver 17 for SQL Server}"
            
            # Create config file for future use
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump({
                    "server": server,
                    "database": database,
                    "username": username,
                    "password": password,
                    "driver": driver
                }, f, indent=4)
        
        # Connect to database
        conn_str = f"DRIVER={driver};SERVER={server};DATABASE={database};UID={username};PWD={password}"
        conn = pyodbc.connect(conn_str)
        
        logger.info(f"Successfully connected to database {database} on {server}")
        return conn
    
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        return None

def load_articles_from_database(conn, limit=1000):
    """Load articles from database"""
    try:
        if conn is None:
            logger.warning("No database connection provided.")
            return None
            
        query = f"""
        SELECT TOP {limit} 
            ArticleText as Content, 
            Title, 
            Category, 
            SourceName as Source,
            PublicationDate as PublishDate
        FROM Articles
        WHERE ArticleText IS NOT NULL AND LEN(ArticleText) > 100
        ORDER BY PublicationDate DESC
        """
        
        df = pd.read_sql(query, conn)
        logger.info(f"Loaded {len(df)} articles from database")
        return df
    
    except Exception as e:
        logger.error(f"Error loading articles from database: {e}")
        return None

def load_articles_from_huggingface():
    """Load articles from the Hugging Face dataset"""
    try:
        # Load dataset
        logger.info("Loading Czech news dataset from Hugging Face...")
        dataset = datasets.load_dataset("CIIRC-NLP/czech_news_simple-cs")
        
        # Convert to pandas DataFrame
        df = pd.DataFrame(dataset['test'])
        
        # Rename columns to match our expected format
        df = df.rename(columns={
            'headline': 'Title',
            'text': 'Content',
            'category': 'Category'
        })
        
        # Add Source column if not present
        if 'Source' not in df.columns:
            df['Source'] = 'CIIRC-NLP'
        
        logger.info(f"Loaded {len(df)} articles from Hugging Face dataset")
        return df
    
    except Exception as e:
        logger.error(f"Error loading articles from Hugging Face: {e}")
        return None

def load_local_articles():
    """Load articles from local files (backup method)"""
    data_paths = [
        os.path.join(project_root, 'data', 'display', 'all_articles.json'),
        os.path.join(project_root, 'data', 'processed_scraped')
    ]
    
    for path in data_paths:
        try:
            if os.path.isfile(path) and path.endswith('.json'):
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                df = pd.DataFrame(data)
                logger.info(f"Loaded {len(df)} articles from {path}")
                return df
            elif os.path.isdir(path):
                json_files = [f for f in os.listdir(path) if f.endswith('.json')]
                if json_files:
                    newest_file = sorted(json_files)[-1]
                    with open(os.path.join(path, newest_file), 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    df = pd.DataFrame(data)
                    logger.info(f"Loaded {len(df)} articles from {os.path.join(path, newest_file)}")
                    return df
        except Exception as e:
            logger.warning(f"Failed to load from {path}: {e}")
    
    return None

def create_minimal_dataset():
    """Create a minimal dataset with guaranteed examples for each sentiment class"""
    logger.warning("Creating minimal dataset for model training")
    
    return pd.DataFrame({
        'Title': ['Positive article', 'Positive article 2', 'Positive article 3',
                 'Neutral article', 'Neutral article 2', 'Neutral article 3',
                 'Negative article', 'Negative article 2', 'Negative article 3'],
        'Content': [
            'Vláda schválila nový zákon o podpoře podnikání, který přinese firmám úlevy na daních.',
            'Ekonomika roste, nezaměstnanost klesá a lidé jsou spokojenější než dříve.',
            'Nová technologie pomáhá zlepšit životní prostředí a šetří energii.',
            'Teplota se dnes pohybuje kolem 20 stupňů Celsia.',
            'Dnes je středa, zítra bude čtvrtek a pak pátek.',
            'Autobus přijel na zastávku podle jízdního řádu.',
            'Tragická nehoda na dálnici D1 si vyžádala tři lidské životy.',
            'Válka pokračuje a podle zpráv zahynulo dalších 50 lidí.',
            'Ekonomická krize způsobila krachy firem a nárůst nezaměstnanosti.'
        ]
    })

def preprocess_articles(df):
    """Preprocess articles for training"""
    if df is None or len(df) == 0:
        logger.error("No data to preprocess")
        return None
        
    # Check and rename columns if needed
    column_mapping = {
        'ArticleText': 'Content',
        'SourceName': 'Source',
    }
    
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns and new_col not in df.columns:
            df[new_col] = df[old_col]
    
    # Combine title and content for better analysis
    if 'Title' in df.columns and 'Content' in df.columns:
        df['text'] = df['Title'] + ' ' + df['Content']
    elif 'Content' in df.columns:
        df['text'] = df['Content']
    elif 'Title' in df.columns:
        df['text'] = df['Title']
    else:
        logger.error("No text content found in DataFrame")
        return None
    
    # Filter out rows with empty text
    df = df[df['text'].notna() & (df['text'].str.len() > 50)]
    logger.info(f"After filtering, {len(df)} articles remain")
    
    return df

def auto_label_data(df, positive_words, negative_words, critical_negative_words):
    """Automatically label data with sentiment based on lexicon analysis"""
    # Create sentiment analyzer for feature extraction
    sentiment_analyzer = SentimentAnalyzer()
    sentiment_analyzer.load_words({
        'positive_words': positive_words,
        'negative_words': negative_words,
        'critical_negative_words': critical_negative_words
    })
    
    # Extract features
    features = sentiment_analyzer.extract_sentiment_features(df['text'].tolist())
    
    # Auto-label data
    labels = []
    for i, row in features.iterrows():
        pos_count = row['positive_word_count']
        neg_count = row['negative_word_count']
        ratio = row['sentiment_ratio']
        
        # Check for critical negative words
        text = df.iloc[i]['text'].lower()
        if any(crit_word in text for crit_word in critical_negative_words):
            labels.append(0)  # negative
        elif ratio > 1.5 or (pos_count >= 3 and neg_count == 0):
            labels.append(2)  # positive
        elif ratio < 0.8 or (neg_count >= 2 and pos_count == 0):
            labels.append(0)  # negative
        else:
            labels.append(1)  # neutral
    
    df['sentiment'] = labels
    
    # Log distribution
    neg_count = (df['sentiment'] == 0).sum()
    neu_count = (df['sentiment'] == 1).sum()
    pos_count = (df['sentiment'] == 2).sum()
    
    logger.info(f"Auto-labeled data distribution: Negative={neg_count}, Neutral={neu_count}, Positive={pos_count}")
    
    return df

def create_balanced_dataset(df, min_samples_per_class=50):
    """Create a balanced dataset with equal representation of each class"""
    # Check if we need to add guaranteed examples
    class_counts = df['sentiment'].value_counts()
    guaranteed_examples = {
        0: [  # Negative
            "Tragická nehoda na dálnici D1 si vyžádala tři lidské životy.",
            "Válka pokračuje a podle zpráv zahynulo dalších 50 lidí.",
            "Ekonomická krize způsobila krachy firem a nárůst nezaměstnanosti."
        ],
        1: [  # Neutral
            "Teplota se dnes pohybuje kolem 20 stupňů Celsia.",
            "Dnes je středa, zítra bude čtvrtek a pak pátek.",
            "Autobus přijel na zastávku podle jízdního řádu."
        ],
        2: [  # Positive
            "Vláda schválila nový zákon o podpoře podnikání, který přinese firmám úlevy na daních.",
            "Ekonomika roste, nezaměstnanost klesá a lidé jsou spokojenější než dříve.",
            "Nová technologie pomáhá zlepšit životní prostředí a šetří energii."
        ]
    }
    
    # Add guaranteed examples if any class has fewer than required samples
    for class_id, examples in guaranteed_examples.items():
        if class_id not in class_counts or class_counts[class_id] < min_samples_per_class:
            for example in examples:
                if example not in df['text'].values:  # Avoid duplicates
                    df = pd.concat([df, pd.DataFrame({
                        'text': [example],
                        'sentiment': [class_id]
                    })], ignore_index=True)
    
    # Create balanced dataset
    balanced_df = pd.DataFrame()
    
    for class_id in [0, 1, 2]:  # negative, neutral, positive
        class_df = df[df['sentiment'] == class_id]
        
        if len(class_df) >= min_samples_per_class:
            # If we have enough, sample without replacement
            class_df = class_df.sample(min_samples_per_class, random_state=42)
        else:
            # If we don't have enough, oversample with replacement
            class_df = class_df.sample(min_samples_per_class, replace=True, random_state=42)
        
        balanced_df = pd.concat([balanced_df, class_df])
    
    # Shuffle the balanced dataset
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    logger.info(f"Created balanced dataset with {len(balanced_df)} samples ({min_samples_per_class} per class)")
    return balanced_df

def train_sentiment_model():
    """Train and save the sentiment model"""
    # Create necessary directories
    os.makedirs(os.path.join(project_root, 'models'), exist_ok=True)
    os.makedirs(os.path.join(project_root, 'data'), exist_ok=True)
    
    # Generate Czech sentiment lexicons
    positive_words, negative_words, critical_negative_words = generate_czech_sentiment_lexicons()
    
    # Set model output directory
    model_dir = os.path.join(project_root, 'models', 'sentiment_analyzer')
    os.makedirs(model_dir, exist_ok=True)
    
    # Load articles from multiple sources
    dfs = []
    
    # 1. Try database connection first
    conn = connect_to_database()
    if conn:
        df_db = load_articles_from_database(conn, limit=500)
        if df_db is not None and len(df_db) > 0:
            dfs.append(df_db)
        conn.close()
    
    # 2. Try Hugging Face dataset
    try:
        df_hf = load_articles_from_huggingface()
        if df_hf is not None and len(df_hf) > 0:
            dfs.append(df_hf)
    except Exception as e:
        logger.error(f"Error loading Hugging Face dataset: {e}")
    
    # 3. Try local files
    df_local = load_local_articles()
    if df_local is not None and len(df_local) > 0:
        dfs.append(df_local)
    
    # Combine all datasets
    if dfs:
        df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Combined dataset has {len(df)} articles")
    else:
        # Create minimal dataset if no data found
        df = create_minimal_dataset()
        logger.warning("Using minimal dataset for training")
    
    # Preprocess articles
    df = preprocess_articles(df)
    if df is None or len(df) < 9:  # Need at least 3 samples per class
        logger.error("Not enough valid articles after preprocessing.")
        df = create_minimal_dataset()
        df = preprocess_articles(df)
    
    # Auto-label data with sentiment
    df = auto_label_data(df, positive_words, negative_words, critical_negative_words)
    
    # Create balanced dataset
    # Use a smaller number if minimal dataset, larger if we have real data
    min_samples = min(50, max(3, len(df) // 10)) 
    balanced_df = create_balanced_dataset(df, min_samples_per_class=min_samples)
    
    # Create the sentiment analyzer
    sentiment_analyzer = SentimentAnalyzer(max_features=min(10000, len(balanced_df) * 10))
    
    # Load lexicons
    sentiment_analyzer.load_words({
        'positive_words': positive_words,
        'negative_words': negative_words,
        'critical_negative_words': critical_negative_words
    })
    
    # Create TF-IDF vectorizer and classifier with appropriate parameters based on dataset size
    n_estimators = min(100, max(50, len(balanced_df) // 3))
    tfidf = TfidfVectorizer(
        max_features=min(5000, len(balanced_df) * 10), 
        ngram_range=(1, 2), 
        min_df=1
    )
    clf = RandomForestClassifier(
        n_estimators=n_estimators, 
        min_samples_leaf=1, 
        random_state=42, 
        class_weight='balanced'
    )
    
    # Create and fit the pipeline
    pipeline = Pipeline([
        ('tfidf', tfidf),
        ('clf', clf)
    ])
    
    X = balanced_df['text'].values
    y = balanced_df['sentiment'].values
    
    try:
        # Try with train/test split if we have enough data
        if len(balanced_df) >= 30:  # With 10 samples per class, we can do a 70/30 split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=0.3, 
                random_state=42, 
                stratify=y
            )
            
            # Train the model
            pipeline.fit(X_train, y_train)
            
            # Evaluate the model
            y_pred = pipeline.predict(X_test)
            report = classification_report(y_test, y_pred, target_names=['negative', 'neutral', 'positive'], output_dict=True)
            
            logger.info("Model evaluation results:")
            for label, metrics in report.items():
                if label in ['negative', 'neutral', 'positive']:
                    logger.info(f"{label}: precision={metrics['precision']:.2f}, recall={metrics['recall']:.2f}, f1-score={metrics['f1-score']:.2f}")
        else:
            # Train on full dataset if too small for splitting
            logger.info("Dataset too small for train/test split, training on full dataset")
            pipeline.fit(X, y)
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        logger.info("Training on full dataset as fallback")
        try:
            pipeline.fit(X, y)
        except Exception as e2:
            logger.error(f"Critical training error: {e2}")
            # Last resort - create a dummy pipeline
            logger.warning("Creating fallback dummy model")
            from sklearn.dummy import DummyClassifier
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=100)),
                ('clf', DummyClassifier(strategy='stratified'))
            ])
            pipeline.fit(X, y)
    
    # Set the pipeline in the analyzer
    sentiment_analyzer.pipeline = pipeline
    
    # Save the model
    sentiment_analyzer.save_model(model_dir)
    logger.info(f"Model saved to {model_dir}")
    
    # Test on sample sentences
    test_sentences = [
        "Vláda schválila nový zákon o podpoře podnikání, který přinese firmám úlevy na daních.",
        "Tragická nehoda na dálnici D1 si vyžádala tři lidské životy.",
        "Spotřebitelské ceny zůstaly v březnu stabilní, inflace se udržela na nízké úrovni."
    ]
    
    logger.info("Testing model on sample sentences:")
    for sentence in test_sentences:
        sentiment_id = sentiment_analyzer.predict([sentence])[0]
        sentiment = sentiment_analyzer.labels[sentiment_id]
        explanation = sentiment_analyzer.explain_prediction(sentence)
        
        logger.info(f"Text: {sentence}")
        logger.info(f"Predicted sentiment: {sentiment}")
        logger.info(f"Reason: {explanation['reason']}")
        logger.info("-" * 50)

if __name__ == "__main__":
    try:
        train_sentiment_model()
        logger.info("Sentiment model training completed successfully")
    except Exception as e:
        logger.error(f"Error in sentiment model training: {e}")