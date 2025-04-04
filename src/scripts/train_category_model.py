import sys
import os
import logging
import pandas as pd
import numpy as np
import json
import pickle
from datetime import datetime
import pyodbc
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix  
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
log_dir = os.path.join(project_root, 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"train_category_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                   handlers=[logging.FileHandler(log_file, encoding='utf-8'), logging.StreamHandler()])
logger = logging.getLogger(__name__)

DB_SERVER = "193.85.203.188"
DB_NAME = "boska"  
DB_USER = "boska"
DB_PASSWORD = "123456"

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)

class CategoryClassifier:
    def __init__(self, max_features=12000, max_categories=15, min_samples_per_category=10):
        self.max_features = max_features
        self.max_categories = max_categories
        self.min_samples_per_category = min_samples_per_category
        self.pipeline = None
        self.categories = []
        
        self.category_map = {
            'sport': 'Sport', 'sports': 'Sport', 'fotbal': 'Sport', 'hokej': 'Sport',
            'tenis': 'Sport', 'olympiáda': 'Sport', 'ms': 'Sport', 'me': 'Sport',
            'liga': 'Sport', 'mistrovství': 'Sport', 'turnaj': 'Sport', 'zápas': 'Sport',
            
            'politika': 'Politika', 'vláda': 'Politika', 'domácí': 'Domácí', 
            'volby': 'Politika', 'parlament': 'Politika', 'strana': 'Politika',
            
            'zahraničí': 'Zahraničí', 'svět': 'Zahraničí', 'světové': 'Zahraničí',
            'eu': 'Zahraničí', 'válka': 'Zahraničí', 'konflikt': 'Zahraničí',
            
            'ekonomika': 'Ekonomika', 'byznys': 'Ekonomika', 'finance': 'Ekonomika',
            'peníze': 'Ekonomika', 'burza': 'Ekonomika', 'trh': 'Ekonomika',
            
            'kultura': 'Kultura', 'film': 'Kultura', 'hudba': 'Kultura', 
            'divadlo': 'Kultura', 'umění': 'Kultura', 'koncert': 'Kultura',
            
            'technologie': 'Technologie', 'it': 'Technologie', 'tech': 'Technologie',
            'software': 'Technologie', 'hardware': 'Technologie', 'internet': 'Technologie'
        }
    
    def build_model(self):
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=self.max_features,  
                ngram_range=(1, 3),
                min_df=2, 
                stop_words='english'
            )),
            ('clf', LogisticRegression(
                C=1.0,   
                multi_class='multinomial',  
                solver='lbfgs',
                max_iter=1000,
                class_weight='balanced' 
            ))
        ])
        return self.pipeline
    
    def normalize_category(self, category):
        if not isinstance(category, str):
            return 'Other'
        
        category_lower = category.lower().strip()
        
        specific_rules = {
            'ekonomika': [
                'elektřina', 'burza', 'ceny', 'trh', 'finance',  
                'podnikatel', 'obchod', 'měna', 'akcie', 'byznys',  
                'fond', 'investice', 'kapitál'
            ],
            'sport': [
                'výhra', 'prohra', 'gól', 'branka', 'skóre',  
                'hřiště', 'stadion', 'tým', 'hráč', 'trenér'  
            ],
            'technologie': [
                'software', 'hardware', 'počítač', 'internet',  
                'aplikace', 'systém', 'technologie', 'it',  
                'programování', 'smartphone'
            ]
        }
        
        for category_name, keywords in specific_rules.items():
            if any(keyword in category_lower for keyword in keywords):
                return category_name.capitalize()
        
        for key, value in self.category_map.items():
            if key in category_lower:
                return value
        
        return category_lower.capitalize()
    
    def preprocess_categories(self, categories):
        normalized_categories = [
            self.normalize_category(str(cat))  
            for cat in categories  
            if str(cat).strip()
        ]
        
        category_counts = pd.Series(normalized_categories).value_counts()
        logger.info(f"Top categories after normalization: {dict(category_counts)}")
        
        valid_categories = category_counts[
            category_counts >= self.min_samples_per_category  
        ].index.tolist()
        
        essential_categories = [
            'Zahraničí', 'Domácí', 'Ekonomika',  
            'Technologie', 'Kultura', 'Sport', 'Politika' 
        ]
        
        for cat in essential_categories:
            if cat in category_counts and cat not in valid_categories:
                valid_categories.append(cat)
        
        if len(valid_categories) > self.max_categories:
            top_categories = category_counts[valid_categories].nlargest(self.max_categories)
            valid_categories = top_categories.index.tolist()
        
        if 'Other' not in valid_categories:
            valid_categories.append('Other')
        
        self.categories = valid_categories
        category_to_id = {cat: i for i, cat in enumerate(valid_categories)}
        
        labels = [
            category_to_id.get(
                self.normalize_category(str(cat)),  
                category_to_id['Other'] 
            )  
            for cat in categories
            if str(cat).strip()  
        ]
        
        return labels, self.categories

    def fit(self, texts, categories):
        texts = list(texts)
        categories = list(categories)
        
        valid_entries = [
            (text, category)  
            for text, category in zip(texts, categories)  
            if text and isinstance(text, str) and text.strip()   
            and category and isinstance(category, str) and category.strip()
        ]
        
        if not valid_entries:
            raise ValueError("No valid text-category pairs found")
        
        filtered_texts, filtered_categories = zip(*valid_entries)
        filtered_texts = list(filtered_texts)
        filtered_categories = list(filtered_categories)
        
        labels, self.categories = self.preprocess_categories(filtered_categories)

        X_train, X_val, y_train, y_val = train_test_split(
            filtered_texts, labels,   
            test_size=0.2,   
            random_state=42,   
            stratify=labels
        )
        
        domain_examples = {
            'Ekonomika': [
                "Ceny elektřiny na burze klesly na nejnižší úroveň",
                "Investiční fond oznámil roční výsledky hospodaření", 
                "Kurz měny se výrazně změnil oproti předchozímu týdnu"
            ],
            'Sport': [
                "Fotbalisté Sparty porazili Slavii ve vypjatém derby",
                "Hokejisté zahájili mistrovství světa výhrou", 
                "Tenisový turnaj přinesl překvapivé výsledky"
            ],
            'Technologie': [
                "Nový software umožňuje rychlejší zpracování dat",
                "Technologická společnost představila inovativní řešení",
                "Vývoj umělé inteligence pokračuje mílovými kroky" 
            ]
        }
        
        for category_name, examples in domain_examples.items():
            if category_name in self.categories:
                category_idx = self.categories.index(category_name)
                X_train.extend(examples)
                y_train.extend([category_idx] * len(examples))
        
        if self.pipeline is None:
            self.build_model() 
        
        logger.info(f"Training on {len(X_train)} samples, {len(self.categories)} categories")
        self.pipeline.fit(X_train, y_train)
        
        val_acc = self.pipeline.score(X_val, y_val)
        y_pred = self.pipeline.predict(X_val)
        cm = confusion_matrix(y_val, y_pred)  

        logger.info(f"Validation accuracy: {val_acc:.4f}")

        labels_present = sorted(set(y_val + y_pred))
        target_names = [self.categories[label] for label in labels_present if label < len(self.categories)] 
        
        return {
            'accuracy': val_acc,
            'confusion_matrix': cm,
            'classification_report': classification_report(
                y_val, y_pred,   
                labels=labels_present,
                target_names=target_names,   
                output_dict=True,
                zero_division=0
            )
        }

    def predict(self, texts):
        if self.pipeline is None:
            raise ValueError("Model not trained yet")
        
        category_ids = self.pipeline.predict(texts)
        predictions = [self.categories[id] for id in category_ids]
        
        return predictions

    def predict_proba(self, texts): 
        if self.pipeline is None:
            raise ValueError("Model not trained yet")
        
        probas = self.pipeline.predict_proba(texts)

        result = []
        for proba in probas:
            result.append({cat: prob for cat, prob in zip(self.categories, proba)})
        
        return result

    def save_model(self, model_dir):
        os.makedirs(model_dir, exist_ok=True)
        
        with open(os.path.join(model_dir, 'pipeline.pkl'), 'wb') as f:
            pickle.dump(self.pipeline, f)
        
        with open(os.path.join(model_dir, 'categories.pkl'), 'wb') as f:
            pickle.dump(self.categories, f)
        
        with open(os.path.join(model_dir, 'category_map.pkl'), 'wb') as f:
            pickle.dump(self.category_map, f)
        
        with open(os.path.join(model_dir, 'config.pkl'), 'wb') as f:
            pickle.dump({
                'max_features': self.max_features,
                'max_categories': self.max_categories,
                'min_samples_per_category': self.min_samples_per_category
            }, f)
        
        logger.info(f"Model saved to {model_dir}")
    
    @classmethod
    def load_model(cls, model_dir):
        with open(os.path.join(model_dir, 'config.pkl'), 'rb') as f:
            config = pickle.load(f)
        
        instance = cls(
            max_features=config['max_features'],
            max_categories=config['max_categories'], 
            min_samples_per_category=config['min_samples_per_category']
        )
        
        with open(os.path.join(model_dir, 'pipeline.pkl'), 'rb') as f:
            instance.pipeline = pickle.load(f)
        
        with open(os.path.join(model_dir, 'categories.pkl'), 'rb') as f:
            instance.categories = pickle.load(f)
        
        try:
            with open(os.path.join(model_dir, 'category_map.pkl'), 'rb') as f:
                instance.category_map = pickle.load(f)
        except:
            pass
        
        logger.info(f"Model loaded from {model_dir}")
        return instance

def connect_to_database():
    try:
        config_path = os.path.join(project_root, 'config', 'database.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                server = config.get('server', DB_SERVER)
                database = config.get('database', DB_NAME) 
                username = config.get('username', DB_USER)
                password = config.get('password', DB_PASSWORD)
        else:
            server = DB_SERVER
            database = DB_NAME
            username = DB_USER 
            password = DB_PASSWORD
        
        conn_str = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}"
        conn = pyodbc.connect(conn_str)
        logger.info(f"Connected to database {database} on {server}")
        return conn
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        return None

def load_articles(conn, limit=3000):
    try:
        query = f"SELECT TOP {limit} ArticleText, Title, Category, SourceName, PublicationDate FROM Articles "                 \
                f"WHERE ArticleText IS NOT NULL AND Category IS NOT NULL AND LEN(ArticleText) > 100 "                         \
                f"ORDER BY PublicationDate DESC"
        
        df = pd.read_sql(query, conn)
        logger.info(f"Loaded {len(df)} articles from database")
        return df
    except Exception as e:
        logger.error(f"Error loading articles: {e}")
        
        data_paths = [
            os.path.join(project_root, 'data', 'display', 'all_articles.json'),
            os.path.join(project_root, 'data', 'all_articles.json') 
        ]
        
        for path in data_paths:
            if os.path.exists(path):
                try:
                    with open(path, 'r', encoding='utf-8') as f:  
                        data = json.load(f)
                    df = pd.DataFrame(data)
                    logger.info(f"Loaded {len(df)} articles from {path}") 
                    return df
                except Exception as json_error:
                    logger.error(f"Error loading JSON from {path}: {json_error}")
        
        logger.error("Could not load articles from database or local files")
        return None

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    
    replacements = {
        'ě': 'e', 'š': 's', 'č': 'c', 'ř': 'r', 'ž': 'z', 'ý': 'y',
        'á': 'a', 'í': 'i', 'é': 'e', 'ú': 'u', 'ů': 'u', 'ň': 'n', 
        'ť': 't', 'ď': 'd'
    }
    
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    
    return text

def main():
    for directory in ['models/category_classifier', 'reports/models', 'reports/figures']:
        dir_path = os.path.join(project_root, directory)
        os.makedirs(dir_path, exist_ok=True)
    
    model_dir = os.path.join(project_root, 'models', 'category_classifier')
    
    force_retrain = '--force' in sys.argv
    
    if not force_retrain and os.path.exists(os.path.join(model_dir, 'pipeline.pkl')):
        logger.info("Loading existing model")
        classifier = CategoryClassifier.load_model(model_dir)
    else:
        logger.info("Training new category model")
        
        conn = connect_to_database()
        df = load_articles(conn) if conn else None
        if conn: conn.close()
        
        if df is None or len(df) < 100:
            logger.error(f"Not enough data: {len(df) if df is not None else 0}")
            return
        
        required_columns = ['Title', 'Content', 'Category']
        rename_map = {'ArticleText': 'Content'}
        
        for old, new in rename_map.items():
            if old in df.columns and new not in df.columns:
                df[new] = df[old]
        
        df['Category'] = df['Category'].fillna('Uncategorized')
        
        df['Title_processed'] = df['Title'].apply(preprocess_text)
        df['Content_processed'] = df['Content'].apply(preprocess_text)
        
        df['Text'] = df['Title_processed'] + ' ' + df['Title_processed'] + ' ' + df['Content_processed']
        
        classifier = CategoryClassifier(max_features=12000, max_categories=15, min_samples_per_category=10)
        results = classifier.fit(df['Text'], df['Category'])
        
        classifier.save_model(model_dir)
        
        results_path = os.path.join(project_root, 'reports', 'models', 'category_classifier_results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
    
    logger.info("Testing model on sample articles...")
    test_texts = [
        "Fotbalisté Sparty Praha porazili Plzeň 3:0 a posunuli se na první místo ligové tabulky.",
        "Vláda schválila nový zákon o dani z příjmu, který přinese změny pro podnikatele.", 
        "Vědecký tým objevil nový druh dinosaura v oblasti jižní Ameriky.",
        "Zápas byl přerušen kvůli nepříznivému počasí po druhém poločase.", 
        "Hokejisté Třince obhájili titul mistra extraligy.",
        "Ceny elektřiny na burze klesly na nejnižší úroveň za poslední dva roky."    
    ]
    
    for text in test_texts:
        prediction = classifier.predict([text])[0]
        logger.info(f"Text: {text}")
        logger.info(f"Predicted category: {prediction}")  
        logger.info("-" * 30)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"Error: {e}")