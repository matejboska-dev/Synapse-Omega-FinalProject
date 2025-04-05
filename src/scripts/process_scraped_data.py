import sys
import os
import logging
import pandas as pd
import numpy as np
import json
from datetime import datetime
import glob

# fix project root based on this file location (src/scripts)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# ensure log directory exists
log_dir = os.path.join(project_root, 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"process_scraped_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# ensure processed directory exists
processed_dir = os.path.join(project_root, 'data', 'processed_scraped')
os.makedirs(processed_dir, exist_ok=True)

# logger configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# add project root to system path for module import
sys.path.append(project_root)

def find_latest_scraped_data():
    data_dir = os.path.join(project_root, 'data', 'scraped')
    if not os.path.exists(data_dir):
        logger.error(f"scraped data directory not found: {data_dir}")
        return None
    json_files = glob.glob(os.path.join(data_dir, 'articles_*.json'))
    if not json_files:
        logger.warning("no scraped data files found")
        return None
    latest_file = max(json_files, key=os.path.getmtime)
    logger.info(f"found latest scraped data file: {latest_file}")
    return latest_file

def load_scraped_data(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"loaded {len(data)} articles from {file_path}")
        return data
    except Exception as e:
        logger.error(f"error loading scraped data: {str(e)}")
        return None

def preprocess_texts(articles):
    try:
        df = pd.DataFrame(articles)
        from text_preprocessor import TextPreprocessor
        preprocessor = TextPreprocessor(language='czech')
        logger.info("preprocessing article texts...")
        if 'Title' in df.columns:
            df = preprocessor.preprocess_dataframe(df, 'Title')
        if 'Content' in df.columns:
            df = preprocessor.preprocess_dataframe(df, 'Content')
        logger.info("preprocessing completed")
        return df
    except Exception as e:
        logger.error(f"error preprocessing texts: {str(e)}")
        return None

def apply_models(df):
    try:
        sentiment_model_path = os.path.join(project_root, 'models', 'sentiment_analyzer')
        logger.info(f"Resolved sentiment model path: {sentiment_model_path}")
        from models.sentiment_analyzer.sentiment_analyzer import SentimentAnalyzer
        try:
            logger.info("loading sentiment analyzer...")
            sentiment_model = SentimentAnalyzer.load_model(sentiment_model_path)
            logger.info("applying sentiment analysis...")
            combined_text = df['Title'] + ' ' + df['Content']
            sentiment_ids = sentiment_model.predict(combined_text)
            df['sentiment'] = [sentiment_model.labels[id] for id in sentiment_ids]
            features = sentiment_model.extract_sentiment_features(combined_text)
            df['positive_words'] = features['positive_word_count'].values
            df['negative_words'] = features['negative_word_count'].values
            df['sentiment_ratio'] = features['sentiment_ratio'].values
            logger.info("sentiment analysis completed")
        except Exception as e:
            logger.error(f"error in sentiment analysis: {str(e)}")
            df['sentiment'] = "neutral"
            df['positive_words'] = 0
            df['negative_words'] = 0
            df['sentiment_ratio'] = 1.0
        logger.info("model application completed")
        return df
    except Exception as e:
        logger.error(f"error applying models: {str(e)}")
        return df

def save_processed_data(df, original_file_name):
    try:
        base_name = os.path.basename(original_file_name)
        processed_name = f"processed_{base_name}"
        processed_path = os.path.join(processed_dir, processed_name)
        df.to_csv(processed_path.replace('.json', '.csv'), index=False, encoding='utf-8')
        logger.info(f"saved processed data to {processed_path.replace('.json', '.csv')}")
        df_json = df.copy()
        for col in df_json.columns:
            if df_json[col].dtype == 'datetime64[ns]':
                df_json[col] = df_json[col].dt.strftime('%Y-%m-%d %H:%M:%S')
        df_json.to_json(processed_path, orient='records', force_ascii=False, indent=2)
        logger.info(f"saved processed data to {processed_path}")
        return True
    except Exception as e:
        logger.error(f"error saving processed data: {str(e)}")
        return False

def create_display_file(df):
    try:
        display_df = df[['Title', 'Content', 'Source', 'Category', 
                         'sentiment', 'PublishDate', 'ArticleUrl', 'ArticleLength', 'WordCount']]
        if 'Id' not in display_df.columns:
            display_df['Id'] = range(1, len(display_df) + 1)
        display_dir = os.path.join(project_root, 'data', 'display')
        os.makedirs(display_dir, exist_ok=True)
        today = datetime.now().strftime('%Y-%m-%d')
        display_path = os.path.join(display_dir, f"display_articles_{today}.json")
        display_df_json = display_df.copy()
        for col in display_df_json.columns:
            if display_df_json[col].dtype == 'datetime64[ns]':
                display_df_json[col] = display_df_json[col].dt.strftime('%Y-%m-%d %H:%M:%S')
        display_df_json.to_json(display_path, orient='records', force_ascii=False, indent=2)
        logger.info(f"saved display data to {display_path}")
        display_df.to_csv(display_path.replace('.json', '.csv'), index=False, encoding='utf-8')
        try:
            all_display_files = glob.glob(os.path.join(display_dir, "display_articles_*.json"))
            all_data = []
            for file in all_display_files:
                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    all_data.extend(data)
            combined_path = os.path.join(display_dir, "all_articles.json")
            with open(combined_path, 'w', encoding='utf-8') as f:
                json.dump(all_data, f, ensure_ascii=False, indent=2)
            logger.info(f"saved combined display data to {combined_path}")
        except Exception as e:
            logger.error(f"error creating combined display file: {str(e)}")
        return True
    except Exception as e:
        logger.error(f"error creating display file: {str(e)}")
        return False

def main():
    logger.info("starting scraped data processing")
    latest_file = find_latest_scraped_data()
    if not latest_file:
        logger.error("no scraped data found, exiting")
        return
    articles = load_scraped_data(latest_file)
    if not articles:
        logger.error("failed to load scraped data, exiting")
        return
    df = preprocess_texts(articles)
    if df is None:
        logger.error("text preprocessing failed, exiting")
        return
    df = apply_models(df)
    save_processed_data(df, latest_file)
    create_display_file(df)
    logger.info("scraped data processing completed")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"unexpected error: {str(e)}")
