import sys
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime
import json

# add parent directory to system path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ensure log directory exists
log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"data_preparation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# absolute imports from src directory
from data.database_connector import DatabaseConnector
from data.data_analyzer import DataAnalyzer
from data.text_preprocessor import TextPreprocessor

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

# custom JSON encoder to handle numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)

def main():
    """
    main function for data analysis and preprocessing
    """
    # create output directories if they don't exist
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    for directory in ['data/processed', 'data/raw', 'reports/figures']:
        dir_path = os.path.join(project_root, directory)
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"ensuring directory exists: {dir_path}")
    
    # connect to database and load articles
    logger.info("== connecting to database and loading data ==")
    
    # connection parameters - default values
    server = "193.85.203.188"
    database = "boska"
    username = "boska"
    password = "123456"
    
    # try to load configuration
    try:
        config_path = os.path.join(project_root, 'config', 'database.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                server = config.get('server', server)
                database = config.get('database', database)
                username = config.get('username', username)
                password = config.get('password', password)
            logger.info("configuration loaded from config/database.json")
        else:
            logger.warning("configuration file not found, using default values")
    except Exception as e:
        logger.warning(f"error loading configuration: {str(e)}, using default values")
    
    # create connector instance and connect to database
    db = DatabaseConnector(server, database, username, password)
    if not db.connect():
        logger.error("cannot connect to database, exiting")
        return
    
    # load articles
    df = db.load_articles()
    if df is None or df.empty:
        logger.error("failed to load data, exiting")
        db.disconnect()
        return
    
    # rename columns to match expected names if needed
    column_mapping = {
        'SourceName': 'Source',
        'ArticleText': 'Content',
        'PublicationDate': 'PublishDate'
    }
    
    # rename only columns that exist
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns and new_col not in df.columns:
            df.rename(columns={old_col: new_col}, inplace=True)
    
    # basic information about the data
    logger.info(f"loaded {len(df)} articles with {len(df.columns)} columns")
    logger.info(f"columns: {', '.join(df.columns)}")
    
    # save raw data
    raw_data_path = os.path.join(project_root, 'data', 'raw', 'articles_raw.csv')
    df.to_csv(raw_data_path, index=False, encoding='utf-8')
    logger.info(f"raw data saved to {raw_data_path}")
    
    # data analysis
    logger.info("== analyzing data ==")
    analyzer = DataAnalyzer(df)
    stats = analyzer.compute_basic_stats()
    
    # print basic statistics
    logger.info(f"total number of articles: {stats['total_articles']}")
    
    if 'articles_by_source' in stats:
        logger.info(f"number of sources: {stats['num_sources']}")
        for source, count in sorted(stats['articles_by_source'].items(), key=lambda x: x[1], reverse=True)[:5]:
            logger.info(f"  {source}: {count} articles")
    
    if 'articles_by_category' in stats:
        logger.info(f"number of categories: {stats['num_categories']}")
        for category, count in sorted(stats['articles_by_category'].items(), key=lambda x: x[1], reverse=True)[:5]:
            logger.info(f"  {category}: {count} articles")
    
    if 'date_range' in stats:
        logger.info(f"date range: {stats['date_range']['min']} to {stats['date_range']['max']}")
    
    if 'content_length' in stats:
        logger.info(f"average article length: {stats['content_length']['mean']:.1f} characters")
        logger.info(f"median article length: {stats['content_length']['median']:.1f} characters")
    
    if 'word_count' in stats:
        logger.info(f"average word count: {stats['word_count']['mean']:.1f}")
        logger.info(f"median word count: {stats['word_count']['median']:.1f}")
    
    # save statistics
    reports_dir = os.path.join(project_root, 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    stats_path = os.path.join(reports_dir, 'data_stats.json')
    
    # use custom encoder to handle numpy types
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
    logger.info(f"statistics saved to {stats_path}")
    
    # visualize basic statistics
    figures_dir = os.path.join(reports_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    analyzer.visualize_basic_stats(output_dir=figures_dir)
    
    # text preprocessing
    logger.info("== preprocessing text ==")
    preprocessor = TextPreprocessor(language='czech')
    
    # preprocess titles and contents
    if 'Title' in df.columns:
        df = preprocessor.preprocess_dataframe(df, 'Title')
    
    if 'Content' in df.columns:
        df = preprocessor.preprocess_dataframe(df, 'Content')
    
    # save preprocessed data
    processed_dir = os.path.join(project_root, 'data', 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    processed_data_path = os.path.join(processed_dir, 'articles_processed.csv')
    df.to_csv(processed_data_path, index=False, encoding='utf-8')
    logger.info(f"preprocessed data saved to {processed_data_path}")
    
    # disconnect from database
    db.disconnect()
    
    logger.info("data analysis and preprocessing completed")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("unexpected error: %s", str(e))