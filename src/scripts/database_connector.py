import pyodbc
import pandas as pd
import logging
import os
import json
from datetime import datetime

# Database connection parameters matching scraper.py
DB_SERVER = "193.85.203.188"
DB_NAME = "boska"
DB_USER = "boska"
DB_PASSWORD = "123456"

# Set up logging
logger = logging.getLogger(__name__)

class DatabaseConnector:
    """Database connector for connecting to SQL Server database"""
    
    def __init__(self, server=DB_SERVER, database=DB_NAME, username=DB_USER, password=DB_PASSWORD):
        """
        Initialize database connector with connection parameters
        
        Args:
            server (str): Database server address
            database (str): Database name
            username (str): Username
            password (str): Password
        """
        self.server = server
        self.database = database
        self.username = username
        self.password = password
        self.conn = None
    
    def connect(self):
        """
        Establish connection to the database
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            # Connection string identical to the one in scraper.py
            conn_str = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={self.server};DATABASE={self.database};UID={self.username};PWD={self.password}"
            self.conn = pyodbc.connect(conn_str)
            logger.info(f"Successfully connected to database {self.database} on {self.server}")
            return True
        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
            return False
    
    def disconnect(self):
        """
        Close the database connection
        
        Returns:
            bool: True if disconnection successful, False otherwise
        """
        try:
            if self.conn:
                self.conn.close()
                self.conn = None
                logger.info("Database connection closed")
            return True
        except Exception as e:
            logger.error(f"Error disconnecting from database: {e}")
            return False

    def create_tables_if_not_exist(self):
        """
        Create necessary tables if they don't exist
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.conn and not self.connect():
                return False
                
            cursor = self.conn.cursor()
            
            # Check and create Articles table if it doesn't exist
            cursor.execute("""
            IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'Articles')
            CREATE TABLE Articles (
                Id INT IDENTITY(1,1) PRIMARY KEY,
                SourceName NVARCHAR(255),
                Title NVARCHAR(500),
                ArticleUrl NVARCHAR(1000),
                PublicationDate DATETIME,
                Category NVARCHAR(255),
                ArticleLength INT,
                WordCount INT,
                ArticleText NVARCHAR(MAX),
                ScrapedDate DATETIME
            )
            """)
            self.conn.commit()
            
            logger.info("Database tables created/verified successfully")
            return True
        except Exception as e:
            logger.error(f"Error creating tables: {e}")
            return False

    def article_exists(self, url, title=None):
        """
        Check if article already exists in database by URL or title
        
        Args:
            url (str): Article URL
            title (str, optional): Article title
            
        Returns:
            bool: True if article exists, False otherwise
        """
        try:
            if not self.conn and not self.connect():
                return False
                
            cursor = self.conn.cursor()
            
            # Check by URL (primary method)
            cursor.execute("SELECT COUNT(*) FROM Articles WHERE ArticleUrl = ?", (url,))
            count = cursor.fetchone()[0]
            
            # If title is provided, also check by title
            if count == 0 and title:
                cursor.execute("SELECT COUNT(*) FROM Articles WHERE Title = ?", (title,))
                count += cursor.fetchone()[0]
                
            return count > 0
        except Exception as e:
            logger.error(f"Error checking article existence: {e}")
            return False
    
    def get_article_count(self):
        """
        Get number of articles in database
        
        Returns:
            int: Number of articles or 0 if error
        """
        try:
            if not self.conn and not self.connect():
                return 0
                
            cursor = self.conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM Articles")
            count = cursor.fetchone()[0]
            return count
        except Exception as e:
            logger.error(f"Error getting article count: {e}")
            return 0
    
    def load_articles(self, limit=None):
        """
        Load articles from database
        
        Args:
            limit (int, optional): Maximum number of articles to load
            
        Returns:
            pd.DataFrame: DataFrame with articles or None if error
        """
        try:
            if not self.conn and not self.connect():
                return None
                
            # Base query
            query = """
            SELECT
                Id, 
                SourceName as Source, 
                Title, 
                ArticleUrl, 
                PublicationDate as PublishDate, 
                Category, 
                ArticleLength, 
                WordCount, 
                ArticleText as Content, 
                ScrapedDate
            FROM Articles
            """
            
            # Add limit if specified
            if limit:
                query += f" ORDER BY PublicationDate DESC OFFSET 0 ROWS FETCH NEXT {limit} ROWS ONLY"
                
            df = pd.read_sql(query, self.conn)
            logger.info(f"Loaded {len(df)} articles from database")
            return df
        except Exception as e:
            logger.error(f"Error loading articles: {e}")
            return None
    
    def save_article(self, source_name, title, url, pub_date, category, char_count, word_count, article_text):
        """
        Save article to database (matching the format in scraper.py)
        
        Args:
            source_name (str): Article source
            title (str): Article title
            url (str): Article URL
            pub_date (datetime): Publication date
            category (str): Article category
            char_count (int): Character count
            word_count (int): Word count
            article_text (str): Article text
            
        Returns:
            bool: True if article saved successfully, False otherwise
        """
        try:
            if not self.conn and not self.connect():
                return False
                
            # Check if article already exists
            if self.article_exists(url, title):
                logger.info(f"Article already exists in database: {title}")
                return True
                
            # Handle field length limitations as in scraper.py
            if len(title) > 500:
                title = title[:497] + "..."
            if len(url) > 1000:
                url = url[:997] + "..."
            if len(source_name) > 255:
                source_name = source_name[:252] + "..."
            if category and len(category) > 255:
                category = category[:252] + "..."
            
            # Current date and time for ScrapedDate field
            scraped_date = datetime.now()
            
            # Execute insert query
            cursor = self.conn.cursor()
            sql = """
            INSERT INTO Articles (SourceName, Title, ArticleUrl, PublicationDate, Category, 
                                 ArticleLength, WordCount, ArticleText, ScrapedDate)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            cursor.execute(sql, (source_name, title, url, pub_date, category, 
                               char_count, word_count, article_text, scraped_date))
            self.conn.commit()
            
            logger.info(f"Article saved to database: {title}")
            return True
        except Exception as e:
            logger.error(f"Error saving article to database: {e}")
            try:
                self.conn.rollback()
            except:
                pass
            return False