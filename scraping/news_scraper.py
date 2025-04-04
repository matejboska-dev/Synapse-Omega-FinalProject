import requests
from bs4 import BeautifulSoup
import feedparser
import datetime
import time
import random
import re
import pyodbc
from tqdm import tqdm
import logging
from urllib.parse import urlparse
import concurrent.futures
import os
import json

# Ensure logs directory exists
logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(logs_dir, exist_ok=True)

# Setup logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(logs_dir, "scraper.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("synapse-scraper")

# Database connection configuration
DB_SERVER = "193.85.203.188"
DB_NAME = "boska"
DB_USER = "boska"
DB_PASSWORD = "123456"

# Ensure data directory exists
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
os.makedirs(data_dir, exist_ok=True)

# News sources and their RSS feeds
news_sources = {
    "idnes": "https://servis.idnes.cz/rss.aspx?c=zpravodaj",
    "novinky": "https://www.novinky.cz/rss",
    "seznamzpravy": "https://www.seznamzpravy.cz/rss",
    "aktualne": "https://zpravy.aktualne.cz/rss/",
    "ihned": "https://ihned.cz/rss/",
    "denik-n": "https://denikn.cz/feed/",
    "ct24": "https://ct24.ceskatelevize.cz/rss/hlavni-zpravy",
    "irozhlas": "https://www.irozhlas.cz/rss/irozhlas/",
    "denik": "https://www.denik.cz/rss/all.html",
    "lidovky": "https://servis.lidovky.cz/rss.aspx?c=ln_domov",
    "reflex": "https://www.reflex.cz/rss",
    "echo24": "https://echo24.cz/rss",
    "info-cz": "https://www.info.cz/feed/rss",
    "forum24": "https://www.forum24.cz/feed/",
    "blesk": "https://www.blesk.cz/rss",
    "cnn-iprima": "https://cnn.iprima.cz/rss",
    "parlamentnilisty": "https://www.parlamentnilisty.cz/export/rss.aspx",
    "eurozpravy": "https://eurozpravy.cz/rss/",
    "tyden": "https://www.tyden.cz/rss/",
    "e15": "https://www.e15.cz/rss"
}

# Additional category mappings to help with classification
category_mappings = {
    "politika": "Politika",
    "domácí": "Domácí",
    "zahraničí": "Zahraničí",
    "ekonomika": "Ekonomika",
    "byznys": "Ekonomika",
    "finance": "Ekonomika",
    "sport": "Sport",
    "fotbal": "Sport",
    "hokej": "Sport",
    "tenis": "Sport",
    "kultura": "Kultura",
    "film": "Kultura",
    "hudba": "Kultura",
    "divadlo": "Kultura",
    "zdraví": "Zdraví",
    "věda": "Věda a technologie",
    "tech": "Věda a technologie",
    "technologie": "Věda a technologie",
    "auto": "Auto-moto",
    "cestování": "Cestování",
    "životní styl": "Životní styl",
    "komentáře": "Komentáře a názory"
}

# Function to connect to the database
def connect_to_db():
    """
    Create a connection to SQL Server database.
    Returns:
        pyodbc.Connection: Database connection object or None if error
    """
    try:
        # Try to load config from file first
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config', 'db_config.json')
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
        
        conn_str = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}"
        conn = pyodbc.connect(conn_str)
        logger.info("Successfully connected to database.")
        return conn
    except Exception as e:
        logger.error(f"Error connecting to database: {e}")
        return None

# Function to extract article text with retry logic
def get_article_text(url, source, max_retries=3):
    """
    Extract article text from a given URL.
    
    Args:
        url (str): Article URL
        source (str): Article source (e.g., "idnes", "novinky")
        max_retries (int): Maximum number of retry attempts
    
    Returns:
        tuple: (article_text, char_count, word_count, extracted_category)
    """
    for retry in range(max_retries):
        try:
            # Add random delay to reduce server load
            time.sleep(random.uniform(0.5, 2))
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "cs,en-US;q=0.7,en;q=0.3",
                "Referer": "https://www.google.com/",
                "Cache-Control": "no-cache"  # Added to avoid cached responses
            }
            
            response = requests.get(url, headers=headers, timeout=15)
            if response.status_code != 200:
                logger.warning(f"Attempt {retry+1}/{max_retries}: Failed to download article: {url}, status code: {response.status_code}")
                if retry == max_retries - 1:
                    return "Failed to download article", 0, 0, ""
                continue
            
            soup = BeautifulSoup(response.content, "html.parser")
            
            # Remove unwanted elements
            for element in soup(["script", "style", "nav", "footer", "header", "aside", "iframe", "noscript", "form", "button"]):
                element.decompose()
            
            # Detect source from domain if not explicitly specified
            if not source or source == "":
                parsed_url = urlparse(url)
                domain = parsed_url.netloc
                # Simplified source detection based on domain
                for src_name in news_sources:
                    if src_name.lower() in domain.lower():
                        source = src_name
                        break
            
            # Extract category from breadcrumbs or other navigation elements
            extracted_category = ""
            category_elements = soup.find_all(['span', 'a', 'div'], class_=lambda x: x and any(word in (x.lower() if x else "") for word in 
                                                                                         ["breadcrumb", "category", "rubrika", "sekce", "tema"]))
            
            for element in category_elements:
                text = element.get_text().strip().lower()
                
                # Try to match with predefined categories
                for key, value in category_mappings.items():
                    if key in text:
                        extracted_category = value
                        break
                
                # If category found, break
                if extracted_category:
                    break
            
            # Try special page structure approaches if no category found
            if not extracted_category:
                # Look for meta tags
                meta_category = soup.find('meta', {'property': 'article:section'}) or soup.find('meta', {'name': 'category'})
                if meta_category and meta_category.get('content'):
                    cat_text = meta_category.get('content').lower()
                    for key, value in category_mappings.items():
                        if key in cat_text:
                            extracted_category = value
                            break
            
            # Universal text extraction using selectors for different sources
            selectors = {
                "idnes": {"div": ["article-body", "art-full"]},
                "novinky": {"div": ["articleBody", "article_text", "b-content"]},
                "seznamzpravy": {"div": ["article-body", "sznp-article-body", "e_1xnl"]},
                "aktualne": {"div": ["article-text", "clanek", "text"]},
                "ihned": {"div": ["article-body", "clanek", "detail__body"]},
                "denik-n": {"div": ["post-content", "a_content"]},
                "ct24": {"div": ["article-body", "article_text", "article-detail"]},
                "irozhlas": {"div": ["b-detail", "article", "main-content"]},
                "denik": {"div": ["article-body", "clanek", "article__body"]},
                "lidovky": {"div": ["article-body", "leadtext"]},
                "reflex": {"div": ["article-content"]},
                "echo24": {"div": ["article-detail__content", "article_body"]},
                "info-cz": {"div": ["article-body"]},
                "forum24": {"div": ["entry-content"]},
                "blesk": {"div": ["o-article__content", "article-content"]},
                "cnn-iprima": {"div": ["article-content"]},
                "parlamentnilisty": {"div": ["article-body", "article-full"]},
                "eurozpravy": {"div": ["article-content"]},
                "tyden": {"div": ["text-block"]},
                "e15": {"div": ["article-body", "entry-content"]}
            }
            
            # Extract text based on source
            article_text = ""
            if source in selectors:
                for element_type, class_names in selectors[source].items():
                    for class_name in class_names:
                        article_div = soup.find(element_type, class_=class_name)
                        if article_div:
                            # Get all paragraphs, headings, and list items
                            elements = article_div.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li'])
                            if elements:
                                article_text = " ".join([el.get_text().strip() for el in elements])
                                break
                            else:
                                # If no structured elements found, get all text
                                article_text = article_div.get_text(separator=' ', strip=True)
                    if article_text:
                        break
            
            # General method as fallback
            if not article_text:
                # Attempt 1: Search by typical class names
                for class_name in ["article-body", "article-content", "post-content", "news-content", 
                                  "story-content", "main-content", "entry-content", "article", "clanek", 
                                  "text", "content", "body-content"]:
                    article_div = soup.find(["div", "article", "section", "main"], 
                                           class_=lambda x: x and class_name in x.lower())
                    if article_div:
                        # Get all text elements
                        elements = article_div.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li'])
                        if elements:
                            article_text = " ".join([el.get_text().strip() for el in elements])
                        else:
                            article_text = article_div.get_text(separator=' ', strip=True)
                        break
                
                # Attempt 2: Search by typical HTML elements
                if not article_text:
                    main_content = (soup.find("main") or 
                                   soup.find("article") or 
                                   soup.find("div", class_=lambda x: x and any(c in (x.lower() if x else "") 
                                                                              for c in ["content", "article", "text", "body"])))
                    if main_content:
                        elements = main_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li'])
                        if elements:
                            article_text = " ".join([el.get_text().strip() for el in elements])
                        else:
                            article_text = main_content.get_text(separator=' ', strip=True)
                    else:
                        # Last attempt - get everything from body except excluded elements
                        body = soup.find("body")
                        if body:
                            article_text = body.get_text(separator=" ", strip=True)
            
            # Text cleaning
            article_text = re.sub(r'\s+', ' ', article_text).strip()
            article_text = re.sub(r'[^\w\s.,?!;:()\[\]{}"\'–—-]', '', article_text)
            
            # If article has a title but we didn't find a title in the text, use the title from response
            title_tag = soup.find('title')
            if title_tag and len(article_text) < 50 and title_tag.text:
                article_text = title_tag.text + " " + article_text
            
            # Count characters and words
            char_count = len(article_text)
            word_count = len(article_text.split())
            
            # Accept articles with very little text (50 chars) if that's all we can get
            # This is changed from previous 100 char minimum
            if char_count < 50:
                logger.warning(f"Very little text extracted from {url}: only {char_count} characters")
                if retry < max_retries - 1:
                    continue
            
            return article_text, char_count, word_count, extracted_category
            
        except Exception as e:
            logger.error(f"Attempt {retry+1}/{max_retries}: Error extracting text from {url}: {e}")
            if retry == max_retries - 1:
                return f"Error: {e}", 0, 0, ""
            # Increase wait time between retries to reduce chance of being blocked
            time.sleep(random.uniform(3, 5))
    
    return "Failed to extract text after repeated attempts", 0, 0, ""

# Function to save article to database
def save_article_to_db(conn, source_name, title, url, pub_date, category, char_count, word_count, article_text):
    """
    Save article to database.
    
    Args:
        conn: Database connection
        source_name: Article source name
        title: Article title
        url: Article URL
        pub_date: Publication date
        category: Article category
        char_count: Character count
        word_count: Word count
        article_text: Article text
    
    Returns:
        bool: True on success, False on failure
    """
    try:
        cursor = conn.cursor()
        
        # Handle overly long values
        if len(title) > 500:
            title = title[:497] + "..."
        if len(url) > 1000:
            url = url[:997] + "..."
        if len(source_name) > 255:
            source_name = source_name[:252] + "..."
        if category and len(category) > 255:
            category = category[:252] + "..."
        
        # Current date and time for ScrapedDate field
        scraped_date = datetime.datetime.now()
        
        # Insert query
        sql = """
        INSERT INTO Articles (SourceName, Title, ArticleUrl, PublicationDate, Category, 
                              ArticleLength, WordCount, ArticleText, ScrapedDate)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        cursor.execute(sql, (source_name, title, url, pub_date, category, 
                             char_count, word_count, article_text, scraped_date))
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"Error saving article to database: {e}")
        try:
            conn.rollback()
        except:
            pass
        
        # Save to local file if database fails
        try:
            local_file = os.path.join(data_dir, f"article_{int(time.time())}.json")
            with open(local_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "source": source_name,
                    "title": title,
                    "url": url,
                    "date": str(pub_date) if pub_date else None,
                    "category": category,
                    "length": char_count,
                    "words": word_count,
                    "text": article_text,
                    "scraped": str(scraped_date)
                }, f, ensure_ascii=False)
            logger.info(f"Saved article to local file: {local_file}")
        except Exception as backup_e:
            logger.error(f"Also failed to save article locally: {backup_e}")
            
        return False

# Function to check if article already exists in database
def article_exists(conn, url, title):
    """
    Check if article already exists in database by URL or title.
    
    Args:
        conn: Database connection
        url: Article URL
        title: Article title
    
    Returns:
        bool: True if article exists, False otherwise
    """
    try:
        cursor = conn.cursor()
        # Just check URL to avoid false positives with titles
        cursor.execute("SELECT COUNT(*) FROM Articles WHERE ArticleUrl = ?", (url,))
        count = cursor.fetchone()[0]
        return count > 0
    except Exception as e:
        logger.error(f"Error checking article existence: {e}")
        return False

# Function to parse date from RSS
def parse_date(date_str):
    """
    Convert RSS date string to datetime object.
    
    Args:
        date_str: Date string
    
    Returns:
        datetime: Datetime object or None
    """
    if not date_str:
        return None
    
    try:
        # RSS dates can have various formats
        for date_format in [
            "%a, %d %b %Y %H:%M:%S %z",      # Standard RSS format
            "%a, %d %b %Y %H:%M:%S %Z",      # Variant with text timezone
            "%a, %d %b %Y %H:%M:%S GMT",     # Variant without timezone
            "%Y-%m-%dT%H:%M:%S%z",           # ISO format
            "%Y-%m-%dT%H:%M:%S%Z",           # ISO with text timezone
            "%Y-%m-%dT%H:%M:%SZ",            # ISO without timezone
            "%d.%m.%Y %H:%M:%S",             # Czech format
            "%d.%m.%Y",                      # Shorter Czech format
        ]:
            try:
                return datetime.datetime.strptime(date_str, date_format)
            except ValueError:
                continue
                
        return datetime.datetime.now()  # Fallback if no format matches
    except:
        return datetime.datetime.now()

# Function to get article count in database
def get_article_count(conn):
    """
    Get number of articles in database.
    
    Args:
        conn: Database connection
    
    Returns:
        int: Number of articles in database
    """
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM Articles")
        count = cursor.fetchone()[0]
        return count
    except Exception as e:
        logger.error(f"Error getting article count: {e}")
        return 0

# Function to process articles from one source
def process_source(source_name, rss_url, conn, current_count, target_count, max_articles_per_source, new_articles_added):
    """
    Process articles from one source.
    
    Args:
        source_name: Source name
        rss_url: RSS feed URL
        conn: Database connection
        current_count: Current article count
        target_count: Target article count
        max_articles_per_source: Max articles per source
        new_articles_added: Reference variable for tracking newly added articles
    
    Returns:
        int: Number of newly added articles from this source
    """
    source_articles = 0
    
    try:
        logger.info(f"Processing RSS feed: {source_name}")
        feed = feedparser.parse(rss_url)
        
        if not feed.entries:
            logger.warning(f"No articles found in RSS feed: {source_name}")
            
            # Try again with a different approach if feed fails
            alt_feeds = {
                "idnes": "https://www.idnes.cz/zpravy/rss.axd",
                "novinky": "https://www.novinky.cz/rss2/",
                "aktualne": "https://zpravy.aktualne.cz/rss/",
                "ihned": "https://ihned.cz/?m=rss",
                "denik": "https://www.denik.cz/rss/",
                "seznamzpravy": "https://www.seznamzpravy.cz/rss/seznam-zpravy-18"
            }
            
            if source_name in alt_feeds:
                logger.info(f"Trying alternative feed for {source_name}")
                feed = feedparser.parse(alt_feeds[source_name])
                if not feed.entries:
                    logger.warning(f"Alternative feed also returned no articles for {source_name}")
                    return 0
            else:
                return 0
        
        # Iterate through articles in RSS feed
        for entry in tqdm(feed.entries, desc=f"Articles from {source_name}"):
            # Check if we already have enough articles
            if current_count + new_articles_added.value >= target_count:
                logger.info(f"Reached target article count: {current_count + new_articles_added.value}")
                break
                
            # Check if we have enough articles from this source
            if source_articles >= max_articles_per_source:
                logger.info(f"Reached maximum articles from source {source_name}: {source_articles}")
                break
            
            # Get basic data from RSS
            title = entry.title if hasattr(entry, 'title') else "Untitled"
            url = entry.link if hasattr(entry, 'link') else ""
            
            if not url:
                continue
            
            # Check if article already exists in database
            if article_exists(conn, url, title):
                logger.debug(f"Article already exists in database: {title}")
                continue
            
            # Process publication date
            pub_date = None
            if hasattr(entry, 'published'):
                pub_date = parse_date(entry.published)
            elif hasattr(entry, 'pubDate'):
                pub_date = parse_date(entry.pubDate)
            elif hasattr(entry, 'updated'):
                pub_date = parse_date(entry.updated)
            
            # Try to get category
            category = ""
            if hasattr(entry, 'tags') and entry.tags:
                try:
                    category = entry.tags[0].term
                except:
                    try:
                        category = entry.tags[0]['term']
                    except:
                        category = ""
            elif hasattr(entry, 'category'):
                category = entry.category
                
            # Map category to standard categories if possible
            category_lower = category.lower()
            for key, value in category_mappings.items():
                if key in category_lower:
                    category = value
                    break
            
            # Get full article text
            logger.info(f"Downloading article: {title}")
            article_text, char_count, word_count, extracted_category = get_article_text(url, source_name)
            
            # Use extracted category if we didn't get one from RSS
            if not category and extracted_category:
                category = extracted_category
                
            # Accept articles with very little text (30 chars) if that's all we can get
            # MODIFIED: Accept all articles even with minimal text
            if char_count < 30:
                logger.warning(f"Article has very short text: {title} ({char_count} chars), but still accepting it")
            
            # Add title to text if text is short
            if char_count < 100 and title and len(title) > 10:
                article_text = title + ". " + article_text
                char_count = len(article_text)
                word_count = len(article_text.split())
            
            # Save article to database
            success = save_article_to_db(
                conn, source_name, title, url, pub_date, category, 
                char_count, word_count, article_text
            )
            
            if success:
                with new_articles_added.get_lock():
                    new_articles_added.value += 1
                source_articles += 1
                logger.info(f"Article saved: {title} ({char_count} chars, {word_count} words)")
            
            # Progress info
            if source_articles % 5 == 0:
                logger.info(f"Total new articles downloaded: {new_articles_added.value}")
            
            # Short pause between articles to reduce load
            time.sleep(random.uniform(0.5, 1.5))
        
        return source_articles
        
    except Exception as e:
        logger.error(f"Error processing source {source_name}: {e}")
        return source_articles

# Main function for collecting articles
def collect_news(target_count=3000, max_articles_per_source=300):
    """
    Collect articles from various sources until target count is reached.
    
    Args:
        target_count: Target total article count
        max_articles_per_source: Max articles per source
    
    Returns:
        int: Number of newly added articles
    """
    # Connect to database
    conn = connect_to_db()
    if not conn:
        logger.error("Cannot continue without database connection.")
        return 0
    
    # Get current article count
    current_count = get_article_count(conn)
    logger.info(f"Current article count in database: {current_count}")
    
    # If we already have enough articles, exit
    if current_count >= target_count:
        logger.info(f"Already have enough articles ({current_count}/{target_count}), no need to download more.")
        conn.close()
        return 0
    
    # Initialize shared variable for counting new articles
    import multiprocessing
    new_articles_added = multiprocessing.Value('i', 0)
    
    try:
        # Process sources sequentially (parallel would cause database issues)
        for source_name, rss_url in news_sources.items():
            # Check if we already have enough articles
            if current_count + new_articles_added.value >= target_count:
                logger.info(f"Reached target article count: {current_count + new_articles_added.value}")
                break
            
            process_source(source_name, rss_url, conn, current_count, target_count, 
                           max_articles_per_source, new_articles_added)
    
    finally:
        if conn:
            conn.close()
            logger.info("Database connection closed.")
    
    logger.info(f"Completed! Added {new_articles_added.value} new articles.")
    logger.info(f"Current total article count in database: {current_count + new_articles_added.value}")
    return new_articles_added.value

# Save data locally to avoid database issues
def save_to_local_json(articles, filename=None):
    """Save articles to local JSON file"""
    if not filename:
        filename = f"articles_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    file_path = os.path.join(data_dir, filename)
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(articles)} articles to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving articles to file: {e}")
        return False

# Main program execution
if __name__ == "__main__":
    logger.info("Starting news article collection for Synapse project...")
    
    # Target article count
    target_count = 3000
    
    # Maximum articles per source (for more balanced distribution)
    max_per_source = 300
    
    # Continuously run until target count is reached
    attempt = 0
    max_attempts = 20  # Limit number of attempts
    
    while attempt < max_attempts:
        attempt += 1
        current_count = get_article_count(connect_to_db())
        if current_count >= target_count:
            logger.info(f"Target article count reached: {current_count}/{target_count}")
            break
            
        logger.info(f"Continuing collection, currently have {current_count}/{target_count} articles (attempt {attempt}/{max_attempts})")
        collected = collect_news(target_count=target_count, max_articles_per_source=max_per_source)
        
        if collected == 0:
            logger.warning("No new articles added, may have exhausted available sources.")
            # Wait 10 minutes before next attempt (reduced from 30)
            logger.info("Waiting 10 minutes before next attempt...")
            time.sleep(600)  # 10 minutes
    
    logger.info(f"Collection completed, target article count reached: {get_article_count(connect_to_db())}")