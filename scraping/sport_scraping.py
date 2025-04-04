import requests
from bs4 import BeautifulSoup
import feedparser
import datetime
import time
import random
import re
import logging
import os

# Setup logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("czech-sport-scraper")

# Ensure data directory exists
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
os.makedirs(data_dir, exist_ok=True)

# Czech sports news sources and their RSS feeds
sport_news_sources = {
    "sport.cz": "https://rsss.sport.cz/rss",
    "isport": "https://isport.blesk.cz/rss/",
    "efotbal": "https://www.efotbal.cz/rss/",
    "sportrevue": "https://www.sportrevue.cz/feed/",
    "nhl.cz": "https://nhl.cz/rss/",
    "cslh.cz": "https://www.cslh.cz/feed/"
}

def get_article_text(url, max_retries=3):
    for retry in range(max_retries):
        try:
            time.sleep(random.uniform(0.5, 2))
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "cs,en-US;q=0.7,en;q=0.3",
                "Cache-Control": "no-cache"
            }
            
            response = requests.get(url, headers=headers, timeout=15)
            if response.status_code != 200:
                logger.warning(f"Attempt {retry+1}/{max_retries}: Failed to download article: {url}, status code: {response.status_code}")
                if retry == max_retries - 1:
                    return "Failed to download article", 0
                continue
            
            soup = BeautifulSoup(response.content, "html.parser")
            
            for element in soup(["script", "style", "nav", "footer", "header", "aside", "iframe", "form"]):
                element.decompose()
            
            article_div = soup.find("div", class_=lambda x: x and any(c in (x.lower() if x else "") for c in ["content", "article", "clanek", "text", "body"]))
            if article_div:
                article_text = article_div.get_text(separator=' ', strip=True)
            else:
                article_text = soup.get_text(separator=' ', strip=True)
            
            article_text = re.sub(r'\s+', ' ', article_text).strip()
            char_count = len(article_text)
            
            if char_count < 100:
                logger.warning(f"Very little text extracted from {url}: only {char_count} characters")
                if retry < max_retries - 1:
                    continue
            
            return article_text, char_count
            
        except Exception as e:
            logger.error(f"Attempt {retry+1}/{max_retries}: Error extracting text from {url}: {e}")
            if retry == max_retries - 1:
                return f"Error: {e}", 0
            time.sleep(random.uniform(3, 5))
    
    return "Failed to extract text after repeated attempts", 0

def save_article_to_file(source_name, title, url, pub_date, article_text, char_count):
    try:
        filename = re.sub(r'[^\w\-_\. ]', '_', f"{source_name}_{title}")[:100] + ".txt"
        file_path = os.path.join(data_dir, filename)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"Source: {source_name}\n")
            f.write(f"Title: {title}\n")
            f.write(f"URL: {url}\n")
            f.write(f"Date: {pub_date}\n")
            f.write(f"Length: {char_count} characters\n\n")
            f.write(article_text)
        
        logger.info(f"Article saved to file: {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving article to file: {e}")
        return False

def collect_sport_news(target_count=100):
    new_articles_added = 0
    
    for source_name, rss_url in sport_news_sources.items():
        if new_articles_added >= target_count:
            logger.info(f"Reached target article count: {new_articles_added}")
            break
        
        try:
            logger.info(f"Processing RSS feed: {source_name}")
            feed = feedparser.parse(rss_url)
            
            if not feed.entries:
                logger.warning(f"No articles found in RSS feed: {source_name}")
                continue
            
            for entry in feed.entries[:target_count-new_articles_added]:
                title = entry.title if hasattr(entry, 'title') else "Untitled"
                url = entry.link if hasattr(entry, 'link') else ""
                
                if not url:
                    continue
                
                pub_date = entry.published if hasattr(entry, 'published') else None
                
                logger.info(f"Downloading article: {title}")
                article_text, char_count = get_article_text(url)
                
                if save_article_to_file(source_name, title, url, pub_date, article_text, char_count):
                    new_articles_added += 1
                    logger.info(f"Article saved: {title} ({char_count} chars)")
                
                time.sleep(random.uniform(0.5, 1.5))
                
                if new_articles_added >= target_count:
                    break
        
        except Exception as e:
            logger.error(f"Error processing source {source_name}: {e}")
    
    logger.info(f"Completed! Added {new_articles_added} new sports articles.")
    return new_articles_added

if __name__ == "__main__":
    logger.info("Starting sports article collection...")
    target_count = 100
    
    collected = collect_sport_news(target_count=target_count)
    
    logger.info(f"Sports article collection completed, {collected}/{target_count} articles saved.")