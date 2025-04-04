import re
import logging
import string
import unicodedata

# Set up logging
logger = logging.getLogger(__name__)

class TextPreprocessor:
    """Text preprocessor for NLP tasks with specialized support for Czech language"""
    
    def __init__(self, language='czech'):
        """
        Initialize text preprocessor with language preference
        
        Args:
            language (str): Language to use for preprocessing (default: czech)
        """
        self.language = language.lower()
        
        # Czech diacritics replacement map (with lowercase)
        self.diacritic_map = {
            'ě': 'e', 'š': 's', 'č': 'c', 'ř': 'r', 'ž': 'z', 'ý': 'y',
            'á': 'a', 'í': 'i', 'é': 'e', 'ú': 'u', 'ů': 'u', 'ň': 'n', 
            'ť': 't', 'ď': 'd'
        }
        
        # Add uppercase versions
        uppercase_map = {k.upper(): v.upper() for k, v in self.diacritic_map.items()}
        self.diacritic_map.update(uppercase_map)
        
        # Load stopwords for the language
        self.stopwords = self._load_stopwords()
        
        logger.info(f"Text preprocessor initialized for {language} language")
    
    def _load_stopwords(self):
        """
        Load stopwords for specified language
        
        Returns:
            set: Set of stopwords
        """
        if self.language == 'czech':
            # Common Czech stopwords
            return {
                'a', 'aby', 'ale', 'ani', 'ano', 'asi', 'az', 'bez', 'bude', 'budem',
                'budes', 'by', 'byl', 'byla', 'byli', 'bylo', 'byt', 'ci', 'clanek',
                'clanku', 'clanky', 'co', 'coz', 'cz', 'dalsi', 'dnes', 'do',
                'ho', 'i', 'jak', 'jako', 'je', 'jeho', 'jej', 'jeji', 'jejich',
                'jen', 'jeste', 'ji', 'jine', 'jiz', 'jsem', 'jses', 'jsme', 'jsou', 'jste',
                'k', 'kam', 'kde', 'kdo', 'kdyz', 'ke', 'ktera', 'ktere', 'kteri', 'kterou',
                'ktery', 'ma', 'mate', 'me', 'mezi', 'mi', 'mit', 'mnou', 'muj', 'muze', 'my',
                'na', 'nad', 'nam', 'napiste', 'nas', 'nasi', 'ne', 'nebo', 'nebylo', 'neni',
                'nez', 'nic', 'nove', 'novy', 'o', 'od', 'ode', 'on', 'pak', 'po', 'pod',
                'podle', 'pokud', 'pouze', 'prave', 'pred', 'pres', 'pri', 'pro', 'proc', 'proto',
                'protoze', 'prvni', 's', 'se', 'si', 'sice', 'strana', 'sve', 'svych', 'svym',
                'svymi', 'ta', 'tak', 'take', 'takze', 'tam', 'tato', 'tedy', 'ten',
                'tento', 'teto', 'tim', 'timto', 'to', 'tohle', 'toho', 'tohoto', 'tom',
                'tomto', 'tomuto', 'tu', 'tuto', 'ty', 'tyto', 'u', 'uz', 'v', 'vam', 'vas',
                'vase', 've', 'vice', 'vsak', 'vse', 'z', 'za', 'zde', 'ze'
            }
        else:
            logger.warning(f"No stopwords available for {self.language} language")
            return set()
    
    def preprocess_text(self, text):
        """
        Preprocess text for NLP tasks - main method used by models
        
        Args:
            text (str): Text to preprocess
            
        Returns:
            str: Preprocessed text
        """
        if not isinstance(text, str) or not text:
            return ""
            
        try:
            # Convert to lowercase
            text = text.lower()
            
            # Remove URLs
            text = re.sub(r'https?://\S+|www\.\S+', '', text)
            
            # Remove email addresses
            text = re.sub(r'\S+@\S+', '', text)
            
            # Remove special characters and numbers
            text = re.sub(r'[^\w\s.,?!]', '', text)
            text = re.sub(r'\d+', '', text)
            
            # Remove extra spaces
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Remove diacritics (Czech-specific)
            for char, replacement in self.diacritic_map.items():
                text = text.replace(char, replacement)
            
            # Tokenize by whitespace
            tokens = text.split()
            
            # Remove stopwords
            tokens = [token for token in tokens if token not in self.stopwords]
            
            # Rejoin tokens
            cleaned_text = ' '.join(tokens)
            
            return cleaned_text
        except Exception as e:
            logger.error(f"Error preprocessing text: {e}")
            return text  # Return original text if preprocessing fails
    
    def preprocess_dataframe(self, df, text_column):
        """
        Preprocess a text column in a DataFrame
        
        Args:
            df (pd.DataFrame): DataFrame with text data
            text_column (str): Name of column containing text
            
        Returns:
            pd.DataFrame: DataFrame with preprocessed text
        """
        try:
            # Create a copy to avoid modifying the original
            df = df.copy()
            
            # Check if text column exists
            if text_column not in df.columns:
                logger.warning(f"Column {text_column} not found in DataFrame")
                return df
                
            # Create preprocessed column
            preprocessed_column = f"{text_column}_processed"
            
            # Apply preprocessing function to each row
            df[preprocessed_column] = df[text_column].apply(
                lambda x: self.preprocess_text(x) if isinstance(x, str) else ""
            )
            
            logger.info(f"Preprocessed {len(df)} rows in column {text_column}")
            return df
        except Exception as e:
            logger.error(f"Error preprocessing DataFrame: {e}")
            return df
    
    def remove_diacritics(self, text):
        """
        Remove Czech diacritical marks from text
        
        Args:
            text (str): Text with diacritics
            
        Returns:
            str: Text without diacritics
        """
        if not isinstance(text, str):
            return ""
            
        # Use direct mapping for Czech diacritics
        for char, replacement in self.diacritic_map.items():
            text = text.replace(char, replacement)
            
        return text
    
    def extract_keywords(self, text, top_n=10):
        """
        Extract top keywords from text
        
        Args:
            text (str): Text to extract keywords from
            top_n (int): Number of top keywords to return
            
        Returns:
            list: List of top keywords
        """
        if not isinstance(text, str) or not text:
            return []
            
        try:
            # Preprocess text
            processed_text = self.preprocess_text(text)
            
            # Tokenize
            tokens = processed_text.split()
            
            # Count word frequencies
            word_freq = {}
            for word in tokens:
                if len(word) > 2:  # Ignore very short words
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            # Sort by frequency
            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            
            # Return top N keywords
            return [word for word, _ in sorted_words[:top_n]]
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            return []