import os
import logging
import pandas as pd
import numpy as np
import pickle

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """
    Sentiment Analyzer pro české texty, který používá natrénovaný model
    """
    
    def __init__(self, max_features=15000):
        """
        Inicializace analyzátoru
        """
        self.max_features = max_features
        self.pipeline = None
        self.labels = ['negative', 'neutral', 'positive']
        self.positive_words = []
        self.negative_words = []
        self.critical_negative_words = set()
    
    def predict(self, texts):
        """
        Predikce sentimentu pro seznam textů
        
        Args:
            texts (list): Seznam textů k analýze
            
        Returns:
            list: Seznam sentiment ID (0=negative, 1=neutral, 2=positive)
        """
        if self.pipeline is None:
            raise ValueError("Model nebyl načten. Použijte load_model() nejdřív.")
            
        if not isinstance(texts, list):
            texts = [texts]
            
        try:
            # Ensure texts are strings
            texts = [str(text) if text is not None else "" for text in texts]
            
            # Predict using pipeline
            predictions = self.pipeline.predict(texts)
            return predictions
        except Exception as e:
            logger.error(f"Chyba při predikci sentimentu: {e}")
            return [1] * len(texts)  # Default to neutral
    
    def extract_sentiment_features(self, texts):
        """
        Extrakce sentiment příznaků z textů
        
        Args:
            texts (list): Seznam textů
            
        Returns:
            pd.DataFrame: DataFrame s příznaky
        """
        if not isinstance(texts, list):
            texts = [texts]
            
        # Ensure texts are strings
        texts = [str(text) if text is not None else "" for text in texts]
        
        features = pd.DataFrame()
        
        # Count positive and negative words
        positive_word_counts = []
        negative_word_counts = []
        sentiment_ratios = []
        
        for text in texts:
            text_lower = text.lower()
            words = text_lower.split()
            
            pos_count = sum(1 for word in words if word in self.positive_words)
            neg_count = sum(1 for word in words if word in self.negative_words)
            
            positive_word_counts.append(pos_count)
            negative_word_counts.append(neg_count)
            sentiment_ratios.append((pos_count + 1) / (neg_count + 1))  # +1 to avoid division by zero
        
        features['positive_word_count'] = positive_word_counts
        features['negative_word_count'] = negative_word_counts
        features['sentiment_ratio'] = sentiment_ratios
        
        return features
    
    def explain_prediction(self, text):
        """
        Vysvětlení predikce sentimentu pro daný text
        
        Args:
            text (str): Text k analýze
            
        Returns:
            dict: Vysvětlení se seznamem pozitivních a negativních slov
        """
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
            
        text_lower = text.lower()
        words = text_lower.split()
        
        # Check for positive/negative words
        positive_words_found = [word for word in words if word in self.positive_words]
        negative_words_found = [word for word in words if word in self.negative_words]
        
        # Get counts
        positive_count = len(positive_words_found)
        negative_count = len(negative_words_found)
        
        # Get sentiment classification
        sentiment_id = self.predict([text])[0]
        sentiment = self.labels[sentiment_id]
        
        # Calculate sentiment ratio
        sentiment_ratio = (positive_count + 1) / (negative_count + 1)
        
        # Create explanation
        explanation = {
            'text': text,
            'predicted_sentiment': sentiment,
            'positive_words': positive_words_found[:10],  # Limit to top 10
            'negative_words': negative_words_found[:10],
            'positive_word_count': positive_count,
            'negative_word_count': negative_count,
            'sentiment_ratio': sentiment_ratio
        }
        
        return explanation
    
    @classmethod
    def load_model(cls, model_dir):
        """
        Načtení modelu ze složky
        
        Args:
            model_dir (str): Cesta ke složce s modelem
            
        Returns:
            SentimentAnalyzer: Instance analyzátoru s načteným modelem
        """
        try:
            # Load model info
            with open(os.path.join(model_dir, 'model_info.pkl'), 'rb') as f:
                model_info = pickle.load(f)
            
            # Create instance
            instance = cls(max_features=model_info.get('max_features', 15000))
            
            # Load pipeline
            with open(os.path.join(model_dir, 'pipeline.pkl'), 'rb') as f:
                instance.pipeline = pickle.load(f)
            
            # Load lexicons
            with open(os.path.join(model_dir, 'lexicons.pkl'), 'rb') as f:
                lexicons = pickle.load(f)
                instance.positive_words = lexicons.get('positive_words', [])
                instance.negative_words = lexicons.get('negative_words', [])
                instance.critical_negative_words = set(lexicons.get('critical_negative_words', []))
            
            # Set labels
            if 'labels' in model_info:
                instance.labels = model_info['labels']
            
            logger.info(f"Model načten z {model_dir}")
            return instance
        except Exception as e:
            logger.error(f"Chyba při načítání modelu: {e}")
            raise