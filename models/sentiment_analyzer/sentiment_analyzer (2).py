import os
import logging
import pickle
import pandas as pd
import inspect

logger = logging.getLogger(__name__)

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
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
        elif not isinstance(texts, list):
            texts = [texts]
        
        features = pd.DataFrame()
        
        features['positive_word_count'] = [
            sum(1 for word in str(text).lower().split() if word in self.positive_words) 
            for text in texts
        ]
        
        features['negative_word_count'] = [
            sum(1 for word in str(text).lower().split() if word in self.negative_words)
            for text in texts
        ]
        
        features['sentiment_ratio'] = (features['positive_word_count'] + 1) / (features['negative_word_count'] + 1)
        
        return features
    
    def predict(self, texts):
        """Predict sentiment for texts"""
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
        elif not isinstance(texts, list):
            texts = [texts]
        
        if self.pipeline is None:
            raise ValueError("Model not trained. Must load model first.")
        
        predictions = []
        for text in texts:
            text_lower = str(text).lower()
            
            if any(crit_word in text_lower for crit_word in self.critical_negative_words):
                predictions.append(0)
            else:
                try:
                    pred = self.pipeline.predict([text])[0]
                    predictions.append(pred)
                except:
                    pos_count = sum(1 for word in text_lower.split() if word in self.positive_words)
                    neg_count = sum(1 for word in text_lower.split() if word in self.negative_words)
                    
                    if neg_count > pos_count * 1.2:
                        predictions.append(0)
                    elif pos_count > neg_count * 1.2:
                        predictions.append(2)
                    else:
                        predictions.append(1)
        
        return predictions
    
    def explain_prediction(self, text):
        """Provide explanation for sentiment prediction"""
        text_lower = str(text).lower()
        
        positive_words_found = [word for word in text_lower.split() if word in self.positive_words]
        negative_words_found = [word for word in text_lower.split() if word in self.negative_words]
        critical_words_found = [word for word in text_lower.split() 
                               if any(crit_word in word for crit_word in self.critical_negative_words)]
        
        positive_count = len(positive_words_found)
        negative_count = len(negative_words_found)
        critical_count = len(critical_words_found)
        
        sentiment_id = self.predict([text])[0]
        sentiment = self.labels[sentiment_id]
        
        sentiment_ratio = (positive_count + 1) / (negative_count + 1)
        
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
        
        with open(os.path.join(model_dir, 'pipeline.pkl'), 'wb') as f:
            pickle.dump(self.pipeline, f)
        
        with open(os.path.join(model_dir, 'lexicons.pkl'), 'wb') as f:
            pickle.dump({
                'positive_words': self.positive_words,
                'negative_words': self.negative_words,
                'critical_negative_words': self.critical_negative_words
            }, f)
        
        with open(os.path.join(model_dir, 'model_info.pkl'), 'wb') as f:
            pickle.dump({
                'max_features': self.max_features,
                'labels': self.labels
            }, f)
        
        with open(os.path.join(model_dir, 'sentiment_analyzer.py'), 'w', encoding='utf-8') as f:
            f.write(inspect.getsource(self.__class__))
            
        logger.info(f"Model saved to {model_dir}")
    
    @classmethod
    def load_model(cls, model_dir):
        """Load model from disk"""
        instance = cls()
        
        try:
            with open(os.path.join(model_dir, 'model_info.pkl'), 'rb') as f:
                model_info = pickle.load(f)
                instance.max_features = model_info.get('max_features', 15000)
                instance.labels = model_info.get('labels', ['negative', 'neutral', 'positive'])
            
            with open(os.path.join(model_dir, 'pipeline.pkl'), 'rb') as f:
                instance.pipeline = pickle.load(f)
            
            with open(os.path.join(model_dir, 'lexicons.pkl'), 'rb') as f:
                lexicons = pickle.load(f)
                instance.load_words(lexicons)
            
            return instance
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return instance
