import os
import sys
import importlib.util

def load_sentiment_analyzer():
    """Load the compiled sentiment analyzer module"""
    try:
        # Add the models directory to the path
        models_dir = os.path.dirname(os.path.abspath(__file__))
        if models_dir not in sys.path:
            sys.path.append(models_dir)
            
        # Try loading from the .pyc file directly
        pyc_path = os.path.join(models_dir, '__pycache__', 'sentiment_analyzer.cpython-311.pyc')
        
        if os.path.exists(pyc_path):
            import importlib.machinery
            loader = importlib.machinery.SourcelessFileLoader('sentiment_analyzer', pyc_path)
            spec = importlib.util.spec_from_loader('sentiment_analyzer', loader)
            sentiment_analyzer = importlib.util.module_from_spec(spec)
            loader.exec_module(sentiment_analyzer)
            return sentiment_analyzer
        else:
            # Try normal import as fallback (will still show warning in editor but might work at runtime)
            try:
                # Use exec to avoid Pylance warning
                module_name = 'sentiment_analyzer'
                exec(f"import {module_name}")
                exec(f"sentiment_analyzer = {module_name}")
                return eval("sentiment_analyzer")
            except ImportError:
                raise ImportError(f"Could not find sentiment_analyzer module or .pyc file")
    except Exception as e:
        print(f"Error loading compiled sentiment analyzer: {e}")
        # Return None to indicate failure
        return None