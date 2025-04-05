from models.sentiment_analyzer.sentiment_analyzer import SentimentAnalyzer

# Load the model
model = SentimentAnalyzer.load_model("models/sentiment_analyzer")

# Example test text
text = "Tragická nehoda si vyžádala tři životy. Byla to opravdu smutná událost."

# Run sentiment prediction and explanation
result = model.explain_prediction(text)

# Print it nicely
import json
print(json.dumps(result, ensure_ascii=False, indent=2))
