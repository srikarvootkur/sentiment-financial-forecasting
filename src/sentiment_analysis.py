from transformers import BertTokenizer, BertForSequenceClassification
import torch

class SentimentAnalyzer:
    def __init__(self, model_name="yiyanghkust/finbert-tone"):
        #Initialize the sentiment analyzer with the specified FinBERT model.
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name)
    
    def analyze_sentiment(self, text):
        #Analyze the sentiment of a single text input.
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        sentiment = torch.argmax(outputs.logits)
        return sentiment.item()  
        # 0: Negative, 1: Neutral, 2: Positive
    
    def batch_analyze_sentiment(self, texts):
        #Analyze the sentiment of a batch of texts.
        sentiments = [self.analyze_sentiment(text) for text in texts]
        return sentiments
