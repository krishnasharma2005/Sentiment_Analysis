import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch.nn.functional as F
model_name = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def analyze_sentiment(text):
    # tokenizing the input text
    inputs = tokenizer(text, return_tensors="pt")

    outputs = model(**inputs)
    
    probs = F.softmax(outputs.logits, dim=-1)
    # sentiment with highest probab
    sentiment = torch.argmax(probs, dim=-1).item()
    # sentiment mapping
    sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}

    return sentiment_map[sentiment]
texts = [
        "I love this",
        "It's okay, not great but not bad either.",
        "I really hate this",
        "My name is Krishna"
    ]
for text in texts:
        sentiment = analyze_sentiment(text)
        print(f"Text: {text}\nSentiment: {sentiment}\n")
