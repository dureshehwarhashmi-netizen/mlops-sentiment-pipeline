from transformers import pipeline

pipe = pipeline("sentiment-analysis", model="models/")

def predict(text):
    return pipe(text)
