from fastapi import FastAPI
from transformers import pipeline

app = FastAPI()
pipe = pipeline("sentiment-analysis", model="models/")

@app.post("/predict")
def predict(text: str):
    return pipe(text)
