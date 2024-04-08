from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import joblib
import re
import os
from typing import List, Dict


model_path = "model.pkl"
vectorizer_path = "vectorizer.pkl"
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)


class InputData(BaseModel):
    text: str

class PredictionResult(BaseModel):
    class_name: str


app = FastAPI()


@app.post("/predict", response_model=PredictionResult)
async def predict(data: InputData):
    
    processed_text = preprocess_text(data.text)

    
    input_vector = vectorizer.transform([processed_text])

    
    prediction = model.predict(input_vector)[0]

    
    return PredictionResult(class_name=prediction)


def preprocess_text(text: str) -> str:
    
    text = text.lower()

    
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)

    
    tokens = text.split()

    
    stopwords = set(["the", "a", "and", "to", "in", "is", "was", "were", "are"])
    tokens = [token for token in tokens if token not in stopwords]

    
    lemmatizer = joblib.load("lemmatizer.pkl")
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]

    
    preprocessed_text = " ".join(lemmatized_tokens)
    return preprocessed_text

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)