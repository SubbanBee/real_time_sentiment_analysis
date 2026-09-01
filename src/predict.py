import json
from pathlib import Path

import joblib

from src.preprocessing import preprocess_text

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "sentiment_model.pkl"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"
METRICS_PATH = MODELS_DIR / "model_metrics.json"


class SentimentPredictor:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.metrics = None

    def load(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                "Trained model not found. Run: python -m src.train_model"
            )

        if not VECTORIZER_PATH.exists():
            raise FileNotFoundError(
                "TF-IDF vectorizer not found. Run: python -m src.train_model"
            )

        self.model = joblib.load(MODEL_PATH)
        self.vectorizer = joblib.load(VECTORIZER_PATH)

        if METRICS_PATH.exists():
            with open(METRICS_PATH, "r", encoding="utf-8") as file:
                self.metrics = json.load(file)

        return self

    def predict_one(self, text: str):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Please enter text to analyze.")

        cleaned_text = preprocess_text(text)

        if not cleaned_text:
            raise ValueError(
                "The entered text has no usable words after preprocessing."
            )

        features = self.vectorizer.transform([cleaned_text])
        prediction = int(self.model.predict(features)[0])
        sentiment = "POSITIVE" if prediction == 1 else "NEGATIVE"

        confidence = None
        probabilities = None

        if hasattr(self.model, "predict_proba"):
            values = self.model.predict_proba(features)[0]
            probabilities = {
                "NEGATIVE": float(values[0]),
                "POSITIVE": float(values[1])
            }
            confidence = float(max(values))

        return {
            "original_text": text,
            "cleaned_text": cleaned_text,
            "sentiment": sentiment,
            "confidence": confidence,
            "probabilities": probabilities
        }


def load_predictor():
    return SentimentPredictor().load()
