import re
import string
from functools import lru_cache

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


def ensure_nltk_resources():
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)


@lru_cache(maxsize=1)
def get_stop_words():
    ensure_nltk_resources()
    return set(stopwords.words("english"))


@lru_cache(maxsize=1)
def get_stemmer():
    return PorterStemmer()


def preprocess_text(text: str) -> str:
    """One shared preprocessing pipeline for every text source."""
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    stop_words = get_stop_words()
    stemmer = get_stemmer()

    return " ".join(
        stemmer.stem(word)
        for word in text.split()
        if word not in stop_words and len(word) > 1
    )


def preprocess_series(text_series):
    return text_series.fillna("").astype(str).map(preprocess_text)
