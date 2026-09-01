# SENTI-SCOPE

## Real-Time Sentiment Analysis System with Web Scraping

SENTI-SCOPE is an end-to-end college project for real-time binary sentiment analysis using Sentiment140 tweets for model training and public web/news text for live analysis.

## Objective

The system trains Naive Bayes and Logistic Regression classifiers with TF-IDF features, selects the best model using real evaluation results, and uses the saved model for manual text and live web/news predictions.

## Sentiment Classes

The downloaded Sentiment140 dataset supports two labels:

- `0` = NEGATIVE
- `4` = POSITIVE

This project trains and predicts only NEGATIVE and POSITIVE. It does not fabricate neutral labels.

## Features

- Memory-efficient Sentiment140 CSV loading
- Configurable development subset (`100000`) and full-data support (`None`)
- Shared preprocessing for training, testing, manual input, and scraped text
- Lowercase conversion, URL removal, mention removal, hashtag handling, punctuation removal, whitespace cleanup, stopword removal, and stemming
- TF-IDF feature extraction
- Naive Bayes baseline
- Logistic Regression production candidate
- Accuracy, precision, recall, F1-score, classification reports, confusion matrices, and comparison chart
- Best-model selection based on actual weighted F1-score
- Saved model and vectorizer; dashboard does not retrain models
- Public Google News RSS scraping
- SQLite storage of actual predictions
- Streamlit and Plotly dashboard
- Dashboard, Analyze Text, Live Web Analysis, Trends, Topic Analysis, Model Performance, Alerts, and About pages

## Folder Structure

```text
real_time_sentiment_analysis/
â”œâ”€â”€ dataset/
â”‚   â””â”€â”€ training.1600000.processed.noemoticon.csv
â”œâ”€â”€ models/
â”œâ”€â”€ results/
â”œâ”€â”€ data/
â”œâ”€â”€ src/
â”‚   â”œâ”€â”€ __init__.py
â”‚   â”œâ”€â”€ preprocessing.py
â”‚   â”œâ”€â”€ storage.py
â”‚   â”œâ”€â”€ scraper.py
â”‚   â”œâ”€â”€ predict.py
â”‚   â”œâ”€â”€ evaluate_model.py
â”‚   â””â”€â”€ train_model.py
â”œâ”€â”€ app.py
â”œâ”€â”€ requirements.txt
â””â”€â”€ README.md
```

## Installation

Windows PowerShell:

```powershell
cd C:\Users\subba\real_time_sentiment_analysis
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Dataset

Keep the downloaded file at:

```text
dataset/training.1600000.processed.noemoticon.csv
```

## Training

Development mode uses 100,000 rows. Train the models with:

```powershell
python -m src.train_model
```

For final complete training, open `src/train_model.py`, change:

```python
TRAINING_SAMPLE_SIZE = 100000
```

to:

```python
TRAINING_SAMPLE_SIZE = None
```

Then rerun:

```powershell
python -m src.train_model
```

The training script saves:

```text
models/sentiment_model.pkl
models/tfidf_vectorizer.pkl
models/model_metrics.json
results/naive_bayes_confusion_matrix.png
results/logistic_regression_confusion_matrix.png
results/model_comparison.png
```

## Run Dashboard

```powershell
streamlit run app.py
```

Streamlit normally opens the project at `http://localhost:8501`.

## Testing

Manual test texts:

```text
I absolutely love this product!
This is the worst service ever.
```

Use the Analyze Text page. The dashboard uses the actual selected saved model, not hard-coded results.

## Web Scraping

The Live Web Analysis page requests publicly accessible Google News RSS results for a user-entered topic. Titles and public snippets are collected, processed through the same preprocessing and TF-IDF pipeline, predicted by the saved model, and saved locally with source/topic/timestamp information.

The application does not require Twitter/X credentials, login, private information, CAPTCHA bypassing, or access-control bypassing.

## Faculty Demonstration

1. Show the Sentiment140 dataset inside `dataset/`.
2. Run `python -m src.train_model`.
3. Show real metrics in the training terminal.
4. Run `streamlit run app.py`.
5. Use the positive and negative sample texts in Analyze Text.
6. Show the Model Performance page and confusion matrices.
7. Use Live Web Analysis with a topic such as `technology`.
8. Show actual prediction records in Dashboard and Topic / Keyword Analysis.
9. Set a threshold in Alerts and show the alert calculated from stored real predictions.

## Limitations

- Sentiment140 is tweet-based, while public news headlines and snippets may have a different writing style.
- Public source availability depends on network connectivity and the RSS response.
- This version is binary because the supplied training data has only Positive and Negative labels.
- Neutral analysis may be considered later only with an appropriate neutral-labeled dataset.
