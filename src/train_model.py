import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from src.evaluate_model import (
    calculate_metrics,
    save_comparison_chart,
    save_confusion_matrix,
)
from src.preprocessing import preprocess_series

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "training.1600000.processed.noemoticon.csv"
)
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Development mode. Set to None later to use all 1.6 million rows.
TRAINING_SAMPLE_SIZE = 100000

RANDOM_STATE = 42
TEST_SIZE = 0.20
MAX_FEATURES = 50000

COLUMN_NAMES = ["target", "id", "date", "query", "user", "text"]


def load_dataset(sample_size=TRAINING_SAMPLE_SIZE):
    """Load a balanced positive/negative Sentiment140 training subset."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}\n"
            "Copy the Sentiment140 CSV into the dataset folder."
        )

    if sample_size is not None and sample_size < 2:
        raise ValueError("TRAINING_SAMPLE_SIZE must be at least 2 or None.")

    rows_per_class = None if sample_size is None else sample_size // 2

    negative_parts = []
    positive_parts = []
    negative_count = 0
    positive_count = 0

    try:
        reader = pd.read_csv(
            DATASET_PATH,
            encoding="latin-1",
            header=None,
            names=COLUMN_NAMES,
            usecols=[0, 5],
            chunksize=50000,
            on_bad_lines="skip",
        )

        for chunk in reader:
            chunk = chunk[chunk["target"].isin([0, 4])][["target", "text"]]

            if rows_per_class is None:
                negative_parts.append(chunk[chunk["target"] == 0])
                positive_parts.append(chunk[chunk["target"] == 4])
                continue

            if negative_count < rows_per_class:
                negatives = chunk[chunk["target"] == 0].head(
                    rows_per_class - negative_count
                )
                if not negatives.empty:
                    negative_parts.append(negatives)
                    negative_count += len(negatives)

            if positive_count < rows_per_class:
                positives = chunk[chunk["target"] == 4].head(
                    rows_per_class - positive_count
                )
                if not positives.empty:
                    positive_parts.append(positives)
                    positive_count += len(positives)

            if (
                negative_count >= rows_per_class
                and positive_count >= rows_per_class
            ):
                break

    except Exception as error:
        raise RuntimeError(
            f"Unable to read Sentiment140 CSV: {error}"
        ) from error

    negative_data = (
        pd.concat(negative_parts, ignore_index=True)
        if negative_parts
        else pd.DataFrame(columns=["target", "text"])
    )
    positive_data = (
        pd.concat(positive_parts, ignore_index=True)
        if positive_parts
        else pd.DataFrame(columns=["target", "text"])
    )

    dataframe = pd.concat(
        [negative_data, positive_data],
        ignore_index=True,
    )

    dataframe = dataframe[dataframe["target"].isin([0, 4])].copy()
    dataframe["label"] = dataframe["target"].map({0: 0, 4: 1})
    dataframe = dataframe[["text", "label"]].dropna()

    if dataframe.empty:
        raise ValueError(
            "No usable positive/negative Sentiment140 rows were loaded."
        )

    class_count = dataframe["label"].nunique()
    if class_count < 2:
        raise ValueError(
            "Training requires both Sentiment140 classes: "
            "0 (NEGATIVE) and 4 (POSITIVE)."
        )

    return dataframe.sample(
        frac=1,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)


def main():
    print("=" * 65)
    print("SENTI-SCOPE: MODEL TRAINING")
    print("=" * 65)

    print("\n[1/7] Loading balanced Sentiment140 dataset...")
    dataframe = load_dataset()

    print(f"Loaded rows: {len(dataframe):,}")
    print(f"Negative rows: {(dataframe['label'] == 0).sum():,}")
    print(f"Positive rows: {(dataframe['label'] == 1).sum():,}")

    print("\n[2/7] Applying shared text preprocessing...")
    dataframe["cleaned_text"] = preprocess_series(dataframe["text"])
    dataframe = dataframe[
        dataframe["cleaned_text"].str.len() > 0
    ].copy()

    print(f"Usable rows: {len(dataframe):,}")
    print(f"Negative usable rows: {(dataframe['label'] == 0).sum():,}")
    print(f"Positive usable rows: {(dataframe['label'] == 1).sum():,}")

    if dataframe["label"].nunique() < 2:
        raise ValueError(
            "Preprocessing left fewer than two sentiment classes. "
            "Cannot train binary models."
        )

    print("\n[3/7] Creating reproducible train/test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        dataframe["cleaned_text"],
        dataframe["label"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=dataframe["label"],
    )

    print("\n[4/7] Fitting TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print(f"Training feature shape: {X_train_tfidf.shape}")
    print(f"Testing feature shape: {X_test_tfidf.shape}")

    print("\n[5/7] Training Naive Bayes baseline...")
    naive_bayes = MultinomialNB(alpha=0.5)
    naive_bayes.fit(X_train_tfidf, y_train)
    nb_predictions = naive_bayes.predict(X_test_tfidf)

    print("\n[6/7] Training Logistic Regression...")
    logistic_regression = LogisticRegression(
        max_iter=500,
        solver="saga",
        C=2.0,
        random_state=RANDOM_STATE,
    )
    logistic_regression.fit(X_train_tfidf, y_train)
    lr_predictions = logistic_regression.predict(X_test_tfidf)

    print("\n[7/7] Evaluating, selecting model, and saving files...")
    nb_metrics = calculate_metrics(y_test, nb_predictions)
    lr_metrics = calculate_metrics(y_test, lr_predictions)

    metrics_by_model = {
        "Naive Bayes": nb_metrics,
        "Logistic Regression": lr_metrics,
    }

    best_model_name = max(
        metrics_by_model,
        key=lambda name: metrics_by_model[name]["f1_score"],
    )

    production_model = (
        logistic_regression
        if best_model_name == "Logistic Regression"
        else naive_bayes
    )

    save_confusion_matrix(y_test, nb_predictions, "Naive Bayes")
    save_confusion_matrix(
        y_test,
        lr_predictions,
        "Logistic Regression",
    )
    save_comparison_chart(metrics_by_model)

    joblib.dump(
        production_model,
        MODELS_DIR / "sentiment_model.pkl",
    )
    joblib.dump(
        vectorizer,
        MODELS_DIR / "tfidf_vectorizer.pkl",
    )

    training_output = {
        "training_sample_size": TRAINING_SAMPLE_SIZE,
        "usable_rows": int(len(dataframe)),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "production_model": best_model_name,
        "production_accuracy": metrics_by_model[
            best_model_name
        ]["accuracy"],
        "models": metrics_by_model,
    }

    with open(
        MODELS_DIR / "model_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(training_output, file, indent=4)

    for name, values in metrics_by_model.items():
        print(f"\n{name}")
        print(f"Accuracy : {values['accuracy']:.4f}")
        print(f"Precision: {values['precision']:.4f}")
        print(f"Recall   : {values['recall']:.4f}")
        print(f"F1-score : {values['f1_score']:.4f}")

    print(f"\nProduction model selected: {best_model_name}")
    print("Saved: models/sentiment_model.pkl")
    print("Saved: models/tfidf_vectorizer.pkl")
    print("Saved: models/model_metrics.json")
    print("Saved charts in results/")
    print("\nTraining completed successfully.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nTraining failed: {error}")
        sys.exit(1)