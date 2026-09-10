"""
Task 1: Baseline classifier trained on the full labeled AG News training set.
Run directly: python task1_baseline.py
"""
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from data_utils import load_ag_news, vectorize, LABEL_NAMES


def train_baseline(X_train, y_train):
    model = LogisticRegression(max_iter=1000, C=5.0)
    model.fit(X_train, y_train)
    return model


if __name__ == "__main__":
    print("Loading AG News...")
    X_train_text, y_train, X_test_text, y_test = load_ag_news()

    print("Vectorizing...")
    X_train, X_test, vectorizer = vectorize(X_train_text, X_test_text)

    print("Training baseline classifier...")
    model = train_baseline(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Baseline test accuracy: {acc:.4f}")

    joblib.dump(model, "../models/baseline_classifier.joblib")
    joblib.dump(vectorizer, "../models/tfidf_vectorizer.joblib")
    print("Saved model + vectorizer to ../models/")
