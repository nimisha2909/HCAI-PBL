"""
Data loading and vectorization for the AG News dataset.
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

LABEL_NAMES = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}


def load_ag_news(train_size=8000, test_size=2000, seed=42):
    """
    Loads the AG News dataset via HuggingFace `datasets`.
    Subsamples to keep things fast to train/run locally.
    Returns: X_train_text, y_train, X_test_text, y_test (all numpy arrays)
    """
    from datasets import load_dataset

    ds = load_dataset("fancyzhx/ag_news")
    rng = np.random.default_rng(seed)

    def subsample(split, n):
        idx = rng.choice(len(split), size=min(n, len(split)), replace=False)
        texts = np.array([split[int(i)]["text"] for i in idx])
        labels = np.array([split[int(i)]["label"] for i in idx])
        return texts, labels

    X_train, y_train = subsample(ds["train"], train_size)
    X_test, y_test = subsample(ds["test"], test_size)
    return X_train, y_train, X_test, y_test


def vectorize(X_train_text, X_test_text, max_features=20000):
    """
    Fits a TF-IDF vectorizer on train text, transforms both splits.
    Returns: X_train, X_test, vectorizer
    """
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)
    return X_train, X_test, vectorizer
