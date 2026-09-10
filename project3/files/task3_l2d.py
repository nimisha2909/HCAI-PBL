"""
Task 3: Learning-to-defer.

Strategy: confidence-based deferral. The baseline classifier's own predicted
probability is used as a proxy for how likely it is to be correct. When the
classifier's top-class probability falls below a threshold, the system
defers to the (simulated) expert instead of trusting the model. The
threshold is tuned per predicted class on a validation split to maximize
combined accuracy, penalized by a small cost per deferral (querying an
expert is not free).
"""
import numpy as np
from sklearn.metrics import accuracy_score
from data_utils import LABEL_NAMES


def get_confidence(model, X):
    """Top-class predicted probability for each sample."""
    proba = model.predict_proba(X)
    preds = np.argmax(proba, axis=1)
    conf = proba[np.arange(len(preds)), preds]
    return preds, conf


def l2d_predict(preds, conf, expert_labels, threshold):
    """Defer to expert wherever confidence < threshold."""
    final = preds.copy()
    deferred_mask = conf < threshold
    final[deferred_mask] = expert_labels[deferred_mask]
    return final, deferred_mask


def tune_threshold(model, X_val, y_val, expert_val_labels, defer_cost=0.02,
                    n_steps=50):
    """
    Sweep thresholds in [0, 1], pick the one maximizing
    accuracy - defer_cost * deferral_rate.
    """
    preds, conf = get_confidence(model, X_val)
    best_score, best_t = -np.inf, 0.0
    results = []
    for t in np.linspace(0, 1, n_steps):
        final, deferred = l2d_predict(preds, conf, expert_val_labels, t)
        acc = accuracy_score(y_val, final)
        defer_rate = deferred.mean()
        score = acc - defer_cost * defer_rate
        results.append((t, acc, defer_rate, score))
        if score > best_score:
            best_score, best_t = score, t
    return best_t, results


def evaluate_l2d(model, X_test, y_test, expert_test_labels, threshold):
    """Full evaluation of the deferral system on the test set."""
    preds, conf = get_confidence(model, X_test)
    final, deferred = l2d_predict(preds, conf, expert_test_labels, threshold)

    combined_acc = accuracy_score(y_test, final)
    classifier_only_acc = accuracy_score(y_test, preds)
    defer_rate = deferred.mean()

    # Deferral "quality": among deferred points, was deferring actually the
    # right call? (i.e. classifier would have been wrong, expert was right)
    if deferred.sum() > 0:
        clf_wrong_on_deferred = (preds[deferred] != y_test[deferred]).mean()
        expert_right_on_deferred = (expert_test_labels[deferred] == y_test[deferred]).mean()
    else:
        clf_wrong_on_deferred = expert_right_on_deferred = np.nan

    return {
        "combined_accuracy": combined_acc,
        "classifier_only_accuracy": classifier_only_acc,
        "deferral_rate": defer_rate,
        "classifier_would_be_wrong_rate_on_deferred": clf_wrong_on_deferred,
        "expert_correct_rate_on_deferred": expert_right_on_deferred,
        "threshold": threshold,
    }
