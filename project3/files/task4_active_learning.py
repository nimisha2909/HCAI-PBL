"""
Task 4: Active learning for expert-competence discovery.

Setting: the classifier is already trained (Task 1), but we have NO expert
labels yet. We have a query budget: we may ask the expert for a label on a
chosen subset of points, one batch at a time, to learn the deferral
threshold (i.e. learn when the classifier should be trusted vs when to
defer).

Strategy compared:
 - Uncertainty sampling: query the points where the classifier is least
   confident first (these are exactly the points deferral decisions matter
   most for).
 - Random sampling: query a random subset each round (baseline to compare
   against).

At each round we retune the deferral threshold (Task 3 logic) using only the
expert labels queried so far, then evaluate combined accuracy on a fixed held
-out test set.
"""
import numpy as np
from task3_l2d import get_confidence, tune_threshold, evaluate_l2d


def active_learning_curve(model, X_pool, y_pool_true, expert_pool_labels,
                           X_test, y_test, expert_test_labels,
                           query_budget=500, batch_size=50, strategy="uncertainty",
                           seed=0):
    """
    Simulates querying the expert in batches, retuning the deferral
    threshold after each batch, and evaluating on the held-out test set.
    Returns list of (n_queried, combined_test_accuracy).
    """
    rng = np.random.default_rng(seed)
    n_pool = X_pool.shape[0]
    preds_pool, conf_pool = get_confidence(model, X_pool)

    unqueried = np.ones(n_pool, dtype=bool)
    queried_idx = np.array([], dtype=int)
    history = []

    n_rounds = query_budget // batch_size
    for _ in range(n_rounds):
        available = np.where(unqueried)[0]
        if len(available) == 0:
            break

        if strategy == "uncertainty":
            # pick the least-confident available points
            order = np.argsort(conf_pool[available])
            chosen = available[order[:batch_size]]
        else:  # random
            chosen = rng.choice(available, size=min(batch_size, len(available)), replace=False)

        unqueried[chosen] = False
        queried_idx = np.concatenate([queried_idx, chosen])

        # Use only the expert labels queried so far as our "validation" set
        # to tune the deferral threshold.
        X_known = X_pool[queried_idx]
        y_known = y_pool_true[queried_idx]
        expert_known = expert_pool_labels[queried_idx]

        threshold, _ = tune_threshold(model, X_known, y_known, expert_known)
        metrics = evaluate_l2d(model, X_test, y_test, expert_test_labels, threshold)

        history.append({
            "n_queried": len(queried_idx),
            "combined_accuracy": metrics["combined_accuracy"],
            "threshold": threshold,
            "deferral_rate": metrics["deferral_rate"],
        })

    return history
