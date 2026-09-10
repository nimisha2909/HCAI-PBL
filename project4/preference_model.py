"""
Task 2 - Ranking extension of the Bradley-Terry model.

Standard Bradley-Terry (pairwise):
    P(i > j | w) = exp(w.x_i) / (exp(w.x_i) + exp(w.x_j))

Extension to a full ranking i1 > i2 > ... > in (Plackett-Luce model):
    P(i1>i2>...>in | w) = prod_{k=1}^{n} exp(w.x_ik) / sum_{l=k}^{n} exp(w.x_il)

Interpretation: generate the ranking by repeatedly drawing a "winner" from
the remaining pool with Bradley-Terry-style probability proportional to
exp(w.x); the winner is placed next in the ranking, then removed from the
pool, and the process repeats. Setting n=2 recovers exactly the pairwise
Bradley-Terry model, so this is a strict generalization -- both study
designs (pairwise choice, top-to-bottom rank of 10) are special cases of
the same underlying utility model U(x) = w^T x, which is what lets us
compare the two elicitation methods on equal footing (same model class,
same log-likelihood units).

We fit w by maximum likelihood (equivalently, minimizing the negative
log-likelihood) with gradient descent. Both pairwise comparisons and
full rankings can be mixed in the same optimization, which is exactly
what we need in the study: every trial (from either interface) is just
one more ranking observation (a pairwise choice is a ranking of length 2).
"""
import numpy as np


def neg_log_likelihood(w, rankings, X):
    """rankings: list of lists of movie indices, each already ordered
    most-preferred-first, e.g. [i1, i2, i3] means i1 > i2 > i3.
    X: (n_movies, d) feature matrix."""
    nll = 0.0
    for ranking in rankings:
        utilities = X[ranking] @ w  # utilities in ranked order
        # sequential Plackett-Luce terms
        for k in range(len(ranking)):
            remaining = utilities[k:]
            # log-sum-exp for numerical stability
            m = remaining.max()
            log_denom = m + np.log(np.exp(remaining - m).sum())
            nll -= (utilities[k] - log_denom)
    return nll


def grad_neg_log_likelihood(w, rankings, X):
    d = X.shape[1]
    grad = np.zeros(d)
    for ranking in rankings:
        idx = np.array(ranking)
        utilities = X[idx] @ w
        for k in range(len(idx)):
            remaining_idx = idx[k:]
            remaining_u = utilities[k:]
            m = remaining_u.max()
            weights = np.exp(remaining_u - m)
            weights /= weights.sum()  # softmax over remaining pool
            grad += -X[idx[k]] + weights @ X[remaining_idx]
    return grad


def fit_preference_vector(rankings, X, l2=0.05, lr=0.05, n_iter=500):
    """Fits w via gradient descent with L2 regularization (a Gaussian prior
    on w, which also keeps the estimate well-behaved when only a handful
    of interactions are available -- exactly the "quick adaptation" regime
    this project targets)."""
    d = X.shape[1]
    w = np.zeros(d)
    for _ in range(n_iter):
        g = grad_neg_log_likelihood(w, rankings, X) + l2 * w
        w -= lr * g
    return w


if __name__ == "__main__":
    import pandas as pd
    from features import build_feature_matrix

    df = pd.read_csv("movie_metadata_sample.csv")
    X, names, titles = build_feature_matrix(df)

    # sanity check: synthetic user who only likes Animation, dislikes Horror
    true_w = np.zeros(X.shape[1])
    true_w[names.index("genre::Animation")] = 3.0
    true_w[names.index("genre::Horror")] = -3.0

    rng = np.random.default_rng(0)
    rankings = []
    for _ in range(40):
        idx = rng.choice(len(titles), size=2, replace=False)
        u = X[idx] @ true_w
        order = idx[np.argsort(-u)]
        rankings.append(list(order))

    w_hat = fit_preference_vector(rankings, X)
    print("Recovered top weighted features:")
    for n, v in sorted(zip(names, w_hat), key=lambda t: -abs(t[1]))[:6]:
        print(f"  {n:22s} {v:+.3f}")
