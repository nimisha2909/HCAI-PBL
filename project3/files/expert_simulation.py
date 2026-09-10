"""
Task 2: Simulated expert.

The expert is designed to be imperfect but non-random: strong on classes with
distinctive vocabulary (World, Sports), weak on classes that are easy to
confuse with each other (Business vs Sci/Tech), mimicking a human editor who
reads news casually but isn't a domain specialist in finance/technology.
"""
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from data_utils import LABEL_NAMES

# Per-class probability the expert gives the CORRECT label.
# World / Sports: expert is reliable (distinctive vocabulary).
# Business / Sci/Tech: expert often confuses the two (overlapping vocabulary
# -- companies, markets, products).
EXPERT_ACCURACY_BY_CLASS = {
    0: 0.95,  # World
    1: 0.96,  # Sports
    2: 0.62,  # Business
    3: 0.60,  # Sci/Tech
}

# When the expert is wrong on Business/Sci-Tech, it confuses them with each
# other specifically (not a uniform random wrong class).
CONFUSION_PAIR = {2: 3, 3: 2}


def simulate_expert_labels(y_true, seed=0):
    """
    Given true labels, produce the expert's simulated prediction for each
    point, according to the per-class competence profile above.
    """
    rng = np.random.default_rng(seed)
    y_expert = np.array(y_true, copy=True)

    for i, y in enumerate(y_true):
        p_correct = EXPERT_ACCURACY_BY_CLASS[int(y)]
        if rng.random() > p_correct:
            if int(y) in CONFUSION_PAIR:
                y_expert[i] = CONFUSION_PAIR[int(y)]
            else:
                # pick a random other class
                choices = [c for c in LABEL_NAMES if c != y]
                y_expert[i] = rng.choice(choices)
    return y_expert


def analyze_expert(y_true, y_expert):
    """Report overall + per-class expert accuracy."""
    overall_acc = accuracy_score(y_true, y_expert)
    report = classification_report(
        y_true, y_expert, target_names=[LABEL_NAMES[i] for i in sorted(LABEL_NAMES)],
        output_dict=True,
    )
    return overall_acc, report


if __name__ == "__main__":
    # Quick standalone sanity check with random labels
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 4, size=2000)
    y_expert = simulate_expert_labels(y_true)
    acc, report = analyze_expert(y_true, y_expert)
    print(f"Simulated expert overall accuracy: {acc:.3f}")
    for cls, name in LABEL_NAMES.items():
        print(f"  {name}: recall {report[name]['recall']:.3f}")
