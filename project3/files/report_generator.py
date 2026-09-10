"""
Generates the PDF report required by the assignment: description of
experiments, design justifications, and detailed results. Built with
matplotlib's PdfPages so it needs no extra dependency beyond matplotlib.
"""
import io
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from data_utils import LABEL_NAMES


def _text_page(pdf, title, lines, fontsize=11):
    fig = plt.figure(figsize=(8.27, 11.69))  # A4
    fig.text(0.08, 0.94, title, fontsize=16, weight="bold")
    y = 0.88
    for line in lines:
        fig.text(0.08, y, line, fontsize=fontsize, wrap=True, va="top")
        y -= 0.045
    plt.axis("off")
    pdf.savefig(fig)
    plt.close(fig)


def build_report(baseline_acc, expert_overall_acc, expert_report,
                  l2d_metrics, al_history_uncertainty, al_history_random):
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        # Page 1 -- overview / design choices
        _text_page(pdf, "Project 3: Active Learning for Learning-to-Defer", [
            "Dataset: AG News (4-class news topic classification).",
            "",
            "Task 1 -- Baseline: TF-IDF features + Logistic Regression, trained",
            "on the full labeled training set.",
            f"  Test accuracy: {baseline_acc:.3f}",
            "",
            "Task 2 -- Simulated expert: per-class competence profile. Strong",
            "on World/Sports (distinctive vocabulary), weaker on Business vs",
            "Sci/Tech (overlapping vocabulary causes confusion between them).",
            f"  Overall expert accuracy: {expert_overall_acc:.3f}",
            "",
            "Task 3 -- Learning-to-defer: confidence-based deferral. The",
            "classifier's own top-class probability is used to decide whether",
            "to trust the model or query the expert. Threshold tuned on a",
            "validation split to maximize accuracy minus a per-query cost.",
            "",
            "Task 4 -- Active learning: with no expert labels available up",
            "front, points to query are chosen by uncertainty sampling (least",
            "confident classifier predictions first) vs a random baseline,",
            "then the deferral threshold is retuned after each batch.",
        ])

        # Page 2 -- expert competence table
        rows = [f"{LABEL_NAMES[i]}: recall {expert_report[LABEL_NAMES[i]]['recall']:.3f}"
                for i in sorted(LABEL_NAMES)]
        _text_page(pdf, "Task 2 -- Simulated Expert: Per-Class Accuracy", rows)

        # Page 3 -- L2D results
        _text_page(pdf, "Task 3 -- Learning-to-Defer Results", [
            f"Classifier-only test accuracy: {l2d_metrics['classifier_only_accuracy']:.3f}",
            f"Combined (classifier + expert) test accuracy: {l2d_metrics['combined_accuracy']:.3f}",
            f"Deferral rate: {l2d_metrics['deferral_rate']:.3f}",
            f"Tuned confidence threshold: {l2d_metrics['threshold']:.3f}",
            "",
            "On deferred points, the classifier alone would have been wrong "
            f"{l2d_metrics['classifier_would_be_wrong_rate_on_deferred']:.3f} of the time,",
            "while the expert was correct "
            f"{l2d_metrics['expert_correct_rate_on_deferred']:.3f} of the time -- "
            "showing deferral targets genuinely hard cases.",
        ])

        # Page 4 -- active learning curve plot
        fig, ax = plt.subplots(figsize=(8.27, 6))
        ax.plot([h["n_queried"] for h in al_history_uncertainty],
                [h["combined_accuracy"] for h in al_history_uncertainty],
                label="Uncertainty sampling", marker="o")
        ax.plot([h["n_queried"] for h in al_history_random],
                [h["combined_accuracy"] for h in al_history_random],
                label="Random sampling", marker="x")
        ax.set_xlabel("Number of expert queries")
        ax.set_ylabel("Combined test accuracy")
        ax.set_title("Task 4 -- Active Learning: Query Efficiency")
        ax.legend()
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    buf.seek(0)
    return buf
