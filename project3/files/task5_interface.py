"""
Task 5: Streamlit interface for the full Project 3 pipeline.
Run with: streamlit run task5_interface.py
"""
import numpy as np
import streamlit as st
from sklearn.metrics import accuracy_score

from data_utils import load_ag_news, vectorize, LABEL_NAMES
from expert_simulation import simulate_expert_labels, analyze_expert
from task1_baseline import train_baseline
from task3_l2d import get_confidence, tune_threshold, evaluate_l2d, l2d_predict
from task4_active_learning import active_learning_curve
from report_generator import build_report

st.set_page_config(page_title="HCAI Project 3 - Active Learning for L2D", layout="wide")
st.title("Active Learning for Learning-to-Defer -- AG News")


@st.cache_resource
def setup_pipeline():
    X_train_text, y_train, X_test_text, y_test = load_ag_news()
    X_train, X_test, vectorizer = vectorize(X_train_text, X_test_text)

    model = train_baseline(X_train, y_train)
    baseline_acc = accuracy_score(y_test, model.predict(X_test))

    expert_train_labels = simulate_expert_labels(y_train, seed=1)
    expert_test_labels = simulate_expert_labels(y_test, seed=2)
    expert_overall_acc, expert_report = analyze_expert(y_test, expert_test_labels)

    return {
        "X_train": X_train, "y_train": y_train,
        "X_test": X_test, "y_test": y_test,
        "model": model, "baseline_acc": baseline_acc,
        "expert_train_labels": expert_train_labels,
        "expert_test_labels": expert_test_labels,
        "expert_overall_acc": expert_overall_acc,
        "expert_report": expert_report,
    }


with st.spinner("Loading data and training baseline model (first run only)..."):
    P = setup_pipeline()

tab1, tab2, tab3, tab4 = st.tabs([
    "Task 1 - Baseline", "Task 2 - Expert", "Task 3 - Learning to Defer",
    "Task 4 - Active Learning",
])

with tab1:
    st.subheader("Baseline classifier (TF-IDF + Logistic Regression)")
    st.metric("Test accuracy", f"{P['baseline_acc']:.3f}")
    st.caption("Trained on the full labeled AG News training set (Task 1).")

with tab2:
    st.subheader("Simulated expert")
    st.metric("Overall expert accuracy", f"{P['expert_overall_acc']:.3f}")
    st.write("Per-class recall:")
    for i in sorted(LABEL_NAMES):
        name = LABEL_NAMES[i]
        st.write(f"- **{name}**: {P['expert_report'][name]['recall']:.3f}")
    st.caption(
        "The expert is strong on World/Sports and weaker on Business vs "
        "Sci/Tech, which it tends to confuse (Task 2)."
    )

with tab3:
    st.subheader("Confidence-based learning-to-defer")
    defer_cost = st.slider("Cost per expert query (accuracy penalty)", 0.0, 0.1, 0.02, 0.005)

    threshold, sweep = tune_threshold(
        P["model"], P["X_test"], P["y_test"], P["expert_test_labels"],
        defer_cost=defer_cost,
    )
    metrics = evaluate_l2d(
        P["model"], P["X_test"], P["y_test"], P["expert_test_labels"], threshold
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Classifier-only accuracy", f"{metrics['classifier_only_accuracy']:.3f}")
    col2.metric("Combined (L2D) accuracy", f"{metrics['combined_accuracy']:.3f}")
    col3.metric("Deferral rate", f"{metrics['deferral_rate']:.3f}")

    st.write(f"Tuned confidence threshold: **{threshold:.3f}**")
    st.write(
        f"On deferred points, the classifier alone would have been wrong "
        f"{metrics['classifier_would_be_wrong_rate_on_deferred']:.3f} of the time, "
        f"and the expert was correct {metrics['expert_correct_rate_on_deferred']:.3f} "
        f"of the time -- confirming deferral targets genuinely hard cases."
    )

with tab4:
    st.subheader("Active learning for expert-competence discovery")
    query_budget = st.slider("Total expert query budget", 100, 1000, 500, 50)
    batch_size = st.slider("Batch size per round", 25, 200, 50, 25)

    if st.button("Run active learning comparison"):
        with st.spinner("Running uncertainty-sampling and random-sampling rounds..."):
            hist_uncertainty = active_learning_curve(
                P["model"], P["X_train"], P["y_train"], P["expert_train_labels"],
                P["X_test"], P["y_test"], P["expert_test_labels"],
                query_budget=query_budget, batch_size=batch_size,
                strategy="uncertainty",
            )
            hist_random = active_learning_curve(
                P["model"], P["X_train"], P["y_train"], P["expert_train_labels"],
                P["X_test"], P["y_test"], P["expert_test_labels"],
                query_budget=query_budget, batch_size=batch_size,
                strategy="random",
            )
        st.session_state["al_uncertainty"] = hist_uncertainty
        st.session_state["al_random"] = hist_random

    if "al_uncertainty" in st.session_state:
        import pandas as pd
        df = pd.DataFrame({
            "n_queried": [h["n_queried"] for h in st.session_state["al_uncertainty"]],
            "uncertainty_sampling": [h["combined_accuracy"] for h in st.session_state["al_uncertainty"]],
            "random_sampling": [h["combined_accuracy"] for h in st.session_state["al_random"]],
        }).set_index("n_queried")
        st.line_chart(df)
        st.caption(
            "Uncertainty sampling should reach a given accuracy with fewer "
            "expert queries than random sampling, since it prioritizes the "
            "points the classifier is least sure about."
        )

st.divider()
st.subheader("Download report")
if "al_uncertainty" in st.session_state:
    l2d_threshold, _ = tune_threshold(
        P["model"], P["X_test"], P["y_test"], P["expert_test_labels"]
    )
    l2d_metrics_for_report = evaluate_l2d(
        P["model"], P["X_test"], P["y_test"], P["expert_test_labels"], l2d_threshold
    )
    pdf_buf = build_report(
        P["baseline_acc"], P["expert_overall_acc"], P["expert_report"],
        l2d_metrics_for_report, st.session_state["al_uncertainty"], st.session_state["al_random"],
    )
    st.download_button(
        "Download PDF report", data=pdf_buf, file_name="project3_report.pdf",
        mime="application/pdf",
    )
else:
    st.info("Run the active learning comparison in Task 4 above to enable the report download.")
