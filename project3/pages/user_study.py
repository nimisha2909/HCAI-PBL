import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np

# Allow the study to use the existing Project 3 files
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FILES_DIR = PROJECT_ROOT / "files"

if str(FILES_DIR) not in sys.path:
    sys.path.insert(0, str(FILES_DIR))

from data_utils import load_ag_news, vectorize, LABEL_NAMES
from task1_baseline import train_baseline
from expert_simulation import simulate_expert_labels


st.set_page_config(
    page_title="HCAI Project 3 - User Study",
    page_icon="🧠",
    layout="centered",
)


# ---------------------------------------------------------
# Load model and data
# ---------------------------------------------------------

@st.cache_resource
def load_study_data():
    X_train_text, y_train, X_test_text, y_test = load_ag_news()

    X_train, X_test, vectorizer = vectorize(
        X_train_text,
        X_test_text
    )

    model = train_baseline(X_train, y_train)

    expert_test_labels = simulate_expert_labels(
        y_test,
        seed=2
    )

    return (
        X_test_text,
        y_test,
        X_test,
        model,
        expert_test_labels,
    )


# ---------------------------------------------------------
# Session state
# ---------------------------------------------------------

if "study_started" not in st.session_state:
    st.session_state.study_started = False

if "current_trial" not in st.session_state:
    st.session_state.current_trial = 0

if "responses" not in st.session_state:
    st.session_state.responses = []

if "finished" not in st.session_state:
    st.session_state.finished = False


# ---------------------------------------------------------
# Introduction
# ---------------------------------------------------------

if not st.session_state.study_started:

    st.title("User Study")
    st.subheader("AI-Assisted News Classification")

    st.write(
        """
        Thank you for participating in this study.

        In this study, you will interact with an AI system that
        classifies short news articles into one of four categories:

        - World
        - Sports
        - Business
        - Sci/Tech

        The purpose of the study is to investigate how people interact
        with an AI classification system and how expert assistance can
        be used for difficult cases.
        """
    )

    st.divider()

    st.header("Study Instructions")

    st.write(
        """
        For each news article:

        1. Read the article carefully.
        2. Review the AI's prediction.
        3. Decide whether you agree with the AI prediction.
        4. If you are uncertain, you may choose to ask for expert help.
        5. Submit your final decision.

        There are no penalties for making an incorrect decision.
        Please base your answers on your own judgement.
        """
    )

    st.info(
        """
        Your participation is voluntary. You may stop the study at
        any time. No identifying information is required for this
        demonstration study.
        """
    )

    st.divider()

    st.header("Consent")

    consent = st.checkbox(
        "I have read the information above and agree to participate "
        "in the user study."
    )

    if st.button(
        "Start User Study",
        disabled=not consent,
        use_container_width=True,
    ):
        st.session_state.study_started = True
        st.rerun()

    st.stop()


# ---------------------------------------------------------
# Load study pipeline
# ---------------------------------------------------------

with st.spinner("Preparing the study..."):
    (
        X_test_text,
        y_test,
        X_test,
        model,
        expert_test_labels,
    ) = load_study_data()


# ---------------------------------------------------------
# Study completion
# ---------------------------------------------------------

if st.session_state.finished:

    st.title("Study Completed")

    st.success(
        "Thank you for participating in the user study."
    )

    responses = st.session_state.responses

    if responses:

        df = pd.DataFrame(responses)

        st.subheader("Your Study Summary")

        st.metric(
            "Trials completed",
            len(df)
        )

        st.write(
            """
            Your responses have been recorded for the current
            study session.
            """
        )

        csv_data = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Your Responses",
            data=csv_data,
            file_name="user_study_responses.csv",
            mime="text/csv",
        )

    if st.button("Return to Landing Page"):
        st.switch_page("app.py")

    st.stop()


# ---------------------------------------------------------
# Current trial
# ---------------------------------------------------------

TOTAL_TRIALS = 10

trial = st.session_state.current_trial

if trial >= TOTAL_TRIALS:
    st.session_state.finished = True
    st.rerun()


# Select reproducible study examples
rng = np.random.default_rng(42)

if "trial_indices" not in st.session_state:

    st.session_state.trial_indices = rng.choice(
        len(X_test_text),
        size=TOTAL_TRIALS,
        replace=False,
    ).tolist()

index = st.session_state.trial_indices[trial]

article_text = X_test_text[index]

true_label = int(y_test[index])

ai_probabilities = model.predict_proba(
    X_test[index]
)

ai_prediction = int(
    model.predict(X_test[index])[0]
)

ai_confidence = float(
    np.max(ai_probabilities)
)

expert_prediction = int(
    expert_test_labels[index]
)


# ---------------------------------------------------------
# Progress
# ---------------------------------------------------------

st.title("AI-Assisted News Classification")

st.progress(
    (trial + 1) / TOTAL_TRIALS
)

st.caption(
    f"Trial {trial + 1} of {TOTAL_TRIALS}"
)


# ---------------------------------------------------------
# Article
# ---------------------------------------------------------

st.subheader("News Article")

st.container(border=True)

st.write(article_text)


st.divider()


# ---------------------------------------------------------
# AI recommendation
# ---------------------------------------------------------

st.subheader("AI Recommendation")

st.info(
    f"""
    The AI predicts:

    **{LABEL_NAMES[ai_prediction]}**

    AI confidence: **{ai_confidence:.1%}**
    """
)


# ---------------------------------------------------------
# Expert option
# ---------------------------------------------------------

st.subheader("What would you like to do?")

decision = st.radio(
    "Choose one option:",
    [
        "Accept the AI prediction",
        "Ask the expert for help",
        "Reject the AI prediction and choose another category",
    ],
    key=f"decision_{trial}",
)


# ---------------------------------------------------------
# Expert information
# ---------------------------------------------------------

final_prediction = ai_prediction
used_expert = False

if decision == "Ask the expert for help":

    used_expert = True

    st.warning(
        f"""
        Expert recommendation:

        **{LABEL_NAMES[expert_prediction]}**
        """
    )

    expert_choice = st.radio(
        "After seeing the expert recommendation, choose your final answer:",
        list(LABEL_NAMES.values()),
        key=f"expert_choice_{trial}",
    )

    final_prediction = [
        label for label in LABEL_NAMES
        if LABEL_NAMES[label] == expert_choice
    ][0]


elif decision == "Reject the AI prediction and choose another category":

    alternative = st.radio(
        "Select your final category:",
        list(LABEL_NAMES.values()),
        key=f"alternative_{trial}",
    )

    final_prediction = [
        label for label in LABEL_NAMES
        if LABEL_NAMES[label] == alternative
    ][0]


# ---------------------------------------------------------
# Confidence / difficulty
# ---------------------------------------------------------

st.divider()

participant_confidence = st.slider(
    "How confident are you in your final answer?",
    min_value=1,
    max_value=5,
    value=3,
    help="1 = Not confident at all, 5 = Very confident",
    key=f"confidence_{trial}",
)

difficulty = st.slider(
    "How difficult was this article?",
    min_value=1,
    max_value=5,
    value=3,
    help="1 = Very easy, 5 = Very difficult",
    key=f"difficulty_{trial}",
)


# ---------------------------------------------------------
# Submit
# ---------------------------------------------------------

if st.button(
    "Submit Answer",
    type="primary",
    use_container_width=True,
):

    response = {
        "trial": trial + 1,
        "ai_prediction": LABEL_NAMES[ai_prediction],
        "ai_confidence": ai_confidence,
        "participant_decision": decision,
        "used_expert": used_expert,
        "final_prediction": LABEL_NAMES[final_prediction],
        "participant_confidence": participant_confidence,
        "difficulty": difficulty,
        "correct": final_prediction == true_label,
    }

    st.session_state.responses.append(response)

    # save this response to disk right away
    import os
    os.makedirs("results", exist_ok=True)
    log_file = "results/all_responses.csv"
    file_exists = os.path.exists(log_file)
    df_row = pd.DataFrame([response])
    df_row.to_csv(log_file, mode="a", header=not file_exists, index=False)
    st.session_state.current_trial += 1
    st.rerun()