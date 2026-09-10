import streamlit as st

st.set_page_config(
    page_title="HCAI Project 3 - User Study",
    page_icon="🧠",
    layout="centered",
)

st.title("Active Learning for Learning-to-Defer")
st.subheader("HCAI Project 3 — AG News")

st.write(
    """
    Welcome to our Human-Centred Artificial Intelligence study.

    This project investigates how artificial intelligence systems can
    combine automated predictions with expert assistance. In particular,
    we study Learning-to-Defer and Active Learning for classification
    using the AG News dataset.
    """
)

st.divider()

st.header("About the Project")

st.write(
    """
    The project explores how an AI classifier can decide when it should
    make a prediction itself and when it should defer a difficult case
    to an expert. We also investigate how expert queries can be selected
    efficiently through active learning.
    """
)

st.info(
    "Please read the project report before starting the user study."
)

st.divider()

st.header("Project Materials")

# PDF section
st.subheader("📄 Project Report")

st.write(
    """
    The project report explains the motivation, experimental design,
    modelling choices, Learning-to-Defer approach, active learning
    strategy, and user study design.
    """
)

with open("project3_report.pdf", "rb") as pdf_file:
    pdf_data = pdf_file.read()

st.download_button(
    label="📥 Download Project Report (PDF)",
    data=pdf_data,
    file_name="project3_report.pdf",
    mime="application/pdf",
    use_container_width=True,
)

st.divider()

st.header("👤 User Study")

st.write(
    """
    The user study allows participants to interact with the AI-assisted
    classification system and make decisions based on the information
    presented to them.
    """
)

if st.button("▶ Start User Study", use_container_width=True):
    st.switch_page("pages/user_study.py")

st.divider()

st.caption(
    "HCAI Project 3 | Active Learning for Learning-to-Defer | AG News"
)