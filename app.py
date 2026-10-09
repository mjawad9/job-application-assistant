import os
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader

from crew import build_crew

BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR / "app.env")

st.set_page_config(page_title="Job Application Assistant", page_icon="💼", layout="wide")
st.title("💼 Job Application Assistant")
st.caption("A CrewAI team of 4 agents: CV Analyzer, Job Matcher, Cover Letter Writer, Interview Coach.")

with st.sidebar:
    st.header("Settings")
    model = st.text_input(
        "Model",
        value="gemini/gemini-3.8-flash",
        help="Examples: gemini/gemini-3.8-flash, gpt-4o-mini.",
    )
    api_key = st.text_input(
        "Gemini API key",
        type="password",
        help="Leave empty if your app.env file already has the key.",
    )
    if api_key:
        os.environ["GEMINI_API_KEY"] = api_key
    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        st.success("API key found")
    else:
        st.error("No API key found")

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Your CV")
    uploaded = st.file_uploader("Upload CV (PDF)", type=["pdf"])
    cv_text = ""
    if uploaded:
        reader = PdfReader(uploaded)
        cv_text = "\n".join(page.extract_text() or "" for page in reader.pages)
        st.success(f"CV read: {len(cv_text)} characters")
    else:
        cv_text = st.text_area("...or paste your CV text", height=250)

with col2:
    st.subheader("2. Job description")
    job_description = st.text_area("Paste the job description", height=330)

if st.button("Run my application crew", type="primary"):
    if not cv_text.strip() or not job_description.strip():
        st.warning("Please provide both your CV and the job description.")
    else:
        with st.spinner("The agents are working... this can take a minute or two."):
            outputs = None
            last_error = None
            for attempt in range(4):
                try:
                    crew = build_crew(cv_text, job_description, model)
                    result = crew.kickoff()
                    outputs = [t.raw for t in result.tasks_output]
                    break
                except Exception as e:
                    last_error = e
                    if "503" in str(e) or "UNAVAILABLE" in str(e):
                        time.sleep(10 * (attempt + 1))  # wait 10s, 20s, 30s, 40s
                    else:
                        break

            if outputs is None:
                st.error(f"Something went wrong: {last_error}")
                st.stop()
            st.session_state["outputs"] = outputs

if "outputs" in st.session_state:
    outputs = st.session_state["outputs"]
    tab1, tab2, tab3, tab4 = st.tabs(
        ["CV analysis", "Job match", "Cover letter", "Interview prep"]
    )
    for tab, text in zip((tab1, tab2, tab3, tab4), outputs):
        with tab:
            st.markdown(text)

    st.download_button(
        "Download cover letter",
        data=outputs[2],
        file_name="cover_letter.txt",
    )