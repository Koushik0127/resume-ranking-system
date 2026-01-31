# src/streamlit_app.py
import streamlit as st
from main import ResumeRankingSystem
from model_utils import TextPreprocessor, FeatureExtractor
import pickle

st.set_page_config(page_title="Resume Ranking System")
st.title("📄 Resume Ranking System")

# Load model
try:
    rrs = ResumeRankingSystem("resume_ranking_model.pkl")
except Exception as e:
    st.error(f"Error loading model: {e}")
    rrs = None

job_desc = st.text_area("Job Description")

resumes_input = st.text_area(
    "Paste resumes (separate with ----)",
    height=200,
    placeholder="John Smith: Python ML data science experience ---- Aisha Khan: AWS Docker DevOps"
)

if st.button("Rank Resumes") and rrs:
    if not job_desc.strip() or not resumes_input.strip():
        st.warning("Please enter job description and at least one resume!")
    else:
        resume_list = [
            {"name": r.split(":")[0].strip(), "text": r.split(":")[1].strip()}
            for r in resumes_input.split("----") if ":" in r
        ]

        # Temporarily replace loaded resumes for demo
        rrs.resumes = resume_list

        try:
            results = rrs.rank_resumes(job_desc, top_k=5)
            for i, (resume, score) in enumerate(results, 1):
                st.subheader(f"Rank {i}: {resume['name']}")
                st.write(f"Score: {score:.3f}")
                st.write(f"Resume Text: {resume['text']}")
        except Exception as e:
            st.error(f"Error ranking resumes: {e}")
