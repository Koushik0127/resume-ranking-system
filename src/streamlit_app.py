# streamlit_app.py
# Streamlit UI for Resume Ranking System

import streamlit as st
import pandas as pd
import pickle
import os
from io import StringIO
from main import ResumeRankingSystem

import re

# Configure page
st.set_page_config(  
    page_title="Resume Ranking System",
    page_icon="📄",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .ranking-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .score-badge {
        background-color: #1f77b4;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🎯 ML Resume Ranking System</h1>', unsafe_allow_html=True)

# Load model function
@st.cache_resource
def load_ranking_system():
    """Load the trained model and components"""
    try:
        import main  # make sure this is the file you renamed from maain.py
        import pickle

        with open('resume_ranking_model.pkl', 'rb') as f:
            components = pickle.load(f, encoding='latin1')

        
        from main import ResumeRankingSystem  # Import your main class
        return ResumeRankingSystem(
            components['model'],
            components['feature_extractor'],
            components['resumes_df'],
            components['preprocessor']
        )
    except FileNotFoundError:
        st.error("Model file not found. Please run the main training script first.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Initialize session state
if 'resumes_uploaded' not in st.session_state:
    st.session_state.resumes_uploaded = False
if 'custom_resumes' not in st.session_state:
    st.session_state.custom_resumes = []

# Sidebar
st.sidebar.title("⚙️ Settings")
ranking_system = load_ranking_system()

if ranking_system is None:
    st.stop()

# Main content
tab1, tab2, tab3 = st.tabs(["🔍 Rank Resumes", "📤 Upload Resumes", "📊 Analytics"])

with tab1:
    st.subheader("Job Description Input")
    
    # Predefined job examples
    job_examples = {
        "Data Scientist": "Looking for a Data Scientist with Python and machine learning experience. Must know TensorFlow, statistics, and have experience with large datasets. Will build predictive models and work with cross-functional teams.",
        
        "Full Stack Developer": "Need a Full Stack Developer proficient in React, Node.js, and JavaScript. Database experience required. Will work on both frontend and backend systems for our web applications.",
        
        "DevOps Engineer": "Seeking DevOps Engineer with AWS, Docker, and Kubernetes expertise. CI/CD pipeline experience preferred. Will manage cloud infrastructure and automate deployment processes.",
        
        "Product Manager": "Product Manager needed to drive product strategy and roadmap. Need experience with agile methodologies and user research. Will work with engineering and design teams.",
        
        "Custom": "Write your own job description..."
    }
    
    job_type = st.selectbox("Choose a job type or write custom:", list(job_examples.keys()))
    
    if job_type == "Custom":
        job_description = st.text_area(
            "Enter Job Description:",
            height=150,
            placeholder="Describe the role, required skills, experience level, and any specific technologies..."
        )
    else:
        job_description = st.text_area(
            "Job Description:",
            value=job_examples[job_type],
            height=150
        )
    
    # Ranking parameters
    col1, col2 = st.columns(2)
    with col1:
        top_k = st.slider("Number of top resumes to show:", 1, 10, 5)
    with col2:
        min_score = st.slider("Minimum relevance score:", 0.0, 2.0, 0.0, 0.1)
    
    # Rank button
    if st.button("🎯 Rank Resumes", type="primary"):
        if job_description.strip():
            with st.spinner("Analyzing resumes and calculating rankings..."):
                try:
                    # Get rankings
                    rankings = ranking_system.rank_resumes(job_description, top_k=top_k)
                    
                    # Filter by minimum score
                    rankings = rankings[rankings['relevance_score'] >= min_score]
                    
                    if len(rankings) > 0:
                        st.success(f"Found {len(rankings)} relevant resumes!")
                        
                        # Display rankings
                        for idx, (_, resume) in enumerate(rankings.iterrows(), 1):
                            score = resume['relevance_score']
                            score_color = "#28a745" if score >= 1.5 else "#ffc107" if score >= 0.8 else "#dc3545"
                            
                            st.markdown(f"""
                            <div class="ranking-card">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <h4 style="margin: 0; color: #333;">#{idx} {resume['name']}</h4>
                                    <span style="background-color: {score_color}; color: white; padding: 0.2rem 0.8rem; 
                                          border-radius: 15px; font-weight: bold;">
                                        Score: {score:.3f}
                                    </span>
                                </div>
                                <p style="margin: 0.5rem 0; color: #666;"><strong>Skills:</strong> {resume['skills']}</p>
                                <p style="margin: 0; color: #555; font-size: 0.9rem;">{resume['resume_text'][:200]}...</p>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.warning("No resumes meet the minimum score criteria. Try lowering the minimum score.")
                        
                except Exception as e:
                    st.error(f"Error during ranking: {str(e)}")
        else:
            st.warning("Please enter a job description.")

with tab2:
    st.subheader("Upload Additional Resumes")
    st.info("Add custom resumes to expand the database (Currently using built-in sample resumes)")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Choose resume files",
        accept_multiple_files=True,
        type=['txt', 'pdf', 'docx'],
        help="Upload text files, PDFs, or Word documents"
    )
    
    # Manual resume input
    st.subheader("Or Add Resume Manually")
    with st.form("manual_resume"):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Candidate Name")
            skills = st.text_input("Skills (comma-separated)")
        with col2:
            experience = st.text_input("Experience")
            education = st.text_input("Education")
        
        resume_text = st.text_area("Full Resume Text", height=150)
        
        if st.form_submit_button("Add Resume"):
            if name and resume_text:
                new_resume = {
                    'name': name,
                    'skills': skills,
                    'experience': experience,
                    'education': education,
                    'resume_text': resume_text
                }
                st.session_state.custom_resumes.append(new_resume)
                st.success(f"Resume for {name} added successfully!")
            else:
                st.error("Please fill in at least the name and resume text.")
    
    # Show uploaded resumes
    if st.session_state.custom_resumes:
        st.subheader("Custom Resumes Added")
        for i, resume in enumerate(st.session_state.custom_resumes):
            with st.expander(f"{resume['name']} - Resume {i+1}"):
                st.write(f"**Skills:** {resume['skills']}")
                st.write(f"**Experience:** {resume['experience']}")
                st.write(f"**Education:** {resume['education']}")
                st.write(f"**Resume Text:** {resume['resume_text'][:300]}...")

with tab3:
    st.subheader("📊 System Analytics")
    
    # Model performance metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Resumes", len(ranking_system.resumes_df))
    with col2:
        st.metric("Features Used", "6")
    with col3:
        st.metric("Model Type", "Random Forest")
    
    # Sample statistics
    st.subheader("Resume Database Overview")
    
    # Skills analysis
    if 'skills' in ranking_system.resumes_df.columns:
        all_skills = []
        for skills_str in ranking_system.resumes_df['skills'].dropna():
            skills_list = [skill.strip() for skill in skills_str.split(',')]
            all_skills.extend(skills_list)
        
        skill_counts = pd.Series(all_skills).value_counts().head(10)
        
        st.subheader("Top Skills in Database")
        chart_data = pd.DataFrame({
            'Skill': skill_counts.index,
            'Count': skill_counts.values
        })
        st.bar_chart(chart_data.set_index('Skill'))
    
    # Feature importance (if available)
    try:
        if hasattr(ranking_system.model, 'feature_importances_'):
            st.subheader("Model Feature Importance")
            feature_names = ['TF-IDF Similarity', 'Resume Length', 'Job Length', 
                           'Common Words', 'Resume Vocabulary', 'Job Vocabulary']
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': ranking_system.model.feature_importances_
            }).sort_values('Importance', ascending=True)
            
            st.bar_chart(importance_df.set_index('Feature'))
    except:
        st.info("Feature importance not available for this model type.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p>🤖 ML-Powered Resume Ranking System | Built with Streamlit & scikit-learn</p>
    <p>💡 Tip: Use specific technical terms in job descriptions for better matching accuracy</p>
</div>
""", unsafe_allow_html=True)

# Sidebar information
st.sidebar.markdown("---")
st.sidebar.subheader("🔍 How It Works")
st.sidebar.markdown("""
1. **Text Processing**: Cleans and normalizes resume/job text
2. **Feature Extraction**: Uses TF-IDF and similarity metrics
3. **ML Prediction**: Random Forest model predicts relevance
4. **Ranking**: Sorts resumes by predicted relevance score

**Score Interpretation:**
- 🟢 > 1.5: Highly Relevant
- 🟡 0.8-1.5: Moderately Relevant  
- 🔴 < 0.8: Less Relevant
""")

st.sidebar.markdown("---")
st.sidebar.subheader("⚡ Quick Tips")
st.sidebar.markdown("""
- Be specific about required skills
- Include experience level requirements
- Mention specific technologies/tools
- Use industry-standard terminology
""")