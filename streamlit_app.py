import streamlit as st
from ats_score import ATSScorer
import subprocess
import sys

# Page configuration
st.set_page_config(
    page_title="ATS Resume Scorer",
    page_icon="📝",
    layout="wide"
)

# Initialize ATS Scorer
@st.cache_resource
def load_scorer():
    try:
        return ATSScorer()
    except Exception as e:
        st.error(f"Error loading the scorer: {str(e)}")
        st.stop()

def main():
    # Header
    st.title("📝 ATS Resume Scorer")
    st.markdown("""
    This tool helps you evaluate how well your resume matches a job description using AI-powered analysis.
    """)
    
    # Create two columns
    col1, col2 = st.columns(2)
    
    # Job Description Input
    with col1:
        st.subheader("Job Description")
        job_description = st.text_area(
            "Paste the job description here",
            height=300,
            placeholder="Paste the complete job description here..."
        )
    
    # Resume Input
    with col2:
        st.subheader("Your Resume")
        resume_text = st.text_area(
            "Paste your resume text here",
            height=300,
            placeholder="Paste your resume text here..."
        )
    
    # Analysis Button
    if st.button("Analyze Resume", type="primary"):
        if not job_description or not resume_text:
            st.error("Please provide both the job description and resume text.")
            return
        
        with st.spinner("Analyzing your resume..."):
            try:
                # Get the scorer instance
                scorer = load_scorer()
                
                # Calculate scores
                results = scorer.score_resume(resume_text, job_description)
                
                # Display Results
                st.markdown("### Results")
                
                # Create three columns for scores
                score_col1, score_col2, score_col3 = st.columns(3)
                
                with score_col1:
                    st.metric(
                        "Overall Match Score",
                        f"{results['overall_score']}%"
                    )
                
                with score_col2:
                    st.metric(
                        "Technical Match Score",
                        f"{results['score_breakdown']['technical_match']}%"
                    )
                
                with score_col3:
                    st.metric(
                        "Content Similarity Score",
                        f"{results['score_breakdown']['content_similarity']}%"
                    )
                
                # Display missing skills if any
                if results['missing_skills']:
                    st.markdown("### Missing Technical Skills")
                    for category, skills in results['missing_skills'].items():
                        if skills:
                            st.markdown(f"**{category.title()}:** {', '.join(skills)}")
                
                # Display feedback
                st.markdown("### Detailed Feedback")
                for feedback in results['feedback']:
                    st.markdown(f"{feedback}")
                
            except Exception as e:
                st.error(f"An error occurred during analysis: {str(e)}")
                st.error("Please make sure you have installed all required dependencies.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    💡 **Tips:**
    - Paste the complete job description and resume text
    - Make sure to include relevant technical skills and keywords
    - The analysis considers both technical skill matching and content similarity
    """)

if __name__ == "__main__":
    main()
