import streamlit as st
from ats_score import ATSScorer
import subprocess
import sys

# Page configuration
st.set_page_config(
    page_title="ATS Resume Scorer",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for dark theme
st.markdown("""
<style>
    /* Dark theme colors */
    :root {
        --background-color: #0e1117;
        --secondary-background-color: #262730;
        --text-color: #ffffff;
        --secondary-text-color: #b0b0b0;
        --accent-color: #4CAF50;
        --metric-background: #1e1e2d;
    }

    /* Main container styling */
    .main {
        background-color: var(--background-color);
        color: var(--text-color);
    }

    /* Input areas */
    .stTextArea textarea {
        background-color: var(--secondary-background-color);
        color: var(--text-color);
        border: 1px solid #404040;
        border-radius: 5px;
    }

    /* Metric cards */
    .metric-container {
        background-color: var(--metric-background);
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border: 1px solid #404040;
    }

    /* Section headers */
    .section-header {
        color: var(--text-color);
        font-size: 1.5rem;
        font-weight: bold;
        margin: 20px 0;
    }

    /* Results container */
    .results-container {
        background-color: var(--secondary-background-color);
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        border: 1px solid #404040;
    }
    
    /* Metric value styling */
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: var(--accent-color);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

def format_percentage(value):
    """Format percentage to 2 decimal places"""
    return f"{value:.2f}%"

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
    st.markdown("Evaluate how well your resume matches a job description using AI-powered analysis.")
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    # Job Description Input
    with col1:
        st.subheader("Job Description")
        job_description = st.text_area(
            label="Job Description Text",
            key="job_desc",
            height=300,
            placeholder="Paste the complete job description here...",
            label_visibility="collapsed"
        )
    
    # Resume Input
    with col2:
        st.subheader("Your Resume")
        resume_text = st.text_area(
            label="Resume Text",
            key="resume",
            height=300,
            placeholder="Paste your resume text here...",
            label_visibility="collapsed"
        )
    
    # Center the analyze button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_button = st.button("Analyze Resume", type="primary", use_container_width=True)
    
    if analyze_button:
        if not job_description or not resume_text:
            st.error("Please provide both the job description and resume text.")
            return
        
        with st.spinner("Analyzing your resume..."):
            try:
                scorer = load_scorer()
                results = scorer.score_resume(resume_text, job_description)
                
                # Results section
                st.markdown("### Analysis Results")
                
                # Score metrics in columns
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                    st.metric(
                        "Overall Match Score",
                        format_percentage(results['overall_score'])
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                    st.metric(
                        "Technical Match",
                        format_percentage(results['score_breakdown']['technical_match'])
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col3:
                    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                    st.metric(
                        "Content Similarity",
                        format_percentage(results['score_breakdown']['content_similarity'])
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Context understanding score
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.metric(
                    "Context Understanding",
                    format_percentage(results['score_breakdown']['context_understanding'])
                )
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Missing skills section
                if results['missing_skills']:
                    st.markdown("### Missing Technical Skills")
                    st.markdown('<div class="results-container">', unsafe_allow_html=True)
                    for category, skills in results['missing_skills'].items():
                        if skills:
                            st.markdown(f"**{category.title()}:** {', '.join(skills)}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Feedback section
                st.markdown("### Detailed Feedback")
                st.markdown('<div class="results-container">', unsafe_allow_html=True)
                for feedback in results['feedback']:
                    st.markdown(feedback)
                st.markdown('</div>', unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"An error occurred during analysis: {str(e)}")
                st.error("Please make sure you have installed all required dependencies.")
    
    # Tips section
    st.markdown("---")
    st.markdown('<div class="results-container">', unsafe_allow_html=True)
    st.markdown("""
    💡 **Tips for Best Results:**
    - Paste the complete job description and resume text
    - Make sure to include relevant technical skills and keywords
    - The analysis considers both technical skill matching and content similarity
    """)
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
