import streamlit as st
from enhanced_ats_score import EnhancedATSScorer, MistralAIATSScorer
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import json
import re

# Page configuration
st.set_page_config(
    page_title="Advanced ATS Resume Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for improved dark theme
st.markdown("""
<style>
    /* Dark theme colors */
    :root {
        --background-color: #0e1117;
        --secondary-background-color: #1e1e2e;
        --text-color: #ffffff;
        --secondary-text-color: #b0b0b0;
        --accent-color: #7d56f4;
        --success-color: #4CAF50;
        --warning-color: #ff9800;
        --danger-color: #f44336;
        --metric-background: #262639;
        --chart-background: #2d2d44;
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
        border-radius: 8px;
        padding: 12px;
        font-family: 'Roboto Mono', monospace;
    }

    /* Metric cards */
    .metric-container {
        background-color: var(--metric-background);
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border: 1px solid #404040;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s ease-in-out;
    }
    
    .metric-container:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
    }

    /* Section headers */
    .section-header {
        color: var(--text-color);
        font-size: 1.5rem;
        font-weight: bold;
        margin: 20px 0;
        border-bottom: 2px solid var(--accent-color);
        padding-bottom: 8px;
    }

    /* Results container */
    .results-container {
        background-color: var(--secondary-background-color);
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        border: 1px solid #404040;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Metric value styling */
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: var(--accent-color);
    }
    
    /* Score ranges */
    .score-high {
        color: var(--success-color) !important;
    }
    
    .score-medium {
        color: var(--warning-color) !important;
    }
    
    .score-low {
        color: var(--danger-color) !important;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: var(--metric-background);
        border-radius: 4px 4px 0px 0px;
        padding: 10px 16px;
        border: 1px solid #404040;
    }

    .stTabs [aria-selected="true"] {
        background-color: var(--accent-color);
        border-color: var(--accent-color);
    }
    
    /* Button styling */
    .stButton>button {
        background-color: var(--accent-color);
        color: white;
        border-radius: 8px;
        border: none;
        padding: 12px 20px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #6a46e5;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background-color: var(--accent-color);
    }
    
    /* Info boxes */
    .info-box {
        background-color: rgba(125, 86, 244, 0.1);
        border-left: 4px solid var(--accent-color);
        padding: 10px 15px;
        margin: 10px 0;
        border-radius: 0 4px 4px 0;
    }
    
    /* Skill tags */
    .skill-tag {
        display: inline-block;
        background-color: rgba(125, 86, 244, 0.2);
        border-radius: 16px;
        padding: 5px 10px;
        margin: 3px;
        font-size: 0.85em;
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

def get_score_class(score):
    """Return CSS class based on score"""
    if score >= 75:
        return "score-high"
    elif score >= 50:
        return "score-medium"
    else:
        return "score-low"

def section_to_emoji(section):
    """Return emoji for section type"""
    section_emojis = {
        'work_experience': '💼',
        'projects': '🚀',
        'skills': '🔧',
        'education': '🎓',
        'certifications': '🏆',
        'other': '📄'
    }
    return section_emojis.get(section, '📄')

@st.cache_resource
def load_scorer(use_mistral=False):
    try:
        if use_mistral:
            return MistralAIATSScorer()
        else:
            return EnhancedATSScorer()
    except Exception as e:
        st.error(f"Error loading the scorer: {str(e)}")
        st.stop()

def create_section_chart(section_analysis):
    """Create radar chart for section analysis"""
    categories = []
    scores = []
    
    for section, data in section_analysis.items():
        categories.append(f"{section_to_emoji(section)} {section.replace('_', ' ').title()}")
        scores.append(data['score'])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=scores,
        theta=categories,
        fill='toself',
        fillcolor='rgba(125, 86, 244, 0.2)',
        line=dict(color='#7d56f4', width=2),
        name='Section Match'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(color='white'),
            ),
            angularaxis=dict(
                tickfont=dict(color='white'),
            ),
            bgcolor='rgba(30, 30, 46, 0.8)'
        ),
        paper_bgcolor='rgba(30, 30, 46, 0)',
        plot_bgcolor='rgba(30, 30, 46, 0)',
        font=dict(color='white'),
        margin=dict(l=40, r=40, t=20, b=20),
        height=400,
    )
    
    return fig

def create_skills_chart(missing_skills):
    """Create horizontal bar chart for missing skills"""
    categories = []
    counts = []
    
    for category, skills in missing_skills.items():
        if skills:
            categories.append(category.replace('_', ' ').title())
            counts.append(len(skills))
    
    if not categories:
        return None
    
    df = pd.DataFrame({
        'Category': categories,
        'Missing Skills': counts
    })
    
    fig = px.bar(
        df, 
        x='Missing Skills', 
        y='Category',
        orientation='h',
        text='Missing Skills',
        color='Missing Skills',
        color_continuous_scale=['#4CAF50', '#ff9800', '#f44336'],
    )
    
    fig.update_layout(
        paper_bgcolor='rgba(30, 30, 46, 0)',
        plot_bgcolor='rgba(30, 30, 46, 0)',
        font=dict(color='white'),
        margin=dict(l=10, r=10, t=20, b=20),
        height=300,
        xaxis_title=None,
        yaxis_title=None,
        coloraxis_showscale=False,
    )
    
    fig.update_traces(
        textposition='outside',
        textfont=dict(color='white'),
    )
    
    return fig

def display_resume_preview(sections):
    """Display resume preview with section highlighting"""
    st.markdown("### Resume Section Analysis")
    
    for section, content in sections.items():
        with st.expander(f"{section_to_emoji(section)} {section.replace('_', ' ').title()}", expanded=False):
            st.markdown(f"```{content[:500]}{'...' if len(content) > 500 else ''}```")

def main():
    # Sidebar configuration
    st.sidebar.image("https://i.imgur.com/8PsD6Dj.png", width=100)
    st.sidebar.title("Advanced ATS Settings")
    
    analysis_mode = st.sidebar.radio(
        "Analysis Mode",
        ["Standard", "Enhanced (with Mistral AI)"],
        help="Enhanced mode uses Mistral AI for more detailed analysis"
    )
    
    st.sidebar.markdown("---")
    
    with st.sidebar.expander("🎚️ Analysis Parameters", expanded=False):
        semantic_weight = st.slider("Semantic Similarity Weight", 0.1, 0.5, 0.35, 0.05)
        technical_weight = st.slider("Technical Skills Weight", 0.3, 0.7, 0.45, 0.05)
        keyword_weight = st.slider("Keyword Match Weight", 0.1, 0.3, 0.2, 0.05)
    
    with st.sidebar.expander("💡 Tips & Tricks", expanded=True):
        st.markdown("""
        - Use plain text from your resume
        - ATS values skills in experience sections more than just in skills list
        - Quantify achievements with numbers
        - Use industry-standard job titles
        - Match keywords from the job posting exactly
        """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("## About")
    st.sidebar.info(
        """
        This enhanced ATS analyzer evaluates your resume against job descriptions using advanced NLP and ML techniques,
        including section-based analysis and contextual understanding.
        """
    )
    
    # Header
    st.title("🧠 Advanced ATS Resume Analyzer")
    st.markdown(
        """
        Evaluate your resume against job descriptions using section-based analysis and AI-powered matching.
        """
    )
    
    # Create tab structure
    tab1, tab2 = st.tabs(["📊 Analysis", "📘 Learn"])
    
    with tab1:
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
            
            with st.spinner("Analyzing your resume with advanced AI..."):
                try:
                    # Use the appropriate scorer based on selected mode
                    use_mistral = analysis_mode == "Enhanced (with Mistral AI)"
                    scorer = load_scorer(use_mistral)
                    
                    # Analyze the resume
                    results = scorer.score_resume(resume_text, job_description)
                    
                    # Get resume sections
                    resume_sections = scorer.analyze_resume_sections(resume_text)
                    
                    # Results section
                    st.markdown('<div class="section-header">📊 Analysis Results</div>', unsafe_allow_html=True)
                    
                    # Overall score with progress bar
                    score_class = get_score_class(results['overall_score'])
                    st.markdown(f"""
                    <div class="metric-container">
                        <h2>Overall ATS Match Score</h2>
                        <div class="metric-value {score_class}">{results['overall_score']}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.progress(min(results['overall_score']/100, 1.0))
                    
                    # Score breakdown cards in 3 columns
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        tech_score = results['score_breakdown']['technical_match']
                        tech_class = get_score_class(tech_score)
                        st.markdown(f"""
                        <div class="metric-container">
                            <h3>Technical Match</h3>
                            <div class="metric-value {tech_class}">{tech_score}%</div>
                            <p>Skills alignment with job requirements</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        semantic_score = results['score_breakdown']['semantic_similarity']
                        semantic_class = get_score_class(semantic_score)
                        st.markdown(f"""
                        <div class="metric-container">
                            <h3>Semantic Similarity</h3>
                            <div class="metric-value {semantic_class}">{semantic_score}%</div>
                            <p>Overall content relevance to the position</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        keyword_score = results['score_breakdown']['keyword_match']
                        keyword_class = get_score_class(keyword_score)
                        st.markdown(f"""
                        <div class="metric-container">
                            <h3>Keyword Match</h3>
                            <div class="metric-value {keyword_class}">{keyword_score}%</div>
                            <p>Key terms and phrases alignment</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Section tabs
                    section_tab1, section_tab2, section_tab3 = st.tabs(["📋 Section Analysis", "🧩 Skills Gap", "💬 Feedback"])
                    
                    with section_tab1:
                        st.markdown("### Resume Section Effectiveness")
                        
                        # Radar chart for section analysis
                        if results['section_analysis']:
                            fig = create_section_chart(results['section_analysis'])
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Section details table
                            st.markdown("### Section Details")
                            section_data = []
                            for section, data in results['section_analysis'].items():
                                section_data.append({
                                    "Section": section.replace('_', ' ').title(),
                                    "Match Score": f"{data['score']}%",
                                    "Importance Weight": data['importance'],
                                    "Skills Found": data['skills_found'],
                                })
                            
                            section_df = pd.DataFrame(section_data)
                            st.dataframe(section_df, hide_index=True, use_container_width=True)
                            
                            # Section preview
                            display_resume_preview(resume_sections)
                    
                    with section_tab2:
                        st.markdown("### Missing Skills Analysis")
                        
                        # Missing skills visualization
                        if results['missing_skills']:
                            fig = create_skills_chart(results['missing_skills'])
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Detailed missing skills
                            st.markdown("### Missing Skills by Category")
                            for category, skills in results['missing_skills'].items():
                                if skills:
                                    with st.expander(f"{category.replace('_', ' ').title()} ({len(skills)})", expanded=True):
                                        st.markdown('<div style="display: flex; flex-wrap: wrap;">', unsafe_allow_html=True)
                                        for skill in skills:
                                            st.markdown(f'<div class="skill-tag">{skill}</div>', unsafe_allow_html=True)
                                        st.markdown('</div>', unsafe_allow_html=True)
                        else:
                            st.success("Great job! No significant skill gaps were detected.")
                    
                    with section_tab3:
                        st.markdown("### Detailed Improvement Feedback")
                        
                        # Feedback section
                        st.markdown('<div class="results-container">', unsafe_allow_html=True)
                        for feedback in results['feedback']:
                            st.markdown(feedback)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Action plan
                        st.markdown("### Actionable Next Steps")
                        st.markdown('<div class="info-box">', unsafe_allow_html=True)
                        st.markdown("""
                        1. **Address skill gaps** by adding the missing skills to relevant experience sections
                        2. **Improve section scores** by focusing on the lowest performing sections
                        3. **Align keywords** more closely with the job description
                        4. **Quantify achievements** with specific metrics and results
                        5. **Reanalyze** after making changes to track improvements
                        """)
                        st.markdown('</div>', unsafe_allow_html=True)
                
                except Exception as e:
                    st.error(f"An error occurred during analysis: {str(e)}")
                    st.error("Please make sure you have installed all required dependencies.")
    
    with tab2:
        st.markdown("## How Modern ATS Systems Work")
        
        st.markdown("""
        ### Key Factors in ATS Evaluation
        
        Modern Applicant Tracking Systems (ATS) do much more than simple keyword matching:
        
        1. **Contextual Understanding**: Skills mentioned in work experience carry more weight than in a skills section
        
        2. **Section Analysis**: Different resume sections have different importance weights
        
        3. **Semantic Matching**: Systems understand conceptual similarity, not just exact keyword matches
        
        4. **Position Relevance**: Recent experience is valued more than older roles
        
        5. **Quantified Achievements**: Numbers and metrics increase the perceived value of experiences
        """)
        
        st.markdown("---")
        
        st.markdown("### How This Tool Differs From Basic ATS Checkers")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Basic ATS Checkers")
            st.markdown("""
            - Simple keyword matching
            - Equal weighting of all resume sections
            - No context awareness
            - Generic feedback
            - Limited technical skill recognition
            """)
        
        with col2:
            st.markdown("#### Our Advanced Analyzer")
            st.markdown("""
            - Section-based analysis with weighted scoring
            - Context-aware skill identification
            - Semantic understanding of content
            - Specific, actionable feedback
            - Comprehensive technical skill categorization
            """)
        
        if analysis_mode == "Enhanced (with Mistral AI)":
            st.markdown("---")
            st.markdown("### Mistral AI Enhancement")
            st.markdown("""
            Our enhanced mode leverages Mistral AI to provide:
            
            - More nuanced understanding of job requirements
            - Personalized improvement suggestions
            - Industry-specific insights
            - Competitive analysis against typical applicants
            - Advanced natural language understanding
            """)

if __name__ == "__main__":
    main()


# import streamlit as st
# from ats_score import ATSScorer
# import subprocess
# import sys

# # Page configuration
# st.set_page_config(
#     page_title="ATS Resume Scorer",
#     page_icon="📝",
#     layout="wide",
#     initial_sidebar_state="collapsed"
# )

# # Custom CSS for dark theme
# st.markdown("""
# <style>
#     /* Dark theme colors */
#     :root {
#         --background-color: #0e1117;
#         --secondary-background-color: #262730;
#         --text-color: #ffffff;
#         --secondary-text-color: #b0b0b0;
#         --accent-color: #4CAF50;
#         --metric-background: #1e1e2d;
#     }

#     /* Main container styling */
#     .main {
#         background-color: var(--background-color);
#         color: var(--text-color);
#     }

#     /* Input areas */
#     .stTextArea textarea {
#         background-color: var(--secondary-background-color);
#         color: var(--text-color);
#         border: 1px solid #404040;
#         border-radius: 5px;
#     }

#     /* Metric cards */
#     .metric-container {
#         background-color: var(--metric-background);
#         padding: 20px;
#         border-radius: 10px;
#         margin: 10px 0;
#         border: 1px solid #404040;
#     }

#     /* Section headers */
#     .section-header {
#         color: var(--text-color);
#         font-size: 1.5rem;
#         font-weight: bold;
#         margin: 20px 0;
#     }

#     /* Results container */
#     .results-container {
#         background-color: var(--secondary-background-color);
#         padding: 20px;
#         border-radius: 10px;
#         margin: 20px 0;
#         border: 1px solid #404040;
#     }
    
#     /* Metric value styling */
#     .metric-value {
#         font-size: 2rem;
#         font-weight: bold;
#         color: var(--accent-color);
#     }
    
#     /* Hide Streamlit branding */
#     #MainMenu {visibility: hidden;}
#     footer {visibility: hidden;}
#     header {visibility: hidden;}
# </style>
# """, unsafe_allow_html=True)

# def format_percentage(value):
#     """Format percentage to 2 decimal places"""
#     return f"{value:.2f}%"

# @st.cache_resource
# def load_scorer():
#     try:
#         return ATSScorer()
#     except Exception as e:
#         st.error(f"Error loading the scorer: {str(e)}")
#         st.stop()

# def main():
#     # Header
#     st.title("📝 ATS Resume Scorer")
#     st.markdown("Evaluate how well your resume matches a job description using AI-powered analysis.")
    
#     # Create two columns for input
#     col1, col2 = st.columns(2)
    
#     # Job Description Input
#     with col1:
#         st.subheader("Job Description")
#         job_description = st.text_area(
#             label="Job Description Text",
#             key="job_desc",
#             height=300,
#             placeholder="Paste the complete job description here...",
#             label_visibility="collapsed"
#         )
    
#     # Resume Input
#     with col2:
#         st.subheader("Your Resume")
#         resume_text = st.text_area(
#             label="Resume Text",
#             key="resume",
#             height=300,
#             placeholder="Paste your resume text here...",
#             label_visibility="collapsed"
#         )
    
#     # Center the analyze button
#     col1, col2, col3 = st.columns([1, 2, 1])
#     with col2:
#         analyze_button = st.button("Analyze Resume", type="primary", use_container_width=True)
    
#     if analyze_button:
#         if not job_description or not resume_text:
#             st.error("Please provide both the job description and resume text.")
#             return
        
#         with st.spinner("Analyzing your resume..."):
#             try:
#                 scorer = load_scorer()
#                 results = scorer.score_resume(resume_text, job_description)
                
#                 # Results section
#                 st.markdown("### Analysis Results")
                
#                 # Score metrics in columns
#                 col1, col2, col3 = st.columns(3)
                
#                 with col1:
#                     st.markdown('<div class="metric-container">', unsafe_allow_html=True)
#                     st.metric(
#                         "Overall Match Score",
#                         format_percentage(results['overall_score'])
#                     )
#                     st.markdown('</div>', unsafe_allow_html=True)
                
#                 with col2:
#                     st.markdown('<div class="metric-container">', unsafe_allow_html=True)
#                     st.metric(
#                         "Technical Match",
#                         format_percentage(results['score_breakdown']['technical_match'])
#                     )
#                     st.markdown('</div>', unsafe_allow_html=True)
                
#                 with col3:
#                     st.markdown('<div class="metric-container">', unsafe_allow_html=True)
#                     st.metric(
#                         "Content Similarity",
#                         format_percentage(results['score_breakdown']['content_similarity'])
#                     )
#                     st.markdown('</div>', unsafe_allow_html=True)
                
#                 # Context understanding score
#                 st.markdown('<div class="metric-container">', unsafe_allow_html=True)
#                 st.metric(
#                     "Context Understanding",
#                     format_percentage(results['score_breakdown']['context_understanding'])
#                 )
#                 st.markdown('</div>', unsafe_allow_html=True)
                
#                 # Missing skills section
#                 if results['missing_skills']:
#                     st.markdown("### Missing Technical Skills")
#                     st.markdown('<div class="results-container">', unsafe_allow_html=True)
#                     for category, skills in results['missing_skills'].items():
#                         if skills:
#                             st.markdown(f"**{category.title()}:** {', '.join(skills)}")
#                     st.markdown('</div>', unsafe_allow_html=True)
                
#                 # Feedback section
#                 st.markdown("### Detailed Feedback")
#                 st.markdown('<div class="results-container">', unsafe_allow_html=True)
#                 for feedback in results['feedback']:
#                     st.markdown(feedback)
#                 st.markdown('</div>', unsafe_allow_html=True)
                
#             except Exception as e:
#                 st.error(f"An error occurred during analysis: {str(e)}")
#                 st.error("Please make sure you have installed all required dependencies.")
    
#     # Tips section
#     st.markdown("---")
#     st.markdown('<div class="results-container">', unsafe_allow_html=True)
#     st.markdown("""
#     💡 **Tips for Best Results:**
#     - Paste the complete job description and resume text
#     - Make sure to include relevant technical skills and keywords
#     - The analysis considers both technical skill matching and content similarity
#     """)
#     st.markdown('</div>', unsafe_allow_html=True)

# if __name__ == "__main__":
#     main()
