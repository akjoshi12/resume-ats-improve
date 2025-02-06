import streamlit as st

from utils.preprocessing import preprocess_text
from utils.ats_score import (
    generate_embedding,
    compute_cosine_similarity,
    compute_ats_score,
    identify_improvement_areas,
    extract_keywords,
    identify_irrelevant_sections,
    generate_resume_suggestion,
)
from utils.better_resume import improve_resume  # Importing the better_resume module

st.set_page_config(page_title="ATS Resume Analyzer", page_icon="📝")

def main():
    st.title("ATS Resume Analyzer")
    st.write(
        "Paste your resume and job description below. The app will preprocess your text, "
        "compute an ATS score, highlight strengths, improvement areas, and identify sections that "
        "might be irrelevant to the job description."
    )

    resume_input = st.text_area("Enter Resume Text", height=150, placeholder="Paste your resume text here...")
    job_desc_input = st.text_area("Enter Job Description", height=150, placeholder="Paste the job description text here...")

    if st.button("Analyze ATS Score"):
        if not resume_input or not job_desc_input:
            st.warning("Please provide both a resume and a job description.")
        else:
            cleaned_resume = preprocess_text(resume_input)
            cleaned_job = preprocess_text(job_desc_input)

            with st.expander("Show Cleaned Texts"):
                st.subheader("Cleaned Resume Text")
                st.text_area("Cleaned Resume", cleaned_resume, height=150)
                st.subheader("Cleaned Job Description Text")
                st.text_area("Cleaned Job Description", cleaned_job, height=150)

            resume_embedding = generate_embedding(cleaned_resume)
            job_embedding = generate_embedding(cleaned_job)

            similarity = compute_cosine_similarity(resume_embedding, job_embedding)
            ats_score = compute_ats_score(similarity)

            st.subheader("ATS Score")
            st.progress(int(ats_score))
            st.metric(label="ATS Score", value=f"{ats_score}/100")

            weaknesses = identify_improvement_areas(cleaned_resume, cleaned_job)
            resume_keywords = set(extract_keywords(cleaned_resume))
            job_keywords = set(extract_keywords(cleaned_job))
            strengths = list(resume_keywords.intersection(job_keywords))
            irrelevant_sections = identify_irrelevant_sections(cleaned_resume, cleaned_job, threshold=0.5)

            st.subheader("Analysis")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Strengths")
                if strengths:
                    st.write(", ".join(strengths))
                else:
                    st.write("No significant strengths identified. Consider adding keywords from the job description that match your skills.")
            with col2:
                st.markdown("### Improvement Areas (Weaknesses)")
                if weaknesses:
                    st.write(", ".join(weaknesses))
                else:
                    st.write("Your resume seems well-aligned with the job description!")

            if irrelevant_sections:
                st.subheader("Potentially Irrelevant Sections")
                st.write("The following sections may not be relevant to the job description and might hinder your chances. Consider revising or removing them:")
                for idx, section in enumerate(irrelevant_sections, start=1):
                    st.markdown(f"**Section {idx}:** {section}")
            else:
                st.info("All sections of your resume appear relevant to the job description.")

            st.success("Analysis complete!")

            if st.button("Generate Improved Resume Suggestion"):
                with st.spinner("Generating suggestion..."):
            # Correct parameter passing and unpack tuple
                    suggested_resume, score, iterations = improve_resume(cleaned_resume, cleaned_job)
                st.subheader("Suggested Improved Resume")
                st.metric(label="Achieved Score", value=f"{score}/100")
                st.text_area("Improved Resume", suggested_resume, height=300)

if __name__ == "__main__":
    main()



# # streamlit_app.py
# import streamlit as st

# from utils.preprocessing import preprocess_text
# from utils.ats_score import (
#     generate_embedding,
#     compute_cosine_similarity,
#     compute_ats_score,
#     identify_improvement_areas,
#     extract_keywords,
#     identify_irrelevant_sections,
#     generate_resume_suggestion,
# )

# st.set_page_config(page_title="ATS Resume Analyzer", page_icon="📄")

# def main():
#     st.title("ATS Resume Analyzer")
#     st.write(
#         "Paste your resume and job description below. The app will preprocess your text, "
#         "compute an ATS score, highlight strengths, improvement areas, and identify sections that "
#         "might be irrelevant to the job description."
#     )

#     # Two scrollable text areas for resume and job description input
#     resume_input = st.text_area("Enter Resume Text", height=150, placeholder="Paste your resume text here...")
#     job_desc_input = st.text_area("Enter Job Description", height=150, placeholder="Paste the job description text here...")

#     if st.button("Analyze ATS Score"):
#         if not resume_input or not job_desc_input:
#             st.warning("Please provide both a resume and a job description.")
#         else:
#             # Preprocess both inputs
#             cleaned_resume = preprocess_text(resume_input)
#             cleaned_job = preprocess_text(job_desc_input)

#             # Display the cleaned texts (optional)
#             with st.expander("Show Cleaned Texts"):
#                 st.subheader("Cleaned Resume Text")
#                 st.text_area("Cleaned Resume", cleaned_resume, height=150)
#                 st.subheader("Cleaned Job Description Text")
#                 st.text_area("Cleaned Job Description", cleaned_job, height=150)

#             # Compute embeddings for both texts
#             resume_embedding = generate_embedding(cleaned_resume)
#             job_embedding = generate_embedding(cleaned_job)

#             # Compute cosine similarity and derive the ATS score
#             similarity = compute_cosine_similarity(resume_embedding, job_embedding)
#             ats_score = compute_ats_score(similarity)

#             # Display ATS score with a progress bar and metric
#             st.subheader("ATS Score")
#             st.progress(int(ats_score))
#             st.markdown(
#                 f"""
#                 <div style="display: flex; align-items: center; justify-content: center;">
#                     <div style="border-radius: 50%; width: 100px; height: 100px; display: flex; align-items: center; justify-content: center; background-color: #f0f0f0; border: 2px solid #ccc;">
#                         <div style="font-size: 24px; font-weight: bold;">{int(ats_score)}</div>
#                     </div>
#                 </div>
#                 """,
#                 unsafe_allow_html=True
#             )
#             st.metric(label="ATS Score", value=f"{ats_score}/100")

#             # Identify improvement areas (keywords present in job description but missing in resume)
#             weaknesses = identify_improvement_areas(cleaned_resume, cleaned_job)

#             # Calculate strengths as common keywords between resume and job description.
#             resume_keywords = set(extract_keywords(cleaned_resume))
#             job_keywords = set(extract_keywords(cleaned_job))
#             strengths = list(resume_keywords.intersection(job_keywords))

#             # Identify irrelevant sections in the resume (those with low relevance)
#             irrelevant_sections = identify_irrelevant_sections(cleaned_resume, cleaned_job, threshold=0.5)

#             # Display strengths and weaknesses in columns
#             st.subheader("Analysis")
#             col1, col2 = st.columns(2)
#             with col1:
#                 st.markdown("### Strengths")
#                 if strengths:
#                     st.write(", ".join(strengths))
#                 else:
#                     st.write("No significant strengths identified. Consider adding keywords from the job description that match your skills.")
#             with col2:
#                 st.markdown("### Improvement Areas (Weaknesses)")
#                 if weaknesses:
#                     st.write(", ".join(weaknesses))
#                 else:
#                     st.write("Your resume seems well-aligned with the job description!")

#             # Display irrelevant sections, if any.
#             if irrelevant_sections:
#                 st.subheader("Potentially Irrelevant Sections")
#                 st.write("The following sections may not be relevant to the job description and might hinder your chances. Consider revising or removing them:")
#                 for idx, section in enumerate(irrelevant_sections, start=1):
#                     st.markdown(f"**Section {idx}:** {section}")
#             else:
#                 st.info("All sections of your resume appear relevant to the job description.")

#             st.success("Analysis complete!")

#             # Option to generate a suggested resume
#             if st.button("Generate Improved Resume Suggestion"):
#                 with st.spinner("Generating suggestion..."):
#                     suggested_resume = generate_resume_suggestion(cleaned_resume, cleaned_job, weaknesses)
#                 st.subheader("Suggested Improved Resume")
#                 st.text_area("Improved Resume", suggested_resume, height=300)

# if __name__ == "__main__":
#     main()


# # streamlit_app.py
# import streamlit as st

# from utils.preprocessing import preprocess_text
# from utils.ats_score import (
#     generate_embedding,
#     compute_cosine_similarity,
#     compute_ats_score,
#     identify_improvement_areas,
#     extract_keywords,
#     identify_irrelevant_sections,
#     generate_resume_suggestion,
# )

# st.set_page_config(page_title="ATS Resume Analyzer", page_icon="📄")

# def main():
#     st.title("ATS Resume Analyzer")
#     st.write(
#         "Paste your resume and job description below. The app will preprocess your text, "
#         "compute an ATS score, highlight strengths, improvement areas, and identify sections that "
#         "might be irrelevant to the job description."
#     )

#     # Two scrollable text areas for resume and job description input
#     resume_input = st.text_area("Enter Resume Text", height=150, placeholder="Paste your resume text here...")
#     job_desc_input = st.text_area("Enter Job Description", height=150, placeholder="Paste the job description text here...")

#     if st.button("Analyze ATS Score"):
#         if not resume_input or not job_desc_input:
#             st.warning("Please provide both a resume and a job description.")
#         else:
#             # Preprocess both inputs
#             cleaned_resume = preprocess_text(resume_input)
#             cleaned_job = preprocess_text(job_desc_input)

#             # Display the cleaned texts (optional)
#             with st.expander("Show Cleaned Texts"):
#                 st.subheader("Cleaned Resume Text")
#                 st.text_area("Cleaned Resume", cleaned_resume, height=150)
#                 st.subheader("Cleaned Job Description Text")
#                 st.text_area("Cleaned Job Description", cleaned_job, height=150)

#             # Compute embeddings for both texts
#             resume_embedding = generate_embedding(cleaned_resume)
#             job_embedding = generate_embedding(cleaned_job)

#             # Compute cosine similarity and derive the ATS score
#             similarity = compute_cosine_similarity(resume_embedding, job_embedding)
#             ats_score = compute_ats_score(similarity)

#             # Display ATS score with a progress bar and metric
#             st.subheader("ATS Score")
#             st.progress(int(ats_score))
#             st.metric(label="ATS Score", value=f"{ats_score}/100")

#             # Identify improvement areas (keywords present in job description but missing in resume)
#             weaknesses = identify_improvement_areas(cleaned_resume, cleaned_job)
            
#             # Calculate strengths as common keywords between resume and job description.
#             resume_keywords = set(extract_keywords(cleaned_resume))
#             job_keywords = set(extract_keywords(cleaned_job))
#             strengths = list(resume_keywords.intersection(job_keywords))
            
#             # Identify irrelevant sections in the resume (those with low relevance)
#             irrelevant_sections = identify_irrelevant_sections(cleaned_resume, cleaned_job, threshold=0.5)

#             # Display strengths and weaknesses in columns
#             st.subheader("Analysis")
#             col1, col2 = st.columns(2)
#             with col1:
#                 st.markdown("### Strengths")
#                 if strengths:
#                     st.write(", ".join(strengths))
#                 else:
#                     st.write("No significant strengths identified. Consider adding keywords from the job description that match your skills.")
#             with col2:
#                 st.markdown("### Improvement Areas (Weaknesses)")
#                 if weaknesses:
#                     st.write(", ".join(weaknesses))
#                 else:
#                     st.write("Your resume seems well-aligned with the job description!")
            
#             # Display irrelevant sections, if any.
#             if irrelevant_sections:
#                 st.subheader("Potentially Irrelevant Sections")
#                 st.write("The following sections may not be relevant to the job description and might hinder your chances. Consider revising or removing them:")
#                 for idx, section in enumerate(irrelevant_sections, start=1):
#                     st.markdown(f"**Section {idx}:** {section}")
#             else:
#                 st.info("All sections of your resume appear relevant to the job description.")

#             st.success("Analysis complete!")

#             # Option to generate a suggested resume
#             if st.button("Generate Improved Resume Suggestion"):
#                 with st.spinner("Generating suggestion..."):
#                     suggested_resume = generate_resume_suggestion(cleaned_resume, cleaned_job, weaknesses)
#                 st.subheader("Suggested Improved Resume")
#                 st.text_area("Improved Resume", suggested_resume, height=300)

# if __name__ == "__main__":
#     main()


# # streamlit_app.py
# import streamlit as st

# from utils.preprocessing import preprocess_text
# from utils.ats_score import (
#     generate_embedding,
#     compute_cosine_similarity,
#     compute_ats_score,
#     identify_improvement_areas,
#     extract_keywords,
# )

# st.set_page_config(page_title="ATS Resume Analyzer", page_icon="📄")

# def main():
#     st.title("ATS Resume Analyzer")
#     st.write("Paste your resume and job description below. The app will preprocess your text, compute an ATS score, and highlight your strengths and weaknesses based on keyword matching.")

#     # Two scrollable text areas for resume and job description input
#     resume_input = st.text_area("Enter Resume Text", height=100, placeholder="Paste your resume text here...")
#     job_desc_input = st.text_area("Enter Job Description", height=100, placeholder="Paste the job description text here...")

#     if st.button("Analyze ATS Score"):
#         if not resume_input or not job_desc_input:
#             st.warning("Please provide both a resume and a job description.")
#         else:
#             # Preprocess both inputs
#             cleaned_resume = preprocess_text(resume_input)
#             cleaned_job = preprocess_text(job_desc_input)

#             # Display the cleaned texts (optional)
#             with st.expander("Show Cleaned Texts"):
#                 st.subheader("Cleaned Resume Text")
#                 st.text_area("Cleaned Resume", cleaned_resume, height=150)
#                 st.subheader("Cleaned Job Description Text")
#                 st.text_area("Cleaned Job Description", cleaned_job, height=150)

#             # Compute embeddings for both texts
#             resume_embedding = generate_embedding(cleaned_resume)
#             job_embedding = generate_embedding(cleaned_job)

#             # Compute cosine similarity and derive the ATS score
#             similarity = compute_cosine_similarity(resume_embedding, job_embedding)
#             ats_score = compute_ats_score(similarity)

#             # Display ATS score with a progress bar and metric
#             st.subheader("ATS Score")
#             st.progress(int(ats_score))
#             st.metric(label="ATS Score", value=f"{ats_score}/100")

#             # Identify improvement areas (weaknesses)
#             weaknesses = identify_improvement_areas(cleaned_resume, cleaned_job)
            
#             # Calculate strengths as common keywords between resume and job description.
#             resume_keywords = set(extract_keywords(cleaned_resume))
#             job_keywords = set(extract_keywords(cleaned_job))
#             strengths = list(resume_keywords.intersection(job_keywords))
            
#             # Display strengths and weaknesses in columns
#             st.subheader("Analysis")
#             col1, col2 = st.columns(2)
#             with col1:
#                 st.markdown("### Strengths")
#                 if strengths:
#                     st.write(", ".join(strengths))
#                 else:
#                     st.write("No significant strengths identified. Consider adding keywords from the job description that match your skills.")
#             with col2:
#                 st.markdown("### Improvement Areas (Weaknesses)")
#                 if weaknesses:
#                     st.write(", ".join(weaknesses))
#                 else:
#                     st.write("Your resume seems well-aligned with the job description!")

#             st.success("Analysis complete!")

# if __name__ == "__main__":
#     main()


# # # streamlit_app.py
# # import streamlit as st
# # from utils.preprocessing import preprocess_text

# # st.set_page_config(page_title="Resume & Job Description Preprocessor", page_icon="📄")

# # def main():
# #     st.title("Resume & Job Description Preprocessor")
# #     st.write("Paste your resume and job description below to see the cleaned text output.")

# #     # Two scrollable text areas for resume and job description input
# #     resume_input = st.text_area("Enter Resume Text", height=100, placeholder="Paste your resume text here...")
# #     job_desc_input = st.text_area("Enter Job Description", height=100, placeholder="Paste the job description text here...")

# #     if st.button("Process Text"):
# #         if not resume_input and not job_desc_input:
# #             st.warning("Please provide at least one input.")
# #         else:
# #             # Preprocess both inputs
# #             cleaned_resume = preprocess_text(resume_input)
# #             cleaned_job = preprocess_text(job_desc_input)

# #             # Display the cleaned texts
# #             if resume_input:
# #                 st.subheader("Cleaned Resume Text")
# #                 st.text_area("Cleaned Resume", cleaned_resume, height=300)
# #             if job_desc_input:
# #                 st.subheader("Cleaned Job Description Text")
# #                 st.text_area("Cleaned Job Description", cleaned_job, height=300)

# #             st.success("Text preprocessing completed!")

# # if __name__ == "__main__":
# #     main()
