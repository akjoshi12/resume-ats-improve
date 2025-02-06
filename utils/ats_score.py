# ats_score.py
import re
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------
# CONFIGURATION & GLOBAL VARIABLES
# ------------------------------

# Configure the embedding model (you may change the model as needed)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

# Mistral API configuration for RAG (update with your actual endpoint and API key)
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
MISTRAL_API_KEY = "YOUR_MISTRAL_API_KEY"  # Ensure you set your API key here

# ------------------------------
# EMBEDDING & SIMILARITY FUNCTIONS
# ------------------------------

def generate_embedding(text: str) -> np.ndarray:
    """
    Generate an embedding for a given preprocessed text using the SentenceTransformer.
    """
    embedding = embedder.encode(text)
    return embedding

def compute_cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Compute the cosine similarity between two embeddings.
    """
    similarity = cosine_similarity([embedding1], [embedding2])[0][0]
    return similarity

def compute_ats_score(similarity: float) -> float:
    """
    Convert cosine similarity into a normalized ATS score on a 0-100 scale.
    """
    ats_score = similarity * 100  # Scale similarity to a percentage
    ats_score = min(100, ats_score)  # Cap the score at 100
    return round(ats_score, 2)

# ------------------------------
# KEYWORD EXTRACTION & IMPROVEMENT FUNCTIONS
# ------------------------------

from sklearn.feature_extraction.text import TfidfVectorizer

def extract_keywords(text: str, top_n: int = 10) -> list:
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.toarray().flatten()
    sorted_indices = scores.argsort()[::-1]
    return [feature_names[i] for i in sorted_indices[:top_n]]


# def extract_keywords(text: str, top_n: int = 10) -> list:
#     """
#     Extract keywords from the text using a simple approach: select the longest words.
#     (For production use, consider a more advanced keyword extraction technique.)
#     """
#     words = re.findall(r'\w+', text.lower())
#     # Remove duplicates and filter out some basic stopwords
#     stopwords = {"the", "and", "for", "with", "that", "this", "a", "an", "of", "to", "in"}
#     words = [word for word in set(words) if word not in stopwords and len(word) > 3]
#     # Sort words by length as a proxy for importance
#     words.sort(key=lambda w: len(w), reverse=True)
#     return words[:top_n]

def identify_improvement_areas(resume_text: str, job_text: str) -> list:
    """
    Identify improvement areas by comparing keywords from the job description against those in the resume.
    Returns a list of keywords that are present in the job description but missing in the resume.
    """
    resume_keywords = set(extract_keywords(resume_text))
    job_keywords = set(extract_keywords(job_text))
    missing_keywords = list(job_keywords - resume_keywords)
    return missing_keywords

# ------------------------------
# RELEVANCE FUNCTIONS
# ------------------------------

def identify_irrelevant_sections(resume_text: str, job_text: str, threshold: float = 0.5) -> list:
    # Split on common section headings
    sections = re.split(r'\n\s*(?:Experience|Education|Skills|Projects|Summary)\s*\n', resume_text)
    sections = [s.strip() for s in sections if s.strip()]   
    ''' Identify sections of the resume that might be considered irrelevant to the job description.
    
    The resume is split into sections based on newline characters. For each section, the function computes
    the cosine similarity between the section's embedding and the job description's embedding. Sections with
    similarity below the threshold are flagged as irrelevant.
    
    **Relevance Definition:**
      - Relevant Content: Experiences, skills, and accomplishments that align with the job description.
      - Irrelevant Content: Unrelated hobbies, personal interests, or outdated experiences that do not support
        the candidate’s suitability for the role and may hinder selection.
    '''
    # Split the resume by newlines; you may choose a more advanced method (e.g., section headings) if needed.
    # sections = [section.strip() for section in resume_text.split('\n') if section.strip()]
    job_embedding = generate_embedding(job_text)
    irrelevant_sections = []
    for section in sections:
        section_embedding = generate_embedding(section)
        similarity = compute_cosine_similarity(section_embedding, job_embedding)
        if similarity < threshold:
            irrelevant_sections.append(section)
    return irrelevant_sections

# ------------------------------
# MISTRAL API (RAG) FUNCTIONS
# ------------------------------

def call_mistral_api(prompt: str, max_tokens: int = 200) -> str:
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "mistral-tiny",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }
    try:
        response = requests.post(MISTRAL_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        if "choices" in data and len(data["choices"]) > 0:
            generated_text = data["choices"][0]["message"]["content"]
            return generated_text.strip()
        return "Error: No response generated."
    except Exception as e:
        return f"API Error: {str(e)}"
    

# def call_mistral_api(prompt: str, max_tokens: int = 200) -> str:
#     """
#     Call the Mistral API with a given prompt to generate text.
#     Adjust the payload or headers as necessary to match your API’s requirements.
#     """
#     headers = {
#         "Authorization": f"Bearer {MISTRAL_API_KEY}",
#         "Content-Type": "application/json",
#     }
#     payload = {
#         "prompt": prompt,
#         "max_tokens": max_tokens,
#         # Additional parameters may be added here.
#     }
#     try:
#         response = requests.post(MISTRAL_API_URL, headers=headers, json=payload)
#         response.raise_for_status()
#         data = response.json()
#         # Extract the generated text from the API response. Adjust the key as needed.
#         generated_text = data.get("generated_text", "")
#         return generated_text.strip()
#     except Exception as e:
#         return f"Error calling Mistral API: {e}"

def generate_resume_suggestion(resume_text: str, job_text: str, improvement_areas: list) -> str:
    """
    Use the Mistral API to generate a suggested resume that incorporates the missing keywords
    and is optimized to achieve a high ATS score.
    """
    prompt = (
        f"Given the following resume:\n\n{resume_text}\n\n"
        f"And the job description:\n\n{job_text}\n\n"
        f"The resume is missing these important keywords: {', '.join(improvement_areas)}.\n"
        "Please generate an improved resume that is optimized for an ATS score above 90, "
        "ensuring these areas are incorporated effectively. Also, remove or modify sections that "
        "do not relate to the job requirements."
    )
    suggestion = call_mistral_api(prompt, max_tokens=400)
    return suggestion

# ------------------------------
# EXAMPLE USAGE (Run as a script for testing)
# ------------------------------

if __name__ == "__main__":
    # Replace these sample texts with your actual preprocessed resume and job description texts.
    sample_resume = (
        "Experienced data scientist with a strong background in machine learning, statistics, "
        "and programming. Expert in Python, data analysis, and predictive modeling. Proven track "
        "record of implementing data-driven solutions.\n\n"
        "Hobbies include playing guitar and hiking. Recently, participated in a local cooking class."
    )
    sample_job = (
        "We are seeking a Data Scientist proficient in Python, machine learning, and deep learning. "
        "Candidates must have experience with data visualization, big data processing, and model deployment."
    )

    # Generate embeddings for both texts.
    resume_embedding = generate_embedding(sample_resume)
    job_embedding = generate_embedding(sample_job)

    # Compute cosine similarity and derive the ATS score.
    similarity = compute_cosine_similarity(resume_embedding, job_embedding)
    ats_score = compute_ats_score(similarity)

    # Identify improvement areas based on missing keywords.
    improvement_areas = identify_improvement_areas(sample_resume, sample_job)

    # Identify irrelevant sections that might hinder selection.
    irrelevant_sections = identify_irrelevant_sections(sample_resume, sample_job, threshold=0.5)

    # Output the results.
    print(f"ATS Score: {ats_score}/100")
    print("Improvement Areas:", improvement_areas)
    print("\nIrrelevant Sections (may hinder selection, consider removing or revising):")
    for section in irrelevant_sections:
        print(f"- {section}")

    # (Optional) Generate a suggested resume using RAG via the Mistral API.
    suggested_resume = generate_resume_suggestion(sample_resume, sample_job, improvement_areas)
    print("\nSuggested Resume:")
    print(suggested_resume)


# # ats_score.py
# import re
# import numpy as np
# import requests
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity

# # ------------------------------
# # CONFIGURATION & GLOBAL VARIABLES
# # ------------------------------

# # Configure the embedding model (you may change the model as needed)
# EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
# embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

# # Mistral API configuration for RAG (update with your actual endpoint and API key)
# MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"

# # ------------------------------
# # EMBEDDING & SIMILARITY FUNCTIONS
# # ------------------------------

# def generate_embedding(text: str) -> np.ndarray:
#     """
#     Generate an embedding for a given preprocessed text using the SentenceTransformer.
#     """
#     embedding = embedder.encode(text)
#     return embedding

# def compute_cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
#     """
#     Compute the cosine similarity between two embeddings.
#     """
#     similarity = cosine_similarity([embedding1], [embedding2])[0][0]
#     return similarity

# def compute_ats_score(similarity: float) -> float:
#     """
#     Convert cosine similarity into a normalized ATS score on a 0-100 scale.
#     """
#     ats_score = similarity * 100  # Scale similarity to a percentage
#     ats_score = min(100, ats_score)  # Cap the score at 100
#     return round(ats_score, 2)

# # ------------------------------
# # KEYWORD EXTRACTION & IMPROVEMENT FUNCTIONS
# # ------------------------------

# def extract_keywords(text: str, top_n: int = 10) -> list:
#     """
#     Extract keywords from the text using a simple approach: select the longest words.
#     (For production use, consider a more advanced keyword extraction technique.)
#     """
#     words = re.findall(r'\w+', text.lower())
#     # Remove duplicates and filter out some basic stopwords
#     stopwords = {"the", "and", "for", "with", "that", "this", "a", "an", "of", "to", "in"}
#     words = [word for word in set(words) if word not in stopwords and len(word) > 3]
#     # Sort words by length as a proxy for importance
#     words.sort(key=lambda w: len(w), reverse=True)
#     return words[:top_n]

# def identify_improvement_areas(resume_text: str, job_text: str) -> list:
#     """
#     Identify improvement areas by comparing keywords from the job description against those in the resume.
#     Returns a list of keywords that are present in the job description but missing in the resume.
#     """
#     resume_keywords = set(extract_keywords(resume_text))
#     job_keywords = set(extract_keywords(job_text))
#     missing_keywords = list(job_keywords - resume_keywords)
#     return missing_keywords

# # ------------------------------
# # MISTRAL API (RAG) FUNCTIONS
# # ------------------------------

# def call_mistral_api(prompt: str, max_tokens: int = 200) -> str:
#     """
#     Call the Mistral API with a given prompt to generate text.
#     Adjust the payload or headers as necessary to match your API’s requirements.
#     """
#     headers = {
#         "Authorization": f"Bearer {MISTRAL_API_KEY}",
#         "Content-Type": "application/json",
#     }
#     payload = {
#         "prompt": prompt,
#         "max_tokens": max_tokens,
#         # Additional parameters may be added here.
#     }
#     try:
#         response = requests.post(MISTRAL_API_URL, headers=headers, json=payload)
#         response.raise_for_status()
#         data = response.json()
#         # Extract the generated text from the API response. Adjust the key as needed.
#         generated_text = data.get("generated_text", "")
#         return generated_text.strip()
#     except Exception as e:
#         return f"Error calling Mistral API: {e}"

# def generate_resume_suggestion(resume_text: str, job_text: str, improvement_areas: list) -> str:
#     """
#     Use the Mistral API to generate a suggested resume that incorporates the missing keywords
#     and is optimized to achieve a high ATS score.
#     """
#     prompt = (
#         f"Given the following resume:\n\n{resume_text}\n\n"
#         f"And the job description:\n\n{job_text}\n\n"
#         f"The resume is missing these important keywords: {', '.join(improvement_areas)}.\n"
#         "Please generate an improved resume that is optimized for an ATS score above 90, "
#         "ensuring these areas are incorporated effectively."
#     )
#     suggestion = call_mistral_api(prompt, max_tokens=400)
#     return suggestion

# # ------------------------------
# # EXAMPLE USAGE (Run as a script for testing)
# # ------------------------------

# if __name__ == "__main__":
#     # Replace these sample texts with your actual preprocessed resume and job description texts.
#     sample_resume = (
#         "Experienced data scientist with a strong background in machine learning, statistics, "
#         "and programming. Expert in Python, data analysis, and predictive modeling. Proven track "
#         "record of implementing data-driven solutions."
#     )
#     sample_job = (
#         "We are seeking a Data Scientist proficient in Python, machine learning, and deep learning. "
#         "Candidates must have experience with data visualization, big data processing, and model deployment."
#     )

#     # Generate embeddings for both texts.
#     resume_embedding = generate_embedding(sample_resume)
#     job_embedding = generate_embedding(sample_job)

#     # Compute cosine similarity and derive the ATS score.
#     similarity = compute_cosine_similarity(resume_embedding, job_embedding)
#     ats_score = compute_ats_score(similarity)

#     # Identify improvement areas based on missing keywords.
#     improvement_areas = identify_improvement_areas(sample_resume, sample_job)

#     # Output the results.
#     print(f"ATS Score: {ats_score}/100")
#     print("Improvement Areas:", improvement_areas)

#     # (Optional) Generate a suggested resume using RAG via the Mistral API.
#     suggested_resume = generate_resume_suggestion(sample_resume, sample_job, improvement_areas)
#     print("\nSuggested Resume:")
#     print(suggested_resume)
