# better_resume.py
from utils.ats_score import (
    generate_embedding,
    compute_cosine_similarity,
    compute_ats_score,
    identify_improvement_areas,
    generate_resume_suggestion,
)
from utils.preprocessing import preprocess_text

def calculate_score(resume_text: str, job_text: str) -> int:
    """
    Generate embeddings, compute cosine similarity, and derive an ATS score.
    """
    resume_embedding = generate_embedding(resume_text)
    job_embedding = generate_embedding(job_text)
    similarity = compute_cosine_similarity(resume_embedding, job_embedding)
    return compute_ats_score(similarity)

def improve_resume(resume_text: str, job_text: str, target_score: int = 90, max_iter: int = 5):
    cleaned_resume = preprocess_text(resume_text)
    cleaned_job = preprocess_text(job_text)
    
    best_resume = cleaned_resume
    best_score = calculate_score(cleaned_resume, cleaned_job)
    
    iteration = 0
    while best_score < target_score and iteration < max_iter:
        iteration += 1
        
        # Recalculate weaknesses each iteration
        weaknesses = identify_improvement_areas(best_resume, cleaned_job)
        suggestion = generate_resume_suggestion(best_resume, cleaned_job, weaknesses)
        
        suggestion_score = calculate_score(suggestion, cleaned_job)
        
        if suggestion_score > best_score:
            best_score = suggestion_score
            best_resume = suggestion

    return best_resume, best_score, iteration

# def improve_resume(resume_text: str, job_text: str, target_score: int = 90, max_iter: int = 5):
#     """
#     Iteratively improve the resume until the ATS score reaches the target score
#     or until max_iter iterations have been performed.
    
#     Returns a tuple of (best_resume, best_score, iterations_used)
#     """
#     cleaned_resume = preprocess_text(resume_text)
#     cleaned_job = preprocess_text(job_text)
    
#     weaknesses = identify_improvement_areas(cleaned_resume, cleaned_job)
    
#     current_resume = cleaned_resume
#     best_resume = cleaned_resume
#     best_score = calculate_score(current_resume, cleaned_job)
    
#     iteration = 0
#     while best_score < target_score and iteration < max_iter:
#         iteration += 1
        
#         # Generate a new resume suggestion based on the current weaknesses
#         suggestion = generate_resume_suggestion(current_resume, cleaned_job, weaknesses)
        
#         # Calculate the new ATS score
#         suggestion_score = calculate_score(suggestion, cleaned_job)
        
#         # If the suggestion is better, update our best result
#         if suggestion_score > best_score:
#             best_score = suggestion_score
#             best_resume = suggestion
        
#         current_resume = suggestion
        
#         # Early exit if the target score is achieved
#         if best_score >= target_score:
#             break
    
#     return best_resume, best_score, iteration
