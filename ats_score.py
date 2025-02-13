import spacy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import re
from collections import defaultdict

class ATSScorer:
    def __init__(self):
        """Initialize with tech-focused NLP components"""
        try:
            # Using smaller spaCy model instead of en_core_web_lg
            self.nlp = spacy.load('en_core_web_sm')
        except:
            self.nlp = spacy.blank("en")
        
        # Using MPNet for better technical context understanding
        self.bert_model = SentenceTransformer('all-mpnet-base-v2')
        
        # Enhanced TF-IDF with technical focus
        self.tfidf_vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),  # Capture compound technical terms
            stop_words='english',
            max_features=5000
        )
        
        # Technical skill categories for better scoring
        self.tech_categories = {
            'programming_languages': set(['python', 'java', 'javascript', 'c++', 'golang', 'rust']),
            'frameworks': set(['react', 'angular', 'vue', 'django', 'flask', 'spring']),
            'cloud': set(['aws', 'azure', 'gcp', 'kubernetes', 'docker']),
            'databases': set(['sql', 'mongodb', 'postgresql', 'mysql', 'redis']),
            'tools': set(['git', 'jenkins', 'jira', 'terraform', 'prometheus'])
        }
    
    def enhanced_preprocess(self, text):
        """Tech-focused text preprocessing"""
        # Convert to lowercase
        text = text.lower()
        
        # Preserve common technical patterns
        text = re.sub(r'([./\\])', r' \1 ', text)  # Separate version numbers and paths
        text = re.sub(r'([A-Za-z])(\d)', r'\1 \2', text)  # Separate letters and numbers
        
        # Process with spaCy
        doc = self.nlp(text)
        tokens = []
        
        for token in doc:
            # Keep technical terms intact
            if any(token.text in terms for terms in self.tech_categories.values()):
                tokens.append(token.text)
            # Process other terms
            elif not token.is_stop and not token.is_punct and not token.is_space:
                tokens.append(token.lemma_)
                
        return ' '.join(tokens)
    
    def extract_technical_skills(self, text):
        """Extract and categorize technical skills"""
        skills = defaultdict(set)
        text_lower = text.lower()
        
        # Extract skills by category
        for category, terms in self.tech_categories.items():
            for term in terms:
                if term in text_lower:
                    skills[category].add(term)
        
        # Extract version numbers and specific technologies
        version_pattern = r'\b(?:[a-zA-Z.]+(?:js)?(?:\d+(?:\.\d+)*)?)\b'
        versions = re.findall(version_pattern, text)
        skills['versions'] = set(versions)
        
        return skills
    
    def calculate_hybrid_similarity(self, resume_text, job_description):
        """Calculate similarities with tech focus"""
        # TF-IDF similarity
        processed = [
            self.enhanced_preprocess(job_description),
            self.enhanced_preprocess(resume_text)
        ]
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(processed)
        tfidf_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        # BERT semantic similarity
        embeddings = self.bert_model.encode([job_description, resume_text])
        bert_sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        
        # Technical skill match
        job_skills = self.extract_technical_skills(job_description)
        resume_skills = self.extract_technical_skills(resume_text)
        
        skill_match_scores = []
        for category in job_skills:
            if job_skills[category]:
                match_ratio = len(resume_skills[category] & job_skills[category]) / len(job_skills[category])
                skill_match_scores.append(match_ratio)
        
        tech_sim = np.mean(skill_match_scores) if skill_match_scores else 0
        
        return tfidf_sim, bert_sim, tech_sim
    
    def score_resume(self, resume_text, job_description):
        """Advanced scoring system for tech positions"""
        # Calculate similarities
        tfidf_sim, bert_sim, tech_sim = self.calculate_hybrid_similarity(
            resume_text, job_description
        )
        
        # Extract technical requirements
        job_skills = self.extract_technical_skills(job_description)
        resume_skills = self.extract_technical_skills(resume_text)
        
        # Calculate missing skills
        missing_skills = {
            category: list(job_skills[category] - resume_skills[category])
            for category in job_skills
            if job_skills[category]
        }
        
        # Composite scoring with tech focus
        scores = {
            'technical_match': tech_sim,
            'tfidf_similarity': tfidf_sim,
            'bert_similarity': bert_sim,
        }
        
        # Weighted final score (tech-focused weights)
        final_score = (
            scores['technical_match'] * 0.4 +  # Higher weight for technical skills
            scores['tfidf_similarity'] * 0.3 +
            scores['bert_similarity'] * 0.3
        ) * 100
        
        return {
            'overall_score': round(final_score, 2),
            'score_breakdown': {
                'technical_match': round(tech_sim * 100, 2),
                'content_similarity': round(tfidf_sim * 100, 2),
                'context_understanding': round(bert_sim * 100, 2)
            },
            'missing_skills': missing_skills,
            'feedback': self.generate_tech_feedback(scores, missing_skills)
        }
    
    def generate_tech_feedback(self, scores, missing_skills):
        """Generate tech-focused feedback"""
        feedback = []
        
        if scores['technical_match'] < 0.7:
            feedback.append("🔧 Technical Skill Gaps Detected:")
            for category, skills in missing_skills.items():
                if skills:
                    feedback.append(f"- {category.title()}: Missing {', '.join(skills)}")
        
        if scores['bert_similarity'] < 0.75:
            feedback.append("💡 Suggestions for improvement:")
            feedback.append("- Elaborate on technical projects and implementations")
            feedback.append("- Include specific tools and technologies used")
            feedback.append("- Quantify impact and scale of your work")
        
        if scores['tfidf_similarity'] < 0.65:
            feedback.append("📝 Additional recommendations:")
            feedback.append("- Match technical terminology with job description")
            feedback.append("- Include relevant frameworks and versions")
            feedback.append("- Highlight system design and architecture experience")
        
        if not feedback:
            feedback.append("✅ Strong technical alignment with the position!")
            feedback.append("💪 Your skills match well with the job requirements")
        
        return feedback

