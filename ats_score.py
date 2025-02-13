import spacy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch
import re
from collections import defaultdict

class ATSScorer:
    def __init__(self):
        """Initialize with models including DistilBERT"""
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except:
            self.nlp = spacy.blank("en")
        
        # Initialize models as None first
        self.resume_tokenizer = None
        self.resume_model = None
        self.skill_model = None
        
        # Initialize models in a way that's compatible with Streamlit's caching
        self._initialize_models()
        
        # TF-IDF for keyword matching
        self.tfidf_vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            stop_words='english',
            max_features=3000,
            vocabulary=self.load_job_specific_vocabulary()
        )
        
        # Technical skill categories
        self.tech_categories = {
            'programming_languages': set(['python', 'java', 'javascript', 'c++', 'golang', 'rust']),
            'frameworks': set(['react', 'angular', 'vue', 'django', 'flask', 'spring']),
            'cloud': set(['aws', 'azure', 'gcp', 'kubernetes', 'docker']),
            'databases': set(['sql', 'mongodb', 'postgresql', 'mysql', 'redis']),
            'tools': set(['git', 'jenkins', 'jira', 'terraform', 'prometheus'])
        }

    def _initialize_models(self):
        """Initialize models separately to avoid PyTorch class registration issues"""
        if self.resume_tokenizer is None:
            self.resume_tokenizer = AutoTokenizer.from_pretrained(
                "MNG123/msmarco-distilbert-base-tas-b-resume-fit-v2-epoch-3",
                local_files_only=False
            )
        
        if self.resume_model is None:
            self.resume_model = AutoModel.from_pretrained(
                "MNG123/msmarco-distilbert-base-tas-b-resume-fit-v2-epoch-3",
                local_files_only=False
            )
            self.resume_model.eval()
        
        if self.skill_model is None:
            self.skill_model = SentenceTransformer('paraphrase-albert-small-v2')

    def load_job_specific_vocabulary(self):
        """Load curated job-specific vocabulary"""
        return {
            'developed', 'implemented', 'managed', 'led', 'architected',
            'designed', 'optimized', 'reduced', 'increased', 'improved',
            'team', 'project', 'system', 'application', 'software',
            'infrastructure', 'platform', 'service', 'solution', 'database',
            'performance', 'scalability', 'reliability', 'security', 'deployment'
        }

    def mean_pooling(self, model_output, attention_mask):
        """Perform mean pooling on the token embeddings"""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def get_distilbert_embeddings(self, texts):
        """Get embeddings using the resume-specific DistilBERT model"""
        # Tokenize sentences
        encoded_input = self.resume_tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=512)
        
        # Compute token embeddings
        with torch.no_grad():
            model_output = self.resume_model(**encoded_input)
        
        # Perform pooling
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        
        # Normalize embeddings
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings

    def calculate_hybrid_similarity(self, resume_text, job_description):
        """Calculate similarities using DistilBERT and other models"""
        # DistilBERT similarity for resume-specific matching
        resume_embeddings = self.get_distilbert_embeddings([job_description, resume_text])
        distilbert_sim = cosine_similarity(
            resume_embeddings[0].reshape(1, -1).numpy(),
            resume_embeddings[1].reshape(1, -1).numpy()
        )[0][0]
        
        # Skill similarity using ALBERT
        skill_embeddings = self.skill_model.encode([job_description, resume_text])
        skill_sim = cosine_similarity([skill_embeddings[0]], [skill_embeddings[1]])[0][0]
        
        # TF-IDF similarity
        processed = [
            self.enhanced_preprocess(job_description),
            self.enhanced_preprocess(resume_text)
        ]
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(processed)
        tfidf_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        # Technical skill match
        tech_sim = self.calculate_skill_match(resume_text, job_description)
        
        return {
            'distilbert_sim': distilbert_sim,
            'skill_sim': skill_sim,
            'tfidf_sim': tfidf_sim,
            'tech_sim': tech_sim
        }

    def score_resume(self, resume_text, job_description):
        """Score resume with updated weights for DistilBERT"""
        similarities = self.calculate_hybrid_similarity(resume_text, job_description)
        
        # Extract skills
        job_skills = self.extract_technical_skills(job_description)
        resume_skills = self.extract_technical_skills(resume_text)
        
        # Calculate missing skills
        missing_skills = {
            category: list(job_skills[category] - resume_skills[category])
            for category in job_skills
            if job_skills[category]
        }
        
        # Updated weights to emphasize DistilBERT's resume-specific matching
        final_score = (
            similarities['distilbert_sim'] * 0.4 +  # Increased weight for resume-specific matching
            similarities['skill_sim'] * 0.25 +
            similarities['tfidf_sim'] * 0.15 +
            similarities['tech_sim'] * 0.2
        ) * 100
        
        return {
            'overall_score': round(final_score, 2),
            'score_breakdown': {
                'content_similarity': round(similarities['distilbert_sim'] * 100, 2),
                'technical_match': round(similarities['tech_sim'] * 100, 2),
                'context_understanding': round(similarities['skill_sim'] * 100, 2)
            },
            'missing_skills': missing_skills,
            'feedback': self.generate_feedback(similarities, missing_skills)
        }

    def calculate_skill_match(self, resume_text, job_description):
        """Calculate technical skill match score"""
        job_skills = self.extract_technical_skills(job_description)
        resume_skills = self.extract_technical_skills(resume_text)
        
        skill_match_scores = []
        for category in job_skills:
            if job_skills[category]:
                match_ratio = len(resume_skills[category] & job_skills[category]) / len(job_skills[category])
                skill_match_scores.append(match_ratio)
        
        return np.mean(skill_match_scores) if skill_match_scores else 0

    def generate_feedback(self, similarities, missing_skills):
        """Generate actionable feedback"""
        feedback = []
        
        # Technical skill feedback
        if similarities['tech_sim'] < 0.7:
            feedback.append("🔧 Technical Skill Gaps:")
            for category, skills in missing_skills.items():
                if skills:
                    feedback.append(f"- {category.title()}: Missing {', '.join(skills)}")
        
        # Content match feedback
        if similarities['distilbert_sim'] < 0.7:
            feedback.append("📝 Content Improvement Suggestions:")
            feedback.append("- Add more specific details about your role and responsibilities")
            feedback.append("- Include quantifiable achievements and metrics")
            feedback.append("- Highlight relevant projects and technologies used")
        
        # Keyword match feedback
        if similarities['tfidf_sim'] < 0.65:
            feedback.append("🎯 Keyword Alignment:")
            feedback.append("- Align your terminology with the job description")
            feedback.append("- Include industry-standard terms and tools")
            feedback.append("- Mention specific methodologies used")
        
        if not feedback:
            feedback.append("✅ Strong Match!")
            feedback.append("💪 Your resume aligns well with the job requirements")
        
        return feedback

    def enhanced_preprocess(self, text):
        """Preprocess text for analysis"""
        text = text.lower()
        text = re.sub(r'([./\\])', r' \1 ', text)
        text = re.sub(r'([A-Za-z])(\d)', r'\1 \2', text)
        
        doc = self.nlp(text)
        tokens = []
        
        for token in doc:
            if any(token.text in terms for terms in self.tech_categories.values()):
                tokens.append(token.text)
            elif not token.is_stop and not token.is_punct and not token.is_space:
                tokens.append(token.lemma_)
                
        return ' '.join(tokens)
    
    def extract_technical_skills(self, text):
        """Extract technical skills from text"""
        skills = defaultdict(set)
        text_lower = text.lower()
        
        for category, terms in self.tech_categories.items():
            for term in terms:
                if term in text_lower:
                    skills[category].add(term)
        
        version_pattern = r'\b(?:[a-zA-Z.]+(?:js)?(?:\d+(?:\.\d+)*)?)\b'
        versions = re.findall(version_pattern, text)
        skills['versions'] = set(versions)
        
        return skills
    
