import spacy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import torch
from collections import defaultdict
import re
import json

class EnhancedATSScorer:
    def __init__(self):
        """Initialize with improved models and section analysis capabilities"""
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except:
            self.nlp = spacy.blank("en")
            
        # Add sentencizer component to properly detect sentence boundaries
        if 'sentencizer' not in self.nlp.pipe_names:
            self.nlp.add_pipe('sentencizer')
        
        # Initialize sentence transformer model
        self.sentence_model = SentenceTransformer('all-mpnet-base-v2')
        
        # Technical skill categories with expanded terms
        self.tech_categories = {
            'programming_languages': set([
                'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'go', 'golang', 
                'rust', 'swift', 'kotlin', 'php', 'ruby', 'scala', 'perl', 'r', 'matlab', 
                'shell', 'bash', 'powershell', 'haskell', 'clojure', 'groovy', 'dart', 
                'objective-c', 'sql', 'plsql', 'cobol', 'fortran', 'assembly'
            ]),
            'frameworks': set([
                'react', 'angular', 'vue', 'svelte', 'django', 'flask', 'fastapi', 'spring', 
                'spring boot', '.net', 'asp.net', 'laravel', 'symfony', 'express', 'nextjs', 
                'nestjs', 'rails', 'gatsby', 'flutter', 'pytorch', 'tensorflow', 'keras', 
                'pandas', 'numpy', 'scikit-learn', 'apache spark', 'hadoop', 'jquery', 
                'bootstrap', 'tailwind', 'material ui', 'chakra ui'
            ]),
            'cloud': set([
                'aws', 'amazon web services', 'azure', 'microsoft azure', 'gcp', 'google cloud', 
                'kubernetes', 'k8s', 'docker', 'containerization', 'terraform', 'cloudformation', 
                'openshift', 'openstack', 'heroku', 'netlify', 'vercel', 'digital ocean', 
                'lambda', 'ec2', 's3', 'dynamodb', 'rds', 'ecs', 'eks', 'fargate', 'cloudfront', 
                'route53', 'iam', 'cloudwatch', 'sqs', 'sns', 'kinesis', 'redshift'
            ]),
            'databases': set([
                'sql', 'mysql', 'postgresql', 'postgres', 'oracle', 'sql server', 'mongodb', 
                'dynamodb', 'cassandra', 'redis', 'elasticsearch', 'neo4j', 'couchbase', 
                'firebase', 'mariadb', 'sqlite', 'supabase', 'cockroachdb', 'influxdb', 
                'timeseriesdb', 'graphql', 'cosmos db', 'hbase', 'bigtable'
            ]),
            'tools': set([
                'git', 'github', 'gitlab', 'bitbucket', 'ci/cd', 'jenkins', 'travis', 'circle ci', 
                'github actions', 'jira', 'confluence', 'agile', 'scrum', 'kanban', 'terraform', 
                'ansible', 'puppet', 'chef', 'prometheus', 'grafana', 'elk stack', 'kibana', 
                'logstash', 'datadog', 'new relic', 'splunk', 'tableau', 'power bi', 'looker', 
                'sentry', 'sonarqube', 'postman', 'swagger', 'openapi'
            ]),
            'methodologies': set([
                'agile', 'scrum', 'kanban', 'waterfall', 'devops', 'gitops', 'devsecops', 
                'test driven development', 'tdd', 'behavior driven development', 'bdd', 
                'continuous integration', 'continuous delivery', 'continuous deployment', 
                'pair programming', 'microservices', 'serverless', 'domain driven design', 
                'ddd', 'event sourcing', 'cqrs', 'mvc', 'mvvm', 'clean architecture', 
                'solid principles', 'design patterns', 'refactoring'
            ]),
            'certifications': set([
                'aws certified', 'microsoft certified', 'google certified', 'cissp', 'ceh', 
                'comptia', 'pmp', 'scrum master', 'csm', 'psm', 'safe', 'itil', 'cka', 'ckad', 
                'rhce', 'rhcsa', 'oracle certified', 'cisco certified', 'ccna', 'ccnp', 'ccie', 
                'azure', 'gcp', 'hashicorp', 'terraform', 'kubernetes'
            ])
        }
        
        # Section importance weights (higher = more important)
        self.section_weights = {
            'work_experience': 5.0,
            'projects': 4.0,
            'skills': 2.0,
            'education': 1.5,
            'certifications': 2.5,
            'other': 1.0
        }
        
        # Regular expressions to identify resume sections
        self.section_patterns = {
            'work_experience': re.compile(r'(work\s+experience|professional\s+experience|employment(\s+history)?|experience)', re.IGNORECASE),
            'projects': re.compile(r'(projects|personal\s+projects|professional\s+projects|key\s+projects)', re.IGNORECASE),
            'skills': re.compile(r'(technical\s+skills|skills(\s+and\s+abilities)?|qualifications|technical\s+qualifications)', re.IGNORECASE),
            'education': re.compile(r'(education|academic|qualifications|educational\s+background)', re.IGNORECASE),
            'certifications': re.compile(r'(certifications|professional\s+certifications|credentials)', re.IGNORECASE)
        }
        
        # Advanced job_specific vocabulary
        self.job_specific_vocabulary = self._load_job_specific_vocabulary()
        
        # TF-IDF for keyword matching
        self.tfidf_vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            stop_words='english',
            max_features=5000,
            vocabulary=self.job_specific_vocabulary
        )

    def _load_job_specific_vocabulary(self):
        """Load expanded job-specific vocabulary"""
        return {
            # Achievement-oriented terms
            'developed', 'implemented', 'managed', 'led', 'architected', 'designed', 
            'optimized', 'reduced', 'increased', 'improved', 'accelerated', 'streamlined',
            'enhanced', 'launched', 'delivered', 'coordinated', 'spearheaded', 'pioneered', 
            'transformed', 'revamped', 'established', 'generated', 'secured', 'achieved',
            
            # Technical context terms
            'team', 'project', 'system', 'application', 'software', 'infrastructure', 
            'platform', 'service', 'solution', 'database', 'architecture', 'framework',
            'pipeline', 'protocol', 'algorithm', 'interface', 'network', 'environment',
            'repository', 'codebase', 'workflow', 'automation', 'deployment', 'integration',
            
            # Quality metrics
            'performance', 'scalability', 'reliability', 'security', 'efficiency', 
            'quality', 'usability', 'accessibility', 'productivity', 'velocity',
            'throughput', 'latency', 'availability', 'maintainability', 'resilience',
            
            # Business impact terms
            'revenue', 'cost', 'roi', 'profit', 'savings', 'efficiency', 'growth',
            'customer', 'client', 'stakeholder', 'user', 'experience', 'satisfaction',
            'retention', 'engagement', 'conversion', 'acquisition', 'churn', 'metrics'
        }

    def analyze_resume_sections(self, resume_text):
        """Identify and analyze different sections of the resume"""
        # Split resume into lines and normalize whitespace
        lines = [line.strip() for line in resume_text.split('\n') if line.strip()]
        
        # Identify section boundaries
        sections = {}
        current_section = 'other'
        section_content = []
        
        for i, line in enumerate(lines):
            # Check if this line is a section header
            matched_section = None
            for section, pattern in self.section_patterns.items():
                if pattern.match(line) and (i == 0 or len(line) < 50):  # Reasonable header length
                    matched_section = section
                    break
            
            # If a new section is found, save the previous section content
            if matched_section:
                if section_content:
                    sections[current_section] = '\n'.join(section_content)
                current_section = matched_section
                section_content = []
            else:
                section_content.append(line)
        
        # Add the last section
        if section_content:
            sections[current_section] = '\n'.join(section_content)
        
        return sections

    def calculate_section_skill_scores(self, sections, job_skills):
        """Calculate skill matches by section with weighted importance"""
        section_scores = {}
        total_weight = 0
        total_weighted_score = 0
        
        for section_name, content in sections.items():
            section_skills = self.extract_technical_skills(content)
            weight = self.section_weights.get(section_name, self.section_weights['other'])
            
            # Calculate skill match for this section
            section_score = 0
            total_possible = 0
            
            for category, skills in job_skills.items():
                if skills:
                    matched = len(section_skills[category] & skills)
                    total = len(skills)
                    if total > 0:
                        section_score += matched
                        total_possible += total
            
            if total_possible > 0:
                normalized_score = section_score / total_possible
                section_scores[section_name] = {
                    'raw_score': normalized_score,
                    'weighted_score': normalized_score * weight,
                    'weight': weight,
                    'matched_skills': {
                        category: list(section_skills[category] & job_skills[category])
                        for category in job_skills
                    }
                }
                
                total_weight += weight
                total_weighted_score += normalized_score * weight
        
        # Calculate overall weighted skill score
        overall_skill_score = total_weighted_score / total_weight if total_weight > 0 else 0
        
        return {
            'section_scores': section_scores,
            'overall_weighted_skill_score': overall_skill_score
        }

    def calculate_keyword_importance(self, job_description):
        """Calculate importance of keywords in job description"""
        doc = self.nlp(job_description)
        
        # Extract sentences and calculate their embeddings
        sentences = [sent.text for sent in doc.sents]
        embeddings = self.sentence_model.encode(sentences)
        
        # Calculate keyword frequency and position importance
        keyword_scores = defaultdict(float)
        terms = set()
        
        for category, skills in self.tech_categories.items():
            terms.update(skills)
        
        # Add job-specific vocabulary
        terms.update(self.job_specific_vocabulary)
        
        # Calculate scores for each keyword
        for term in terms:
            # Find all occurrences
            term_regex = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
            matches = list(term_regex.finditer(job_description))
            
            if matches:
                # Base score from frequency
                freq_score = len(matches) / len(doc)
                
                # Position score - earlier mentions may be more important
                positions = [match.start() / len(job_description) for match in matches]
                position_score = sum(1 - pos for pos in positions) / len(positions)
                
                # Context score - terms in requirement-heavy sentences are more important
                context_score = 0
                for match in matches:
                    # Find which sentence contains this match
                    for i, sent in enumerate(doc.sents):
                        if sent.start_char <= match.start() < sent.end_char:
                            # Check if sentence contains requirement language
                            sent_text = sent.text.lower()
                            if any(req in sent_text for req in ['required', 'must', 'need', 'essential']):
                                context_score += 1.5
                            break
                
                # Combine scores
                keyword_scores[term] = freq_score + position_score + (context_score / len(matches))
        
        return dict(keyword_scores)

    def extract_technical_skills(self, text):
        """Extract technical skills from text with improved pattern matching"""
        skills = defaultdict(set)
        text_lower = text.lower()
        
        # Use lemmatization for better matching
        doc = self.nlp(text_lower)
        lemmatized_text = ' '.join([token.lemma_ for token in doc])
        
        # Match skills with word boundaries
        for category, terms in self.tech_categories.items():
            for term in terms:
                # Check with word boundaries
                pattern = r'\b' + re.escape(term) + r'\b'
                if re.search(pattern, text_lower) or re.search(pattern, lemmatized_text):
                    skills[category].add(term)
        
        # Extract version information
        version_pattern = r'\b(?:[a-zA-Z.]+(?:js)?(?:\d+(?:\.\d+)*)?)\b'
        versions = re.findall(version_pattern, text_lower)
        skills['versions'] = set(versions)
        
        return skills

    def score_resume(self, resume_text, job_description):
        """Score resume with weighted section analysis and contextual importance"""
        # Preprocess and clean the texts
        resume_text = self._clean_text(resume_text)
        job_description = self._clean_text(job_description)
        
        # Extract required skills from job description
        job_skills = self.extract_technical_skills(job_description)
        
        # Calculate keyword importance
        keyword_importance = self.calculate_keyword_importance(job_description)
        
        # Analyze resume sections
        resume_sections = self.analyze_resume_sections(resume_text)
        
        # Calculate skill match by resume section
        section_skill_analysis = self.calculate_section_skill_scores(resume_sections, job_skills)
        
        # Get semantic similarity score
        semantic_similarity = self.calculate_semantic_similarity(resume_text, job_description)
        
        # Calculate keyword match with TF-IDF
        processed = [
            self._enhanced_preprocess(job_description),
            self._enhanced_preprocess(resume_text)
        ]
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(processed)
        tfidf_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        # Get missing skills across all sections
        resume_skills = self.extract_technical_skills(resume_text)
        missing_skills = {
            category: list(job_skills[category] - resume_skills[category])
            for category in job_skills
            if job_skills[category]
        }
        
        # Calculate overall score with improved weighting
        overall_score = (
            section_skill_analysis['overall_weighted_skill_score'] * 0.45 +
            semantic_similarity * 0.35 +
            tfidf_sim * 0.20
        ) * 100
        
        return {
            'overall_score': round(overall_score, 2),
            'score_breakdown': {
                'section_analysis': section_skill_analysis['section_scores'],
                'semantic_similarity': round(semantic_similarity * 100, 2),
                'keyword_match': round(tfidf_sim * 100, 2),
                'technical_match': round(section_skill_analysis['overall_weighted_skill_score'] * 100, 2)
            },
            'missing_skills': missing_skills,
            'section_analysis': self._format_section_analysis(section_skill_analysis['section_scores']),
            'feedback': self._generate_enhanced_feedback(
                section_skill_analysis, 
                semantic_similarity, 
                tfidf_sim, 
                missing_skills, 
                keyword_importance,
                resume_sections
            )
        }

    def calculate_semantic_similarity(self, resume_text, job_description):
        """Calculate semantic similarity using sentence transformers"""
        # Encode texts to get embeddings
        embeddings = self.sentence_model.encode([job_description, resume_text])
        
        # Calculate cosine similarity
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        
        return similarity

    def _clean_text(self, text):
        """Clean and normalize text"""
        # Replace multiple spaces and newlines with single spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove phone numbers
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '', text)
        
        return text.strip()

    def _enhanced_preprocess(self, text):
        """Enhanced preprocessing for text analysis"""
        text = text.lower()
        
        # Preserve important characters in technical terms
        text = re.sub(r'([./\\])', r' \1 ', text)
        text = re.sub(r'([A-Za-z])(\d)', r'\1 \2', text)
        
        # Process with spaCy for better lemmatization
        doc = self.nlp(text)
        tokens = []
        
        for token in doc:
            # Preserve technical terms
            if any(token.text.lower() in terms for terms in self.tech_categories.values()):
                tokens.append(token.text.lower())
            # Handle normal tokens
            elif not token.is_stop and not token.is_punct and not token.is_space:
                tokens.append(token.lemma_)
                
        return ' '.join(tokens)

    def _format_section_analysis(self, section_scores):
        """Format section analysis for better readability"""
        formatted = {}
        
        for section, data in section_scores.items():
            formatted[section] = {
                'score': round(data['raw_score'] * 100, 2),
                'importance': data['weight'],
                'skills_found': sum(len(skills) for skills in data['matched_skills'].values())
            }
            
        return formatted

    def _generate_enhanced_feedback(self, section_analysis, semantic_similarity, 
                                   keyword_match, missing_skills, keyword_importance,
                                   resume_sections):
        """Generate detailed actionable feedback based on comprehensive analysis"""
        feedback = []
        
        # 1. Missing skills feedback with prioritization based on importance
        if missing_skills:
            feedback.append("🔧 **Critical Skill Gaps:**")
            
            # Sort categories by importance
            prioritized_skills = {}
            for category, skills in missing_skills.items():
                if skills:
                    # Get average importance of skills in this category
                    avg_importance = sum(keyword_importance.get(skill, 0) for skill in skills) / len(skills)
                    prioritized_skills[category] = {
                        'skills': skills,
                        'importance': avg_importance
                    }
            
            # Sort categories by importance
            sorted_categories = sorted(
                prioritized_skills.keys(),
                key=lambda x: prioritized_skills[x]['importance'],
                reverse=True
            )
            
            for category in sorted_categories:
                skills = prioritized_skills[category]['skills']
                feedback.append(f"- {category.title()}: Missing {', '.join(skills)}")
        
        # 2. Section-specific feedback
        weak_sections = []
        for section, data in section_analysis['section_scores'].items():
            if data['raw_score'] < 0.6:  # Below 60% match
                weak_sections.append((section, data['raw_score']))
        
        if weak_sections:
            feedback.append("\n📊 **Section-Specific Improvements:**")
            
            # Sort sections by lowest score first
            sorted_sections = sorted(weak_sections, key=lambda x: x[1])
            
            for section, score in sorted_sections:
                if section == 'work_experience':
                    feedback.append(f"- **Work Experience**: Highlight relevant skills more explicitly in job descriptions (current match: {score*100:.1f}%)")
                    feedback.append("  - Quantify achievements with metrics and specifics")
                    feedback.append("  - Begin bullet points with strong action verbs")
                    feedback.append("  - Explicitly mention tools and technologies used in each role")
                
                elif section == 'projects':
                    feedback.append(f"- **Projects**: Enhance project descriptions with in-demand skills (current match: {score*100:.1f}%)")
                    feedback.append("  - Focus on projects that showcase required skills from the job description")
                    feedback.append("  - Highlight your role and specific technologies used")
                
                elif section == 'skills':
                    feedback.append(f"- **Skills Section**: Update to better reflect job requirements (current match: {score*100:.1f}%)")
                    feedback.append("  - Organize skills by category for better readability")
                    feedback.append("  - Include versions/certifications where applicable")
                
                elif section == 'education':
                    feedback.append(f"- **Education**: Enhance with relevant coursework/projects (current match: {score*100:.1f}%)")
                
                elif section == 'certifications':
                    feedback.append(f"- **Certifications**: Consider adding relevant certifications (current match: {score*100:.1f}%)")
        
        # 3. Context and semantic feedback
        if semantic_similarity < 0.65:
            feedback.append("\n📝 **Content Alignment Suggestions:**")
            feedback.append("- Use more terminology and phrasing similar to the job posting")
            feedback.append("- Match the tone and level of technical detail in the job description")
            feedback.append("- Restructure resume to prioritize the most relevant experience first")
            
            # Check for important missing sections
            if 'work_experience' not in resume_sections:
                feedback.append("- Your resume may be missing a clearly labeled Work Experience section")
            if 'skills' not in resume_sections:
                feedback.append("- Add a dedicated Skills section to improve ATS recognition")
        
        # 4. Keyword optimization feedback
        if keyword_match < 0.55:
            feedback.append("\n🎯 **Keyword Optimization:**")
            
            # Find most important keywords missing
            resume_text = ' '.join(resume_sections.values()).lower()
            important_missing = []
            
            for term, importance in sorted(keyword_importance.items(), key=lambda x: x[1], reverse=True)[:15]:
                if importance > 0.5 and re.search(r'\b' + re.escape(term) + r'\b', resume_text) is None:
                    important_missing.append(term)
            
            if important_missing:
                feedback.append(f"- Consider adding these high-importance terms: {', '.join(important_missing[:5])}")
            
            feedback.append("- Use exact phrases from the job description where applicable")
            feedback.append("- Include industry-standard terminology and methodologies")
        
        # 5. General ATS optimization tips
        if not feedback or (semantic_similarity > 0.7 and section_analysis['overall_weighted_skill_score'] > 0.7):
            feedback.append("✅ **Strong Overall Match!**")
            feedback.append("💪 Your resume aligns well with the job requirements")
            feedback.append("\n📈 **Fine-tuning Suggestions:**")
        else:
            feedback.append("\n🤖 **ATS Optimization Tips:**")
        
        feedback.append("- Use standard section headings that ATS systems can easily recognize")
        feedback.append("- Avoid tables, graphics, headers/footers, and unusual formatting")
        feedback.append("- Save your resume in a standard format (PDF from Word is usually best)")
        feedback.append("- Place most important information in top third of first page")
        
        return feedback


class MistralAIEnhancer:
    """
    Enhances ATS resume analysis using Mistral AI's advanced language understanding capabilities.
    This class integrates with Mistral's API to provide more sophisticated analysis of resumes
    against job descriptions.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Mistral AI enhancer.
        
        Args:
            api_key: Mistral AI API key. If None, will try to read from MISTRAL_API_KEY environment variable.
        """
        self.api_key = api_key or os.environ.get("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("Mistral API key is required. Set it either through the constructor or MISTRAL_API_KEY environment variable.")
        
        self.api_url = "https://api.mistral.ai/v1/chat/completions"
        self.base_scorer = EnhancedATSScorer()
    
    def analyze_resume_with_mistral(self, resume_text: str, job_description: str) -> Dict[str, Any]:
        """
        Perform an enhanced analysis of a resume against a job description using Mistral AI.
        
        Args:
            resume_text: The text content of the resume
            job_description: The text content of the job description
            
        Returns:
            Dict containing enhanced analysis results
        """
        # First get the base analysis from our enhanced scorer
        base_results = self.base_scorer.score_resume(resume_text, job_description)
        
        # Extract resume sections
        resume_sections = self.base_scorer.analyze_resume_sections(resume_text)
        
        # Get Mistral AI enhancements
        mistral_enhancements = self._get_mistral_insights(
            resume_text=resume_text,
            job_description=job_description,
            base_results=base_results,
            resume_sections=resume_sections
        )
        
        # Merge the results
        enhanced_results = self._merge_results(base_results, mistral_enhancements)
        
        return enhanced_results
    
    def _get_mistral_insights(
        self, 
        resume_text: str, 
        job_description: str, 
        base_results: Dict[str, Any],
        resume_sections: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Get insights from Mistral AI to enhance our analysis.
        
        Args:
            resume_text: The text content of the resume
            job_description: The text content of the job description
            base_results: Results from the base scorer
            resume_sections: The resume broken down into sections
            
        Returns:
            Dict containing Mistral AI insights
        """
        # Create a structured prompt for Mistral
        prompt = self._create_mistral_prompt(
            resume_text=resume_text,
            job_description=job_description,
            base_results=base_results,
            resume_sections=resume_sections
        )
        
        # Call Mistral API
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": "mistral-large-latest",  # Use appropriate model
            "messages": [
                {"role": "system", "content": "You are an advanced ATS (Applicant Tracking System) analyzer that helps improve resume-to-job matching. You provide expert insights on resumes and job descriptions."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2,  # Lower temperature for more consistent outputs
            "response_format": {"type": "json_object"}  # Request structured JSON response
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()
            
            # Parse response
            result = response.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "{}")
            
            # Parse the JSON response
            mistral_insights = json.loads(content)
            return mistral_insights
            
        except Exception as e:
            print(f"Error calling Mistral AI: {e}")
            # Return a minimal structure if API call fails
            return {
                "enhanced_feedback": [],
                "industry_insights": [],
                "competitive_analysis": {},
                "improvement_priority": []
            }
    
    def _create_mistral_prompt(
        self, 
        resume_text: str, 
        job_description: str, 
        base_results: Dict[str, Any],
        resume_sections: Dict[str, str]
    ) -> str:
        """
        Create a detailed prompt for Mistral AI.
        
        Args:
            resume_text: The text content of the resume
            job_description: The text content of the job description
            base_results: Results from the base scorer
            resume_sections: The resume broken down into sections
            
        Returns:
            A formatted prompt string
        """
        # Convert base results to a compact JSON string
        base_results_json = json.dumps({
            "overall_score": base_results["overall_score"],
            "score_breakdown": base_results["score_breakdown"],
            "missing_skills": base_results["missing_skills"],
            "section_analysis": base_results["section_analysis"]
        })
        
        # Create sections summary
        sections_summary = "\n".join([
            f"--- {section.upper()} SECTION ---\n{content[:300]}..."
            for section, content in resume_sections.items()
        ])
        
        # Build the prompt with clear instructions
        prompt = f"""
As an advanced ATS analyzer, I need your expertise to enhance my analysis of this resume against a job description.

JOB DESCRIPTION:
```
{job_description[:2000]}
```

RESUME SUMMARY BY SECTION:
```
{sections_summary}
```

BASE ANALYSIS RESULTS:
```json
{base_results_json}
```

Please provide your advanced analysis in the following JSON format:

```json
{{
    "enhanced_feedback": [
        // 5-7 specific suggestions to improve the resume for this exact job
        // Focus on content, structure, emphasis, and wording improvements
    ],
    "industry_insights": [
        // 3-5 insights about this specific industry/role and how the resume could better target them
    ],
    "competitive_analysis": {{
        "strengths": [
            // 2-3 areas where this candidate stands out compared to typical applicants
        ],
        "weaknesses": [
            // 2-3 areas where this candidate may fall behind typical applicants
        ]
    }},
    "improvement_priority": [
        // List of 3-5 changes, ordered by impact (highest impact first)
        // Each item should be specific and actionable
    ],
    "ats_optimization_tips": [
        // 3-5 specific tips to make this resume more ATS-friendly
    ]
}}
```

Focus on providing valuable, specific, and actionable insights that go beyond basic ATS optimization.
"""
        return prompt
    
    def _merge_results(self, base_results: Dict[str, Any], mistral_enhancements: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge base results with Mistral AI enhancements.
        
        Args:
            base_results: Results from the base scorer
            mistral_enhancements: Insights from Mistral AI
            
        Returns:
            Combined results
        """
        # Create a copy of base results
        enhanced_results = base_results.copy()
        
        # Add Mistral enhancements
        enhanced_results["mistral_insights"] = {
            "enhanced_feedback": mistral_enhancements.get("enhanced_feedback", []),
            "industry_insights": mistral_enhancements.get("industry_insights", []),
            "competitive_analysis": mistral_enhancements.get("competitive_analysis", {}),
            "improvement_priority": mistral_enhancements.get("improvement_priority", []),
            "ats_optimization_tips": mistral_enhancements.get("ats_optimization_tips", [])
        }
        
        # Enhance feedback if Mistral provided insights
        if mistral_enhancements.get("enhanced_feedback"):
            # Combine base feedback with enhanced feedback
            enhanced_results["feedback"] = base_results["feedback"] + [
                "\n🔍 **Advanced AI Insights:**"
            ] + mistral_enhancements.get("enhanced_feedback", [])
        
        return enhanced_results


class MistralAIATSScorer:
    """
    Complete ATS Scorer implementation using Mistral AI.
    This class provides a unified interface for resume analysis with Mistral enhancements.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Mistral-enhanced ATS scorer.
        
        Args:
            api_key: Mistral AI API key. If None, will try to use EnhancedATSScorer as fallback.
        """
        try:
            self.mistral_enhancer = MistralAIEnhancer(api_key)
            self.using_mistral = True
        except (ValueError, ImportError):
            # Fall back to base scorer if Mistral setup fails
            self.base_scorer = EnhancedATSScorer()
            self.using_mistral = False
    
    def score_resume(self, resume_text: str, job_description: str) -> Dict[str, Any]:
        """
        Score resume against job description with potential Mistral enhancements.
        
        Args:
            resume_text: The text content of the resume
            job_description: The text content of the job description
            
        Returns:
            Dict containing analysis results
        """
        if self.using_mistral:
            return self.mistral_enhancer.analyze_resume_with_mistral(resume_text, job_description)
        else:
            return self.base_scorer.score_resume(resume_text, job_description)
    
    def analyze_resume_sections(self, resume_text: str) -> Dict[str, str]:
        """
        Analyze and extract sections from the resume.
        
        Args:
            resume_text: The text content of the resume
            
        Returns:
            Dict mapping section names to their content
        """
        if self.using_mistral:
            return self.mistral_enhancer.base_scorer.analyze_resume_sections(resume_text)
        else:
            return self.base_scorer.analyze_resume_sections(resume_text)


# Example usage
if __name__ == "__main__":
    # Example usage with environment variable
    # os.environ["MISTRAL_API_KEY"] = "your-api-key"
    
    try:
        scorer = MistralAIATSScorer()
        print(f"Using Mistral AI: {scorer.using_mistral}")
        
        # Sample data
        resume = "Your resume text here..."
        job_desc = "Job description here..."
        
        # Analyze
        results = scorer.score_resume(resume, job_desc)
        print(f"Overall score: {results['overall_score']}%")
        
        if 'mistral_insights' in results:
            print("\nMistral AI Insights:")
            for key, value in results['mistral_insights'].items():
                print(f"\n{key.upper()}:")
                if isinstance(value, list):
                    for item in value:
                        print(f"- {item}")
                elif isinstance(value, dict):
                    for k, v in value.items():
                        print(f"\n{k}:")
                        for item in v:
                            print(f"- {item}")
    
    except Exception as e:
        print(f"Error: {e}")
        print("Falling back to base ATS scorer...")
        
        scorer = EnhancedATSScorer()
        # Continue with base scorer



# import spacy
# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# from sentence_transformers import SentenceTransformer
# import torch
# from collections import defaultdict
# import re
# import json

# class EnhancedATSScorer:
#     def __init__(self):
#         """Initialize with improved models and section analysis capabilities"""
#         try:
#             self.nlp = spacy.load('en_core_web_sm')
#         except:
#             self.nlp = spacy.blank("en")
        
#         # Initialize sentence transformer model
#         self.sentence_model = SentenceTransformer('all-mpnet-base-v2')
        
#         # Technical skill categories with expanded terms
#         self.tech_categories = {
#             'programming_languages': set([
#                 'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'go', 'golang', 
#                 'rust', 'swift', 'kotlin', 'php', 'ruby', 'scala', 'perl', 'r', 'matlab', 
#                 'shell', 'bash', 'powershell', 'haskell', 'clojure', 'groovy', 'dart', 
#                 'objective-c', 'sql', 'plsql', 'cobol', 'fortran', 'assembly'
#             ]),
#             'frameworks': set([
#                 'react', 'angular', 'vue', 'svelte', 'django', 'flask', 'fastapi', 'spring', 
#                 'spring boot', '.net', 'asp.net', 'laravel', 'symfony', 'express', 'nextjs', 
#                 'nestjs', 'rails', 'gatsby', 'flutter', 'pytorch', 'tensorflow', 'keras', 
#                 'pandas', 'numpy', 'scikit-learn', 'apache spark', 'hadoop', 'jquery', 
#                 'bootstrap', 'tailwind', 'material ui', 'chakra ui'
#             ]),
#             'cloud': set([
#                 'aws', 'amazon web services', 'azure', 'microsoft azure', 'gcp', 'google cloud', 
#                 'kubernetes', 'k8s', 'docker', 'containerization', 'terraform', 'cloudformation', 
#                 'openshift', 'openstack', 'heroku', 'netlify', 'vercel', 'digital ocean', 
#                 'lambda', 'ec2', 's3', 'dynamodb', 'rds', 'ecs', 'eks', 'fargate', 'cloudfront', 
#                 'route53', 'iam', 'cloudwatch', 'sqs', 'sns', 'kinesis', 'redshift'
#             ]),
#             'databases': set([
#                 'sql', 'mysql', 'postgresql', 'postgres', 'oracle', 'sql server', 'mongodb', 
#                 'dynamodb', 'cassandra', 'redis', 'elasticsearch', 'neo4j', 'couchbase', 
#                 'firebase', 'mariadb', 'sqlite', 'supabase', 'cockroachdb', 'influxdb', 
#                 'timeseriesdb', 'graphql', 'cosmos db', 'hbase', 'bigtable'
#             ]),
#             'tools': set([
#                 'git', 'github', 'gitlab', 'bitbucket', 'ci/cd', 'jenkins', 'travis', 'circle ci', 
#                 'github actions', 'jira', 'confluence', 'agile', 'scrum', 'kanban', 'terraform', 
#                 'ansible', 'puppet', 'chef', 'prometheus', 'grafana', 'elk stack', 'kibana', 
#                 'logstash', 'datadog', 'new relic', 'splunk', 'tableau', 'power bi', 'looker', 
#                 'sentry', 'sonarqube', 'postman', 'swagger', 'openapi'
#             ]),
#             'methodologies': set([
#                 'agile', 'scrum', 'kanban', 'waterfall', 'devops', 'gitops', 'devsecops', 
#                 'test driven development', 'tdd', 'behavior driven development', 'bdd', 
#                 'continuous integration', 'continuous delivery', 'continuous deployment', 
#                 'pair programming', 'microservices', 'serverless', 'domain driven design', 
#                 'ddd', 'event sourcing', 'cqrs', 'mvc', 'mvvm', 'clean architecture', 
#                 'solid principles', 'design patterns', 'refactoring'
#             ]),
#             'certifications': set([
#                 'aws certified', 'microsoft certified', 'google certified', 'cissp', 'ceh', 
#                 'comptia', 'pmp', 'scrum master', 'csm', 'psm', 'safe', 'itil', 'cka', 'ckad', 
#                 'rhce', 'rhcsa', 'oracle certified', 'cisco certified', 'ccna', 'ccnp', 'ccie', 
#                 'azure', 'gcp', 'hashicorp', 'terraform', 'kubernetes'
#             ])
#         }
        
#         # Section importance weights (higher = more important)
#         self.section_weights = {
#             'work_experience': 5.0,
#             'projects': 4.0,
#             'skills': 2.0,
#             'education': 1.5,
#             'certifications': 2.5,
#             'other': 1.0
#         }
        
#         # Regular expressions to identify resume sections
#         self.section_patterns = {
#             'work_experience': re.compile(r'(work\s+experience|professional\s+experience|employment(\s+history)?|experience)', re.IGNORECASE),
#             'projects': re.compile(r'(projects|personal\s+projects|professional\s+projects|key\s+projects)', re.IGNORECASE),
#             'skills': re.compile(r'(technical\s+skills|skills(\s+and\s+abilities)?|qualifications|technical\s+qualifications)', re.IGNORECASE),
#             'education': re.compile(r'(education|academic|qualifications|educational\s+background)', re.IGNORECASE),
#             'certifications': re.compile(r'(certifications|professional\s+certifications|credentials)', re.IGNORECASE)
#         }
        
#         # Advanced job_specific vocabulary
#         self.job_specific_vocabulary = self._load_job_specific_vocabulary()
        
#         # TF-IDF for keyword matching
#         self.tfidf_vectorizer = TfidfVectorizer(
#             ngram_range=(1, 3),
#             stop_words='english',
#             max_features=5000,
#             vocabulary=self.job_specific_vocabulary
#         )

#     def _load_job_specific_vocabulary(self):
#         """Load expanded job-specific vocabulary"""
#         return {
#             # Achievement-oriented terms
#             'developed', 'implemented', 'managed', 'led', 'architected', 'designed', 
#             'optimized', 'reduced', 'increased', 'improved', 'accelerated', 'streamlined',
#             'enhanced', 'launched', 'delivered', 'coordinated', 'spearheaded', 'pioneered', 
#             'transformed', 'revamped', 'established', 'generated', 'secured', 'achieved',
            
#             # Technical context terms
#             'team', 'project', 'system', 'application', 'software', 'infrastructure', 
#             'platform', 'service', 'solution', 'database', 'architecture', 'framework',
#             'pipeline', 'protocol', 'algorithm', 'interface', 'network', 'environment',
#             'repository', 'codebase', 'workflow', 'automation', 'deployment', 'integration',
            
#             # Quality metrics
#             'performance', 'scalability', 'reliability', 'security', 'efficiency', 
#             'quality', 'usability', 'accessibility', 'productivity', 'velocity',
#             'throughput', 'latency', 'availability', 'maintainability', 'resilience',
            
#             # Business impact terms
#             'revenue', 'cost', 'roi', 'profit', 'savings', 'efficiency', 'growth',
#             'customer', 'client', 'stakeholder', 'user', 'experience', 'satisfaction',
#             'retention', 'engagement', 'conversion', 'acquisition', 'churn', 'metrics'
#         }

#     def analyze_resume_sections(self, resume_text):
#         """Identify and analyze different sections of the resume"""
#         # Split resume into lines and normalize whitespace
#         lines = [line.strip() for line in resume_text.split('\n') if line.strip()]
        
#         # Identify section boundaries
#         sections = {}
#         current_section = 'other'
#         section_content = []
        
#         for i, line in enumerate(lines):
#             # Check if this line is a section header
#             matched_section = None
#             for section, pattern in self.section_patterns.items():
#                 if pattern.match(line) and (i == 0 or len(line) < 50):  # Reasonable header length
#                     matched_section = section
#                     break
            
#             # If a new section is found, save the previous section content
#             if matched_section:
#                 if section_content:
#                     sections[current_section] = '\n'.join(section_content)
#                 current_section = matched_section
#                 section_content = []
#             else:
#                 section_content.append(line)
        
#         # Add the last section
#         if section_content:
#             sections[current_section] = '\n'.join(section_content)
        
#         return sections

#     def calculate_section_skill_scores(self, sections, job_skills):
#         """Calculate skill matches by section with weighted importance"""
#         section_scores = {}
#         total_weight = 0
#         total_weighted_score = 0
        
#         for section_name, content in sections.items():
#             section_skills = self.extract_technical_skills(content)
#             weight = self.section_weights.get(section_name, self.section_weights['other'])
            
#             # Calculate skill match for this section
#             section_score = 0
#             total_possible = 0
            
#             for category, skills in job_skills.items():
#                 if skills:
#                     matched = len(section_skills[category] & skills)
#                     total = len(skills)
#                     if total > 0:
#                         section_score += matched
#                         total_possible += total
            
#             if total_possible > 0:
#                 normalized_score = section_score / total_possible
#                 section_scores[section_name] = {
#                     'raw_score': normalized_score,
#                     'weighted_score': normalized_score * weight,
#                     'weight': weight,
#                     'matched_skills': {
#                         category: list(section_skills[category] & job_skills[category])
#                         for category in job_skills
#                     }
#                 }
                
#                 total_weight += weight
#                 total_weighted_score += normalized_score * weight
        
#         # Calculate overall weighted skill score
#         overall_skill_score = total_weighted_score / total_weight if total_weight > 0 else 0
        
#         return {
#             'section_scores': section_scores,
#             'overall_weighted_skill_score': overall_skill_score
#         }

#     def calculate_keyword_importance(self, job_description):
#         """Calculate importance of keywords in job description"""
#         doc = self.nlp(job_description)
        
#         # Extract sentences and calculate their embeddings
#         sentences = [sent.text for sent in doc.sents]
#         embeddings = self.sentence_model.encode(sentences)
        
#         # Calculate keyword frequency and position importance
#         keyword_scores = defaultdict(float)
#         terms = set()
        
#         for category, skills in self.tech_categories.items():
#             terms.update(skills)
        
#         # Add job-specific vocabulary
#         terms.update(self.job_specific_vocabulary)
        
#         # Calculate scores for each keyword
#         for term in terms:
#             # Find all occurrences
#             term_regex = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
#             matches = list(term_regex.finditer(job_description))
            
#             if matches:
#                 # Base score from frequency
#                 freq_score = len(matches) / len(doc)
                
#                 # Position score - earlier mentions may be more important
#                 positions = [match.start() / len(job_description) for match in matches]
#                 position_score = sum(1 - pos for pos in positions) / len(positions)
                
#                 # Context score - terms in requirement-heavy sentences are more important
#                 context_score = 0
#                 for match in matches:
#                     # Find which sentence contains this match
#                     for i, sent in enumerate(doc.sents):
#                         if sent.start_char <= match.start() < sent.end_char:
#                             # Check if sentence contains requirement language
#                             sent_text = sent.text.lower()
#                             if any(req in sent_text for req in ['required', 'must', 'need', 'essential']):
#                                 context_score += 1.5
#                             break
                
#                 # Combine scores
#                 keyword_scores[term] = freq_score + position_score + (context_score / len(matches))
        
#         return dict(keyword_scores)

#     def extract_technical_skills(self, text):
#         """Extract technical skills from text with improved pattern matching"""
#         skills = defaultdict(set)
#         text_lower = text.lower()
        
#         # Use lemmatization for better matching
#         doc = self.nlp(text_lower)
#         lemmatized_text = ' '.join([token.lemma_ for token in doc])
        
#         # Match skills with word boundaries
#         for category, terms in self.tech_categories.items():
#             for term in terms:
#                 # Check with word boundaries
#                 pattern = r'\b' + re.escape(term) + r'\b'
#                 if re.search(pattern, text_lower) or re.search(pattern, lemmatized_text):
#                     skills[category].add(term)
        
#         # Extract version information
#         version_pattern = r'\b(?:[a-zA-Z.]+(?:js)?(?:\d+(?:\.\d+)*)?)\b'
#         versions = re.findall(version_pattern, text_lower)
#         skills['versions'] = set(versions)
        
#         return skills

#     def score_resume(self, resume_text, job_description):
#         """Score resume with weighted section analysis and contextual importance"""
#         # Preprocess and clean the texts
#         resume_text = self._clean_text(resume_text)
#         job_description = self._clean_text(job_description)
        
#         # Extract required skills from job description
#         job_skills = self.extract_technical_skills(job_description)
        
#         # Calculate keyword importance
#         keyword_importance = self.calculate_keyword_importance(job_description)
        
#         # Analyze resume sections
#         resume_sections = self.analyze_resume_sections(resume_text)
        
#         # Calculate skill match by resume section
#         section_skill_analysis = self.calculate_section_skill_scores(resume_sections, job_skills)
        
#         # Get semantic similarity score
#         semantic_similarity = self.calculate_semantic_similarity(resume_text, job_description)
        
#         # Calculate keyword match with TF-IDF
#         processed = [
#             self._enhanced_preprocess(job_description),
#             self._enhanced_preprocess(resume_text)
#         ]
#         tfidf_matrix = self.tfidf_vectorizer.fit_transform(processed)
#         tfidf_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
#         # Get missing skills across all sections
#         resume_skills = self.extract_technical_skills(resume_text)
#         missing_skills = {
#             category: list(job_skills[category] - resume_skills[category])
#             for category in job_skills
#             if job_skills[category]
#         }
        
#         # Calculate overall score with improved weighting
#         overall_score = (
#             section_skill_analysis['overall_weighted_skill_score'] * 0.45 +
#             semantic_similarity * 0.35 +
#             tfidf_sim * 0.20
#         ) * 100
        
#         return {
#             'overall_score': round(overall_score, 2),
#             'score_breakdown': {
#                 'section_analysis': section_skill_analysis['section_scores'],
#                 'semantic_similarity': round(semantic_similarity * 100, 2),
#                 'keyword_match': round(tfidf_sim * 100, 2),
#                 'technical_match': round(section_skill_analysis['overall_weighted_skill_score'] * 100, 2)
#             },
#             'missing_skills': missing_skills,
#             'section_analysis': self._format_section_analysis(section_skill_analysis['section_scores']),
#             'feedback': self._generate_enhanced_feedback(
#                 section_skill_analysis, 
#                 semantic_similarity, 
#                 tfidf_sim, 
#                 missing_skills, 
#                 keyword_importance,
#                 resume_sections
#             )
#         }

#     def calculate_semantic_similarity(self, resume_text, job_description):
#         """Calculate semantic similarity using sentence transformers"""
#         # Encode texts to get embeddings
#         embeddings = self.sentence_model.encode([job_description, resume_text])
        
#         # Calculate cosine similarity
#         similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        
#         return similarity

#     def _clean_text(self, text):
#         """Clean and normalize text"""
#         # Replace multiple spaces and newlines with single spaces
#         text = re.sub(r'\s+', ' ', text)
        
#         # Remove URLs
#         text = re.sub(r'https?://\S+', '', text)
        
#         # Remove email addresses
#         text = re.sub(r'\S+@\S+', '', text)
        
#         # Remove phone numbers
#         text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '', text)
        
#         return text.strip()

#     def _enhanced_preprocess(self, text):
#         """Enhanced preprocessing for text analysis"""
#         text = text.lower()
        
#         # Preserve important characters in technical terms
#         text = re.sub(r'([./\\])', r' \1 ', text)
#         text = re.sub(r'([A-Za-z])(\d)', r'\1 \2', text)
        
#         # Process with spaCy for better lemmatization
#         doc = self.nlp(text)
#         tokens = []
        
#         for token in doc:
#             # Preserve technical terms
#             if any(token.text.lower() in terms for terms in self.tech_categories.values()):
#                 tokens.append(token.text.lower())
#             # Handle normal tokens
#             elif not token.is_stop and not token.is_punct and not token.is_space:
#                 tokens.append(token.lemma_)
                
#         return ' '.join(tokens)

#     def _format_section_analysis(self, section_scores):
#         """Format section analysis for better readability"""
#         formatted = {}
        
#         for section, data in section_scores.items():
#             formatted[section] = {
#                 'score': round(data['raw_score'] * 100, 2),
#                 'importance': data['weight'],
#                 'skills_found': sum(len(skills) for skills in data['matched_skills'].values())
#             }
            
#         return formatted

#     def _generate_enhanced_feedback(self, section_analysis, semantic_similarity, 
#                                    keyword_match, missing_skills, keyword_importance,
#                                    resume_sections):
#         """Generate detailed actionable feedback based on comprehensive analysis"""
#         feedback = []
        
#         # 1. Missing skills feedback with prioritization based on importance
#         if missing_skills:
#             feedback.append("🔧 **Critical Skill Gaps:**")
            
#             # Sort categories by importance
#             prioritized_skills = {}
#             for category, skills in missing_skills.items():
#                 if skills:
#                     # Get average importance of skills in this category
#                     avg_importance = sum(keyword_importance.get(skill, 0) for skill in skills) / len(skills)
#                     prioritized_skills[category] = {
#                         'skills': skills,
#                         'importance': avg_importance
#                     }
            
#             # Sort categories by importance
#             sorted_categories = sorted(
#                 prioritized_skills.keys(),
#                 key=lambda x: prioritized_skills[x]['importance'],
#                 reverse=True
#             )
            
#             for category in sorted_categories:
#                 skills = prioritized_skills[category]['skills']
#                 feedback.append(f"- {category.title()}: Missing {', '.join(skills)}")
        
#         # 2. Section-specific feedback
#         weak_sections = []
#         for section, data in section_analysis['section_scores'].items():
#             if data['raw_score'] < 0.6:  # Below 60% match
#                 weak_sections.append((section, data['raw_score']))
        
#         if weak_sections:
#             feedback.append("\n📊 **Section-Specific Improvements:**")
            
#             # Sort sections by lowest score first
#             sorted_sections = sorted(weak_sections, key=lambda x: x[1])
            
#             for section, score in sorted_sections:
#                 if section == 'work_experience':
#                     feedback.append(f"- **Work Experience**: Highlight relevant skills more explicitly in job descriptions (current match: {score*100:.1f}%)")
#                     feedback.append("  - Quantify achievements with metrics and specifics")
#                     feedback.append("  - Begin bullet points with strong action verbs")
#                     feedback.append("  - Explicitly mention tools and technologies used in each role")
                
#                 elif section == 'projects':
#                     feedback.append(f"- **Projects**: Enhance project descriptions with in-demand skills (current match: {score*100:.1f}%)")
#                     feedback.append("  - Focus on projects that showcase required skills from the job description")
#                     feedback.append("  - Highlight your role and specific technologies used")
                
#                 elif section == 'skills':
#                     feedback.append(f"- **Skills Section**: Update to better reflect job requirements (current match: {score*100:.1f}%)")
#                     feedback.append("  - Organize skills by category for better readability")
#                     feedback.append("  - Include versions/certifications where applicable")
                
#                 elif section == 'education':
#                     feedback.append(f"- **Education**: Enhance with relevant coursework/projects (current match: {score*100:.1f}%)")
                
#                 elif section == 'certifications':
#                     feedback.append(f"- **Certifications**: Consider adding relevant certifications (current match: {score*100:.1f}%)")
        
#         # 3. Context and semantic feedback
#         if semantic_similarity < 0.65:
#             feedback.append("\n📝 **Content Alignment Suggestions:**")
#             feedback.append("- Use more terminology and phrasing similar to the job posting")
#             feedback.append("- Match the tone and level of technical detail in the job description")
#             feedback.append("- Restructure resume to prioritize the most relevant experience first")
            
#             # Check for important missing sections
#             if 'work_experience' not in resume_sections:
#                 feedback.append("- Your resume may be missing a clearly labeled Work Experience section")
#             if 'skills' not in resume_sections:
#                 feedback.append("- Add a dedicated Skills section to improve ATS recognition")
        
#         # 4. Keyword optimization feedback
#         if keyword_match < 0.55:
#             feedback.append("\n🎯 **Keyword Optimization:**")
            
#             # Find most important keywords missing
#             resume_text = ' '.join(resume_sections.values()).lower()
#             important_missing = []
            
#             for term, importance in sorted(keyword_importance.items(), key=lambda x: x[1], reverse=True)[:15]:
#                 if importance > 0.5 and re.search(r'\b' + re.escape(term) + r'\b', resume_text) is None:
#                     important_missing.append(term)
            
#             if important_missing:
#                 feedback.append(f"- Consider adding these high-importance terms: {', '.join(important_missing[:5])}")
            
#             feedback.append("- Use exact phrases from the job description where applicable")
#             feedback.append("- Include industry-standard terminology and methodologies")
        
#         # 5. General ATS optimization tips
#         if not feedback or (semantic_similarity > 0.7 and section_analysis['overall_weighted_skill_score'] > 0.7):
#             feedback.append("✅ **Strong Overall Match!**")
#             feedback.append("💪 Your resume aligns well with the job requirements")
#             feedback.append("\n📈 **Fine-tuning Suggestions:**")
#         else:
#             feedback.append("\n🤖 **ATS Optimization Tips:**")
        
#         feedback.append("- Use standard section headings that ATS systems can easily recognize")
#         feedback.append("- Avoid tables, graphics, headers/footers, and unusual formatting")
#         feedback.append("- Save your resume in a standard format (PDF from Word is usually best)")
#         feedback.append("- Place most important information in top third of first page")
        
#         return feedback

# class MistralAIATSScorer:
#     """Class for future Mistral AI implementation"""
#     def __init__(self):
#         # Placeholder for future implementation
#         self.base_scorer = EnhancedATSScorer()
    
#     def score_resume(self, resume_text, job_description):
#         """Score resume using combined approach with base scorer and Mistral AI"""
#         # For now, just use the base scorer
#         base_results = self.base_scorer.score_resume(resume_text, job_description)
        
#         # This is where we would integrate Mistral AI capabilities
#         # For example, using it for more nuanced language understanding
#         # or to provide more specific, actionable feedback
        
#         return base_results


# import spacy
# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# from sentence_transformers import SentenceTransformer
# from transformers import AutoTokenizer, AutoModel
# import torch
# import re
# from collections import defaultdict

# class ATSScorer:
#     def __init__(self):
#         """Initialize with models including DistilBERT"""
#         try:
#             self.nlp = spacy.load('en_core_web_sm')
#         except:
#             self.nlp = spacy.blank("en")
        
#         # Initialize models as None first
#         self.resume_tokenizer = None
#         self.resume_model = None
#         self.skill_model = None
        
#         # Initialize models in a way that's compatible with Streamlit's caching
#         self._initialize_models()
        
#         # TF-IDF for keyword matching
#         self.tfidf_vectorizer = TfidfVectorizer(
#             ngram_range=(1, 3),
#             stop_words='english',
#             max_features=3000,
#             vocabulary=self.load_job_specific_vocabulary()
#         )
        
#         # Technical skill categories
#         self.tech_categories = {
#             'programming_languages': set(['python', 'java', 'javascript', 'c++', 'golang', 'rust']),
#             'frameworks': set(['react', 'angular', 'vue', 'django', 'flask', 'spring']),
#             'cloud': set(['aws', 'azure', 'gcp', 'kubernetes', 'docker']),
#             'databases': set(['sql', 'mongodb', 'postgresql', 'mysql', 'redis']),
#             'tools': set(['git', 'jenkins', 'jira', 'terraform', 'prometheus'])
#         }

#     def _initialize_models(self):
#         """Initialize models separately to avoid PyTorch class registration issues"""
#         if self.resume_tokenizer is None:
#             self.resume_tokenizer = AutoTokenizer.from_pretrained(
#                 "MNG123/msmarco-distilbert-base-tas-b-resume-fit-v2-epoch-3",
#                 local_files_only=False
#             )
        
#         if self.resume_model is None:
#             self.resume_model = AutoModel.from_pretrained(
#                 "MNG123/msmarco-distilbert-base-tas-b-resume-fit-v2-epoch-3",
#                 local_files_only=False
#             )
#             self.resume_model.eval()
        
#         if self.skill_model is None:
#             self.skill_model = SentenceTransformer('paraphrase-albert-small-v2')

#     def load_job_specific_vocabulary(self):
#         """Load curated job-specific vocabulary"""
#         return {
#             'developed', 'implemented', 'managed', 'led', 'architected',
#             'designed', 'optimized', 'reduced', 'increased', 'improved',
#             'team', 'project', 'system', 'application', 'software',
#             'infrastructure', 'platform', 'service', 'solution', 'database',
#             'performance', 'scalability', 'reliability', 'security', 'deployment'
#         }

#     def mean_pooling(self, model_output, attention_mask):
#         """Perform mean pooling on the token embeddings"""
#         token_embeddings = model_output[0]
#         input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
#         return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

#     def get_distilbert_embeddings(self, texts):
#         """Get embeddings using the resume-specific DistilBERT model"""
#         # Tokenize sentences
#         encoded_input = self.resume_tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=512)
        
#         # Compute token embeddings
#         with torch.no_grad():
#             model_output = self.resume_model(**encoded_input)
        
#         # Perform pooling
#         sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        
#         # Normalize embeddings
#         sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
#         return sentence_embeddings

#     def calculate_hybrid_similarity(self, resume_text, job_description):
#         """Calculate similarities using DistilBERT and other models"""
#         # DistilBERT similarity for resume-specific matching
#         resume_embeddings = self.get_distilbert_embeddings([job_description, resume_text])
#         distilbert_sim = cosine_similarity(
#             resume_embeddings[0].reshape(1, -1).numpy(),
#             resume_embeddings[1].reshape(1, -1).numpy()
#         )[0][0]
        
#         # Skill similarity using ALBERT
#         skill_embeddings = self.skill_model.encode([job_description, resume_text])
#         skill_sim = cosine_similarity([skill_embeddings[0]], [skill_embeddings[1]])[0][0]
        
#         # TF-IDF similarity
#         processed = [
#             self.enhanced_preprocess(job_description),
#             self.enhanced_preprocess(resume_text)
#         ]
#         tfidf_matrix = self.tfidf_vectorizer.fit_transform(processed)
#         tfidf_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
#         # Technical skill match
#         tech_sim = self.calculate_skill_match(resume_text, job_description)
        
#         return {
#             'distilbert_sim': distilbert_sim,
#             'skill_sim': skill_sim,
#             'tfidf_sim': tfidf_sim,
#             'tech_sim': tech_sim
#         }

#     def score_resume(self, resume_text, job_description):
#         """Score resume with updated weights for DistilBERT"""
#         similarities = self.calculate_hybrid_similarity(resume_text, job_description)
        
#         # Extract skills
#         job_skills = self.extract_technical_skills(job_description)
#         resume_skills = self.extract_technical_skills(resume_text)
        
#         # Calculate missing skills
#         missing_skills = {
#             category: list(job_skills[category] - resume_skills[category])
#             for category in job_skills
#             if job_skills[category]
#         }
        
#         # Updated weights to emphasize DistilBERT's resume-specific matching
#         final_score = (
#             similarities['distilbert_sim'] * 0.4 +  # Increased weight for resume-specific matching
#             similarities['skill_sim'] * 0.25 +
#             similarities['tfidf_sim'] * 0.15 +
#             similarities['tech_sim'] * 0.2
#         ) * 100
        
#         return {
#             'overall_score': round(final_score, 2),
#             'score_breakdown': {
#                 'content_similarity': round(similarities['distilbert_sim'] * 100, 2),
#                 'technical_match': round(similarities['tech_sim'] * 100, 2),
#                 'context_understanding': round(similarities['skill_sim'] * 100, 2)
#             },
#             'missing_skills': missing_skills,
#             'feedback': self.generate_feedback(similarities, missing_skills)
#         }

#     def calculate_skill_match(self, resume_text, job_description):
#         """Calculate technical skill match score"""
#         job_skills = self.extract_technical_skills(job_description)
#         resume_skills = self.extract_technical_skills(resume_text)
        
#         skill_match_scores = []
#         for category in job_skills:
#             if job_skills[category]:
#                 match_ratio = len(resume_skills[category] & job_skills[category]) / len(job_skills[category])
#                 skill_match_scores.append(match_ratio)
        
#         return np.mean(skill_match_scores) if skill_match_scores else 0

#     def generate_feedback(self, similarities, missing_skills):
#         """Generate actionable feedback"""
#         feedback = []
        
#         # Technical skill feedback
#         if similarities['tech_sim'] < 0.7:
#             feedback.append("🔧 Technical Skill Gaps:")
#             for category, skills in missing_skills.items():
#                 if skills:
#                     feedback.append(f"- {category.title()}: Missing {', '.join(skills)}")
        
#         # Content match feedback
#         if similarities['distilbert_sim'] < 0.7:
#             feedback.append("📝 Content Improvement Suggestions:")
#             feedback.append("- Add more specific details about your role and responsibilities")
#             feedback.append("- Include quantifiable achievements and metrics")
#             feedback.append("- Highlight relevant projects and technologies used")
        
#         # Keyword match feedback
#         if similarities['tfidf_sim'] < 0.65:
#             feedback.append("🎯 Keyword Alignment:")
#             feedback.append("- Align your terminology with the job description")
#             feedback.append("- Include industry-standard terms and tools")
#             feedback.append("- Mention specific methodologies used")
        
#         if not feedback:
#             feedback.append("✅ Strong Match!")
#             feedback.append("💪 Your resume aligns well with the job requirements")
        
#         return feedback

#     def enhanced_preprocess(self, text):
#         """Preprocess text for analysis"""
#         text = text.lower()
#         text = re.sub(r'([./\\])', r' \1 ', text)
#         text = re.sub(r'([A-Za-z])(\d)', r'\1 \2', text)
        
#         doc = self.nlp(text)
#         tokens = []
        
#         for token in doc:
#             if any(token.text in terms for terms in self.tech_categories.values()):
#                 tokens.append(token.text)
#             elif not token.is_stop and not token.is_punct and not token.is_space:
#                 tokens.append(token.lemma_)
                
#         return ' '.join(tokens)
    
#     def extract_technical_skills(self, text):
#         """Extract technical skills from text"""
#         skills = defaultdict(set)
#         text_lower = text.lower()
        
#         for category, terms in self.tech_categories.items():
#             for term in terms:
#                 if term in text_lower:
#                     skills[category].add(term)
        
#         version_pattern = r'\b(?:[a-zA-Z.]+(?:js)?(?:\d+(?:\.\d+)*)?)\b'
#         versions = re.findall(version_pattern, text)
#         skills['versions'] = set(versions)
        
#         return skills
    
