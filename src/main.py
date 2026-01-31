# ML-Based Resume Ranking System
# Complete implementation from data generation to deployment

import pandas as pd
import numpy as np
import re
import random
import nltk
nltk.download('punkt_tab')

from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

print("=== ML-Based Resume Ranking System ===\n")

# ==========================================
# 1. PROBLEM DEFINITION
# ==========================================

print("1. PROBLEM DEFINITION")
print("=" * 50)
print("""
Objective: Build an ML system that ranks resumes by relevance to job descriptions.

Key Components:
- Input: Job description + Collection of resumes
- Output: Ranked list of resumes with relevance scores
- Method: Use text similarity and ML classification/regression
- Evaluation: Precision@K, MRR, and classification metrics
""")

# ==========================================
# 2. HYBRID DATASET CREATION
# ==========================================

print("\n2. DATASET CREATION")
print("=" * 50)

# Sample structure for Kaggle resume dataset (simulate loading)
def simulate_kaggle_data():
    """Simulate loading Kaggle resume dataset with realistic structure"""
    kaggle_resumes = [
        {
            'name': 'John Smith',
            'skills': 'Python, Machine Learning, Data Analysis, SQL, TensorFlow',
            'experience': 'Data Scientist at Tech Corp (2021-2023), Analyst at StartupXYZ (2019-2021)',
            'education': 'MS in Computer Science, BS in Mathematics',
            'resume_text': 'Data Scientist with 4 years experience in machine learning and data analysis. Proficient in Python, SQL, and TensorFlow. Built recommendation systems and predictive models.'
        },
        {
            'name': 'Sarah Johnson',
            'skills': 'Java, Spring Boot, Microservices, REST API, Docker',
            'experience': 'Senior Software Engineer at BigTech (2020-2023), Developer at MidCorp (2018-2020)',
            'education': 'BS in Computer Science',
            'resume_text': 'Senior Software Engineer with 5 years experience in backend development. Expert in Java, Spring Boot, and microservices architecture. Led team of 4 developers.'
        },
        {
            'name': 'Mike Chen',
            'skills': 'React, JavaScript, CSS, Node.js, MongoDB',
            'experience': 'Frontend Developer at WebCorp (2021-2023), Junior Developer at StartupABC (2020-2021)',
            'education': 'BS in Information Technology',
            'resume_text': 'Frontend Developer with 3 years experience in React and JavaScript. Built responsive web applications and improved user experience metrics by 25%.'
        }
    ]
    return pd.DataFrame(kaggle_resumes)

# Generate synthetic resumes
def generate_synthetic_resumes(n=25):
    """Generate realistic synthetic resumes"""
    
    skills_pool = {
        'data_science': ['Python', 'R', 'Machine Learning', 'Deep Learning', 'TensorFlow', 'PyTorch', 'SQL', 'Pandas', 'NumPy', 'Scikit-learn', 'Statistics', 'Data Visualization'],
        'software_dev': ['Java', 'Python', 'JavaScript', 'C++', 'React', 'Angular', 'Node.js', 'Spring Boot', 'Docker', 'Kubernetes', 'Git', 'REST API'],
        'devops': ['AWS', 'Docker', 'Kubernetes', 'Jenkins', 'Terraform', 'Linux', 'Shell Scripting', 'CI/CD', 'Monitoring', 'Cloud Computing'],
        'product': ['Product Management', 'Agile', 'Scrum', 'User Research', 'A/B Testing', 'Analytics', 'Roadmap Planning', 'Stakeholder Management'],
        'marketing': ['Digital Marketing', 'SEO', 'SEM', 'Google Analytics', 'Content Marketing', 'Social Media', 'Email Marketing', 'Marketing Automation']
    }
    
    names = ['Alex Kumar', 'Jessica Wong', 'David Martinez', 'Emma Thompson', 'Raj Patel', 
             'Lisa Zhang', 'Carlos Rodriguez', 'Priya Sharma', 'James Wilson', 'Sophia Lee',
             'Ahmed Hassan', 'Maya Singh', 'Robert Taylor', 'Aisha Khan', 'Kevin O\'Brien']
    
    companies = ['TechCorp', 'DataSoft', 'CloudWorks', 'InnovateLab', 'FutureTech', 
                 'StartupXYZ', 'MegaCorp', 'DigitalSolutions', 'SmartSystems', 'NextGen']
    
    roles = {
        'data_science': ['Data Scientist', 'ML Engineer', 'Data Analyst', 'Research Scientist'],
        'software_dev': ['Software Engineer', 'Full Stack Developer', 'Backend Developer', 'Senior Developer'],
        'devops': ['DevOps Engineer', 'Cloud Engineer', 'Site Reliability Engineer', 'Infrastructure Engineer'],
        'product': ['Product Manager', 'Senior Product Manager', 'Product Owner', 'Product Analyst'],
        'marketing': ['Marketing Manager', 'Digital Marketing Specialist', 'Growth Manager', 'Marketing Analyst']
    }
    
    educations = ['BS in Computer Science', 'MS in Data Science', 'BS in Engineering', 
                  'MBA', 'MS in Computer Science', 'BS in Mathematics', 'PhD in Computer Science']
    
    synthetic_resumes = []
    
    for i in range(n):
        category = random.choice(list(skills_pool.keys()))
        selected_skills = random.sample(skills_pool[category], random.randint(4, 8))
        name = random.choice(names) + f" {i+1}"  # Ensure unique names
        company = random.choice(companies)
        role = random.choice(roles[category])
        education = random.choice(educations)
        years_exp = random.randint(1, 8)
        
        resume_text = f"{role} with {years_exp} years of experience. Skilled in {', '.join(selected_skills[:3])}. " \
                     f"Previously worked at {company} developing solutions and improving processes. " \
                     f"Strong background in {selected_skills[-1].lower()} and team collaboration."
        
        synthetic_resumes.append({
            'name': name,
            'skills': ', '.join(selected_skills),
            'experience': f"{role} at {company} ({2024-years_exp}-2023)",
            'education': education,
            'resume_text': resume_text
        })
    
    return pd.DataFrame(synthetic_resumes)

# Generate synthetic job descriptions
def generate_job_descriptions(n=8):
    """Generate realistic job descriptions"""
    
    job_descriptions = [
        {
            'title': 'Senior Data Scientist',
            'company': 'TechCorp',
            'description': 'We are seeking a Senior Data Scientist with expertise in machine learning and statistical analysis. Must have experience with Python, TensorFlow, and large datasets. Will build predictive models and work with cross-functional teams.',
            'requirements': 'Python, Machine Learning, TensorFlow, Statistics, SQL, 3+ years experience'
        },
        {
            'title': 'Full Stack Developer',
            'company': 'WebSolutions',
            'description': 'Looking for a Full Stack Developer to build modern web applications. Need strong skills in React, Node.js, and database design. Will work on both frontend and backend systems.',
            'requirements': 'React, JavaScript, Node.js, MongoDB, REST API, 2+ years experience'
        },
        {
            'title': 'DevOps Engineer',
            'company': 'CloudFirst',
            'description': 'DevOps Engineer needed to manage cloud infrastructure and CI/CD pipelines. Must have AWS experience and containerization knowledge. Will automate deployment processes.',
            'requirements': 'AWS, Docker, Kubernetes, CI/CD, Linux, Infrastructure as Code'
        },
        {
            'title': 'Product Manager',
            'company': 'InnovateCorp',
            'description': 'Product Manager to drive product strategy and roadmap. Need experience with agile methodologies and user research. Will work with engineering and design teams.',
            'requirements': 'Product Management, Agile, User Research, Analytics, Communication skills'
        },
        {
            'title': 'Machine Learning Engineer',
            'company': 'AI Startup',
            'description': 'ML Engineer to deploy machine learning models in production. Need experience with MLOps, model monitoring, and scalable systems. Will optimize model performance.',
            'requirements': 'Python, PyTorch, MLOps, Model Deployment, Docker, Cloud platforms'
        },
        {
            'title': 'Frontend Developer',
            'company': 'UX Design Co',
            'description': 'Frontend Developer specializing in React applications. Must create responsive, accessible user interfaces. Will collaborate closely with design team.',
            'requirements': 'React, JavaScript, CSS, HTML, Responsive Design, Git'
        },
        {
            'title': 'Data Analyst',
            'company': 'Analytics Plus',
            'description': 'Data Analyst to extract insights from business data. Need SQL expertise and data visualization skills. Will create dashboards and reports for stakeholders.',
            'requirements': 'SQL, Excel, Data Visualization, Statistics, Business Intelligence'
        },
        {
            'title': 'Backend Developer',
            'company': 'ServerTech',
            'description': 'Backend Developer to build scalable APIs and services. Need experience with Java, Spring Boot, and microservices. Will design database schemas and optimize performance.',
            'requirements': 'Java, Spring Boot, Microservices, REST API, Database Design, Performance Optimization'
        }
    ]
    
    return pd.DataFrame(job_descriptions[:n])

# Load/create datasets
print("Creating hybrid dataset...")
kaggle_df = simulate_kaggle_data()
synthetic_df = generate_synthetic_resumes(25)
job_df = generate_job_descriptions(8)

# Combine resume datasets
all_resumes = pd.concat([kaggle_df, synthetic_df], ignore_index=True)
print(f"Total resumes: {len(all_resumes)}")
print(f"Total job descriptions: {len(job_df)}")

# ==========================================
# 3. TEXT PREPROCESSING
# ==========================================

print("\n3. TEXT PREPROCESSING")
print("=" * 50)

class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
    
    def clean_text(self, text):
        """Clean and normalize text"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def tokenize_and_lemmatize(self, text):
        """Tokenize and lemmatize text"""
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        return ' '.join(tokens)
    
    def preprocess(self, text):
        """Complete preprocessing pipeline"""
        cleaned = self.clean_text(text)
        processed = self.tokenize_and_lemmatize(cleaned)
        return processed

# Initialize preprocessor
preprocessor = TextPreprocessor()

# Preprocess resume and job description texts
print("Preprocessing resume texts...")
all_resumes['processed_text'] = all_resumes['resume_text'].apply(preprocessor.preprocess)

print("Preprocessing job descriptions...")
job_df['processed_description'] = job_df['description'].apply(preprocessor.preprocess)
job_df['processed_requirements'] = job_df['requirements'].apply(preprocessor.preprocess)
job_df['combined_job_text'] = job_df['processed_description'] + ' ' + job_df['processed_requirements']

# ==========================================
# 4. CREATE LABELED TRAINING DATA
# ==========================================

print("\n4. CREATING LABELED TRAINING DATA")
print("=" * 50)

def calculate_similarity_score(resume_text, job_text):
    """Calculate similarity between resume and job description using TF-IDF"""
    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform([resume_text, job_text])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return similarity

def create_training_pairs():
    """Create (resume, job, relevance_score) training pairs"""
    training_data = []
    
    for _, job in job_df.iterrows():
        job_text = job['combined_job_text']
        
        for _, resume in all_resumes.iterrows():
            resume_text = resume['processed_text']
            
            # Calculate similarity score
            similarity = calculate_similarity_score(resume_text, job_text)
            
            # Create relevance categories based on similarity
            if similarity >= 0.3:
                relevance = 2  # Highly relevant
            elif similarity >= 0.15:
                relevance = 1  # Moderately relevant
            else:
                relevance = 0  # Not relevant
            
            training_data.append({
                'resume_id': resume.name,
                'job_id': job.name,
                'resume_text': resume_text,
                'job_text': job_text,
                'similarity_score': similarity,
                'relevance_label': relevance,
                'resume_name': resume['name'],
                'job_title': job['title']
            })
    
    return pd.DataFrame(training_data)

# Create training dataset
print("Generating training pairs...")
training_df = create_training_pairs()
print(f"Created {len(training_df)} training pairs")
print(f"Relevance distribution:\n{training_df['relevance_label'].value_counts()}")

# ==========================================
# 5. FEATURE EXTRACTION
# ==========================================

print("\n5. FEATURE EXTRACTION")
print("=" * 50)

class FeatureExtractor:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            stop_words='english'
        )
        self.scaler = StandardScaler()
        
    def fit_transform(self, resume_texts, job_texts):
        """Fit vectorizer and extract features"""
        # Combine all texts for vocabulary
        all_texts = list(resume_texts) + list(job_texts)
        self.tfidf_vectorizer.fit(all_texts)
        
        # Transform texts
        resume_vectors = self.tfidf_vectorizer.transform(resume_texts)
        job_vectors = self.tfidf_vectorizer.transform(job_texts)
        
        # Create feature matrix
        features = []
        for i in range(len(resume_texts)):
            # TF-IDF similarity
            similarity = cosine_similarity(resume_vectors[i], job_vectors[i])[0][0]
            
            # Text length features
            resume_len = len(resume_texts.iloc[i].split())
            job_len = len(job_texts.iloc[i].split())
            
            # Common words count
            resume_words = set(resume_texts.iloc[i].split())
            job_words = set(job_texts.iloc[i].split())
            common_words = len(resume_words.intersection(job_words))
            
            features.append([
                similarity,
                resume_len,
                job_len,
                common_words,
                len(resume_words),
                len(job_words)
            ])
        
        features = np.array(features)
        features = self.scaler.fit_transform(features)
        return features
    
    def transform(self, resume_texts, job_texts):
        """Transform new data using fitted vectorizer"""
        resume_vectors = self.tfidf_vectorizer.transform(resume_texts)
        job_vectors = self.tfidf_vectorizer.transform(job_texts)
        
        features = []
        for i in range(len(resume_texts)):
            similarity = cosine_similarity(resume_vectors[i], job_vectors[i])[0][0]
            resume_len = len(resume_texts.iloc[i].split())
            job_len = len(job_texts.iloc[i].split())
            
            resume_words = set(resume_texts.iloc[i].split())
            job_words = set(job_texts.iloc[i].split())
            common_words = len(resume_words.intersection(job_words))
            
            features.append([
                similarity,
                resume_len,
                job_len,
                common_words,
                len(resume_words),
                len(job_words)
            ])
        
        features = np.array(features)
        features = self.scaler.transform(features)
        return features

# Extract features
feature_extractor = FeatureExtractor()
X = feature_extractor.fit_transform(training_df['resume_text'], training_df['job_text'])
y = training_df['relevance_label'].values

print(f"Feature matrix shape: {X.shape}")
print(f"Target distribution: {np.bincount(y)}")

# ==========================================
# 6. MODEL TRAINING
# ==========================================

print("\n6. MODEL TRAINING")
print("=" * 50)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42  # Removed stratify=y
)

# Train multiple models
models = {
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
}

trained_models = {}
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    
    # Predictions
    if name == 'Random Forest':
        y_pred = model.predict(X_test)
        # Convert regression predictions to classes
        y_pred_class = np.round(np.clip(y_pred, 0, 2)).astype(int)
        mse = mean_squared_error(y_test, y_pred)
        print(f"{name} - MSE: {mse:.4f}")
    else:
        y_pred_class = model.predict(X_test)
        y_pred = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred_class
    
    # Classification metrics
    accuracy = accuracy_score(y_test, y_pred_class)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred_class, average='weighted')
    
    print(f"{name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    trained_models[name] = model

# Select best model (Random Forest for this example)
best_model = trained_models['Random Forest']

# ==========================================
# 7. RESUME RANKING SYSTEM
# ==========================================

print("\n7. RESUME RANKING SYSTEM")
print("=" * 50)

class ResumeRankingSystem:
    def __init__(self, model, feature_extractor, resumes_df, preprocessor):
        self.model = model
        self.feature_extractor = feature_extractor
        self.resumes_df = resumes_df
        self.preprocessor = preprocessor
    
    def rank_resumes(self, job_description, top_k=5):
        """Rank resumes for a given job description"""
        # Preprocess job description
        processed_job = self.preprocessor.preprocess(job_description)
        
        # Create temporary dataframe for feature extraction
        temp_df = pd.DataFrame({
            'resume_text': self.resumes_df['processed_text'],
            'job_text': [processed_job] * len(self.resumes_df)
        })
        
        # Extract features
        features = self.feature_extractor.transform(
            temp_df['resume_text'],
            temp_df['job_text']
        )
        
        # Predict relevance scores
        relevance_scores = self.model.predict(features)
        
        # Create results dataframe
        results = pd.DataFrame({
            'name': self.resumes_df['name'],
            'resume_text': self.resumes_df['resume_text'],
            'skills': self.resumes_df['skills'],
            'relevance_score': relevance_scores
        })
        
        # Sort by relevance score
        results = results.sort_values('relevance_score', ascending=False)
        
        return results.head(top_k)
    
    def evaluate_ranking(self, test_pairs, k=5):
        """Evaluate ranking performance using Precision@K"""
        precisions_at_k = []
        
        # Group test pairs by job
        for job_id in test_pairs['job_id'].unique():
            job_pairs = test_pairs[test_pairs['job_id'] == job_id]
            job_text = job_pairs.iloc[0]['job_text']
            
            # Get rankings
            rankings = self.rank_resumes(job_text, top_k=k)
            
            # Check how many in top-k are actually relevant
            relevant_in_topk = 0
            for _, resume in rankings.iterrows():
                resume_pairs = job_pairs[job_pairs['resume_name'] == resume['name']]
                if not resume_pairs.empty and resume_pairs.iloc[0]['relevance_label'] >= 1:
                    relevant_in_topk += 1
            
            precision_at_k = relevant_in_topk / k
            precisions_at_k.append(precision_at_k)
        
        return np.mean(precisions_at_k)

# Initialize ranking system
ranking_system = ResumeRankingSystem(
    best_model, feature_extractor, all_resumes, preprocessor
)

# ==========================================
# 8. DEMONSTRATION
# ==========================================

print("\n8. SYSTEM DEMONSTRATION")
print("=" * 50)

# Test with sample job descriptions
test_jobs = [
    "Looking for a Data Scientist with Python and machine learning experience. Must know TensorFlow and statistics.",
    "Need a Full Stack Developer proficient in React, Node.js, and JavaScript. Database experience required.",
    "Seeking DevOps Engineer with AWS, Docker, and Kubernetes expertise. CI/CD pipeline experience preferred."
]

for i, job_desc in enumerate(test_jobs, 1):
    print(f"\nTest Job {i}: {job_desc[:80]}...")
    print("-" * 60)
    
    rankings = ranking_system.rank_resumes(job_desc, top_k=3)
    
    for rank, (_, resume) in enumerate(rankings.iterrows(), 1):
        print(f"{rank}. {resume['name']} (Score: {resume['relevance_score']:.3f})")
        print(f"   Skills: {resume['skills'][:100]}...")
        print()

# ==========================================
# 9. MODEL EVALUATION
# ==========================================

print("\n9. MODEL EVALUATION")
print("=" * 50)

# Create test set from training data
test_indices = training_df.sample(n=min(50, len(training_df)), random_state=42).index
test_pairs = training_df.loc[test_indices]

# Calculate Precision@K
precision_at_3 = ranking_system.evaluate_ranking(test_pairs, k=3)
precision_at_5 = ranking_system.evaluate_ranking(test_pairs, k=5)

print(f"Precision@3: {precision_at_3:.4f}")
print(f"Precision@5: {precision_at_5:.4f}")

# ==========================================
# 10. DEPLOYMENT READY FUNCTIONS
# ==========================================

print("\n10. DEPLOYMENT FUNCTIONS")
print("=" * 50)

def save_model_components():
    """Save model components for deployment"""
    import pickle
    
    components = {
        'model': best_model,
        'feature_extractor': feature_extractor,
        'preprocessor': preprocessor,
        'resumes_df': all_resumes
    }
    
    with open('resume_ranking_model.pkl', 'wb') as f:
        pickle.dump(components, f)
    
    print("Model components saved to 'resume_ranking_model.pkl'")

def load_model_components():
    """Load model components for deployment"""
    import pickle
    
    with open('resume_ranking_model.pkl', 'rb') as f:
        components = pickle.load(f)
    
    return ResumeRankingSystem(
        components['model'],
        components['feature_extractor'],
        components['resumes_df'],
        components['preprocessor']
    )

# Save model for deployment
save_model_components()

print("\n" + "="*60)
print("RESUME RANKING SYSTEM COMPLETE!")
print("="*60)
print("""
Next Steps:
1. Run this script to train the model
2. Use the Streamlit app (separate file) for UI
3. Deploy to Streamlit Cloud, Render, or Hugging Face Spaces

Key Features Implemented:
✓ Hybrid dataset (Kaggle simulation + synthetic data)
✓ Text preprocessing with NLTK
✓ TF-IDF feature extraction
✓ Multiple ML models (Random Forest, Logistic Regression)
✓ Resume ranking system
✓ Model evaluation with Precision@K
✓ Deployment-ready functions
""")