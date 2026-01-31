# src/main.py
import pickle
from model_utils import TextPreprocessor, FeatureExtractor
from sklearn.metrics.pairwise import cosine_similarity

class ResumeRankingSystem:
    def __init__(self, model_path):
        # Load pickled model
        with open(model_path, "rb") as f:
            data = pickle.load(f)

        self.extractor = data["extractor"]
        self.resumes = data["resumes"]
        self.preprocessor = data["preprocessor"]

    def rank_resumes(self, job_description, top_k=5):
        job_desc = self.preprocessor.preprocess(job_description)

        resume_texts = [
            self.preprocessor.preprocess(r["text"])
            for r in self.resumes
        ]

        job_vec, resume_vecs = self.extractor.transform(job_desc, resume_texts)

        scores = cosine_similarity(job_vec, resume_vecs)[0]

        ranked = sorted(
            zip(self.resumes, scores),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        return ranked


def train_and_save():
    # Example resumes
    resumes = [
        {"name": "John Smith", "skills": ["Python", "Machine Learning"], "text": "Python ML data science experience"},
        {"name": "Aisha Khan", "skills": ["AWS", "Docker", "DevOps"], "text": "AWS Docker Kubernetes DevOps engineer"},
        {"name": "Mike Chen", "skills": ["React", "Node.js"], "text": "Full stack developer React Node.js"}
    ]

    data = {
        "extractor": FeatureExtractor(),
        "preprocessor": TextPreprocessor(),
        "resumes": resumes
    }

    with open("resume_ranking_model.pkl", "wb") as f:
        pickle.dump(data, f)

    print("✅ Model saved as resume_ranking_model.pkl")


if __name__ == "__main__":
    train_and_save()
