# src/model_utils.py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import nltk

nltk.download("punkt")
nltk.download("wordnet")


class TextPreprocessor:
    def preprocess(self, text):
        if not text:
            return ""
        return text.lower().strip()


class FeatureExtractor:
    def transform(self, job_desc, resumes):
        """Dummy vectorizer: replace with TF-IDF if you want"""
        job_vec = np.random.rand(1, 10)
        resume_vecs = np.random.rand(len(resumes), 10)
        return job_vec, resume_vecs
