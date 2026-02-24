import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class RAG:

    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.docs = []
        self.index = None
        self.cache = {}

    def build(self):

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(BASE_DIR, "sample_data", "hr_docs.txt")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")

        with open(path, encoding="utf-8") as f:
            self.docs = [line.strip() for line in f if line.strip()]

        embeddings = self.model.encode(self.docs)

        dim = len(embeddings[0])
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(embeddings).astype("float32"))

    def retrieve(self, query, k=3):

        if query in self.cache:
            return self.cache[query]

        if not self.index:
            return []

        q = self.model.encode([query]).astype("float32")
        _, indexes = self.index.search(q, k)

        results = [self.docs[i] for i in indexes[0]]

        self.cache[query] = results
        return results


rag_index = RAG()