# rag.py
import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path

class SalonRAG:
    def __init__(self, faq_path: Path):
        if not faq_path.exists():
            raise FileNotFoundError(f"FAQ file not found: {faq_path}")
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.client.get_or_create_collection("salon_faq")
        self.model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        self._index(faq_path)
    def _index(self, faq_path):
        # Split faq.md into chunks by "## "
        text = faq_path.read_text(encoding="utf-8")
        chunks = [c.strip() for c in text.split("## ") if c.strip()]
        for i, chunk in enumerate(chunks):
            emb = self.model.encode(chunk).tolist()
            self.collection.add(ids=[f"faq_{i}"], embeddings=[emb], documents=[chunk])
    
    def query(self, question: str, n: int = 2) -> str:
        emb = self.model.encode(question).tolist()
        results = self.collection.query(query_embeddings=[emb], n_results=n)
        return "\n---\n".join(results["documents"][0])