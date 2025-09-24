import os
import json
import numpy as np
import concurrent.futures
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from config import CONFIG  # load settings from config.py

# ----------------- RAG SYSTEM -----------------
class MedicalRAGSystem:
    def __init__(self):
        """
        Initialize the RAG system:
        - Load cached docs and embeddings from JSON 
        - Set up the embedding model
        - Initialize vector storage
        """
        # Load settings from config
        self.chunk_size = CONFIG.get("rag_chunk_size", 400)
        self.top_k = CONFIG.get("rag_top_k", 3)
        self.embedding_model_name = CONFIG.get(
            "embedding_model", "ibm-granite/granite-embedding-small-english-r2"
        )
        
        # Initialize paths - ensure vector_db_path exists
        self.vector_db_path = CONFIG.get("vector_db_path", "./medical_knowledge_db")
        if not os.path.exists(self.vector_db_path):
            os.makedirs(self.vector_db_path)
            
        self.cache_file = os.path.join(self.vector_db_path, "cache.json")
        self.index_file = os.path.join(self.vector_db_path, "index.json")

        # Initialize embedding model
        self.model = SentenceTransformer(self.embedding_model_name)
        
        # Load cache (docs + embeddings)
        self.cache = self._load_cache()
        self.docs = self.cache.get("docs", [])
        self.embeddings = self.cache.get("embeddings", {})
        
        # Load or initialize index
        self.index = self._load_index()
        
        # Thread pool for async operations
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

    def _load_cache(self) -> Dict[str, Any]:
        """
        Loads docs + embeddings from cache.json
        Returns a dict, never a tuple.
        """
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"[!] Failed to load cache: {e}")
                return {"docs": [], "embeddings": {}}
        return {"docs": [], "embeddings": {}}

    def _save_cache(self) -> None:
        """Save docs and embeddings to cache.json"""
        try:
            self.cache.update({
                "docs": self.docs,
                "embeddings": {k: v.tolist() for k, v in self.embeddings.items()}
            })
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[!] Failed to save cache: {e}")

    def _load_index(self) -> Dict[str, Any]:
        """
        Loads index.json if available, else returns empty dict
        """
        if os.path.exists(self.index_file):
            try:
                with open(self.index_file, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                # Convert lists back to numpy arrays if needed
                if "vectors" in loaded:
                    loaded["vectors"] = np.array(loaded["vectors"], dtype="float32")
                return loaded
            except Exception as e:
                print(f"[!] Failed to load index: {e}")
                return {}
        return {}

    # ----------------- Embedding -----------------
    def embed_text(self, text: str) -> np.ndarray:
        """Get embedding for text, using cache if available"""
        if text in self.embeddings:
            # Convert from list to numpy if needed
            emb = self.embeddings[text]
            if isinstance(emb, list):
                emb = np.array(emb, dtype="float32")
            return emb
            
        # Generate new embedding
        emb = self.model.encode(text, convert_to_numpy=True, show_progress_bar=False)
        self.embeddings[text] = emb
        return emb

    def _save_index(self) -> None:
        """Save vector index to JSON"""
        try:
            # Convert numpy arrays to lists for JSON serialization
            index_data = {
                "vectors": self.index["vectors"].tolist() if "vectors" in self.index else []
            }
            with open(self.index_file, "w", encoding="utf-8") as f:
                json.dump(index_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[!] Failed to save index: {e}")

    # ----------------- Document Chunking -----------------
    def _chunk_doc(self, text: str) -> List[str]:
        """Split document into chunks for embedding"""
        words = text.split()
        return [
            " ".join(words[i:i + self.chunk_size]) 
            for i in range(0, len(words), self.chunk_size)
        ]

    # ----------------- Indexing -----------------
    def build_index(self, docs: List[str]) -> None:
        """
        Build search index from documents:
        1. Split docs into chunks
        2. Generate embeddings
        3. Save to index.json
        """
        self.docs = []
        vectors = []
        
        # Process documents
        for doc in docs:
            chunks = self._chunk_doc(doc)
            for chunk in chunks:
                self.docs.append(chunk)
                vectors.append(self.embed_text(chunk))
                
        # Build index
        self.index = {
            "vectors": np.array(vectors, dtype="float32")
        }
        
        # Save everything
        self._save_index()
        self._save_cache()

    # ----------------- Retrieval -----------------
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[str]:
        """
        Find most relevant document chunks for query
        Using cosine similarity between embeddings
        """
        if not self.docs or "vectors" not in self.index:
            return []

        if top_k is None:
            top_k = self.top_k

        # Get query embedding
        q_emb = self.embed_text(query)
        
        # Get document vectors
        doc_vecs = self.index["vectors"]
        
        # Compute cosine similarities
        # Normalize vectors
        q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-8)
        doc_norms = doc_vecs / (np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-8)
        
        # Compute similarities
        sims = np.dot(doc_norms, q_norm)
        
        # Get top k matches
        top_idxs = np.argsort(-sims)[:top_k]
        return [self.docs[i] for i in top_idxs if i < len(self.docs)]

    def enhance_prompt_with_rag(self, context: str, query: str) -> str:
        retrieved = self.retrieve(query)
        doc_text = "\n".join(retrieved)
        return f"{context}\n\nRelevant medical knowledge:\n{doc_text}\n\nUser query: {query}"

    # ----------------- Async Retrieval -----------------
    def retrieve_async(self, query: str, callback):
        def worker():
            docs = self.retrieve(query)
            callback(docs)
        self.executor.submit(worker)

# ----------------- Emergency Detection -----------------
def is_medical_emergency(text: str) -> bool:
    keywords = CONFIG.get("emergency_keywords", [
        "chest pain", "unconscious", "not breathing", "stroke", "severe bleeding"
    ])
    return any(k in text.lower() for k in keywords)

def get_emergency_response() -> str:
    return CONFIG.get(
        "emergency_response",
        "🚨 This sounds like a medical emergency. Please call your local emergency number immediately!"
    )
