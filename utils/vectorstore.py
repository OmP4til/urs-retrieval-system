import os, json
import numpy as np
from sentence_transformers import SentenceTransformer

try:
    import faiss
    FAISS_AVAILABLE = True
except Exception:
    FAISS_AVAILABLE = False


def normalize(vecs: np.ndarray) -> np.ndarray:
    """L2-normalize vectors row-wise."""
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.clip(norms, 1e-8, None)


class SimpleVectorStore:
    def __init__(
        self,
        model_name="all-MiniLM-L6-v2",
        index_path="data/vectorstore/faiss_index.bin",
        metadata_path="data/vectorstore/metadata.jsonl",
    ):
        self.embedder = SentenceTransformer(model_name)
        self.dim = self.embedder.get_sentence_embedding_dimension()
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.metadatas = []
        self.embeddings = None
        self.index = None
        self.current_file = None  # Track the current file being indexed
        self.files_in_index = set()  # Track all files in the index
        self._cached_embeddings = {}  # Cache for query embeddings

        if FAISS_AVAILABLE and os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            try:
                # Use IVFFlat index for faster search with large datasets
                self.index = faiss.read_index(self.index_path)
                if not isinstance(self.index, faiss.IndexIVFFlat) and self.index.ntotal > 1000:
                    # Convert to IVFFlat for better performance
                    nlist = min(int(np.sqrt(self.index.ntotal)), 100)  # number of clusters
                    quantizer = faiss.IndexFlatIP(self.dim)
                    new_index = faiss.IndexIVFFlat(quantizer, self.dim, nlist, faiss.METRIC_INNER_PRODUCT)
                    new_index.train(faiss.vector_to_array(self.index).reshape(-1, self.dim))
                    new_index.add(faiss.vector_to_array(self.index).reshape(-1, self.dim))
                    self.index = new_index
                
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    self.metadatas = [json.loads(l) for l in f]
                    self.files_in_index = {m["source_file"] for m in self.metadatas}
            except Exception as e:
                print(f"Error loading index: {e}")
                self.index = None
                self.metadatas = []
                self.files_in_index = set()
        else:
            self.embeddings = np.zeros((0, self.dim), dtype="float32")
            
    def clear(self, keep_previous=True):
        """Clear all documents from the store.
        
        Args:
            keep_previous (bool): If True, keeps documents from the previous file
                               If False, clears everything
        """
        if not keep_previous:
            # Clear everything
            self.metadatas = []
            self.files_in_index = set()
            if FAISS_AVAILABLE:
                self.index = faiss.IndexFlatIP(self.dim)
            else:
                self.embeddings = np.zeros((0, self.dim), dtype="float32")
            
            # Remove existing index and metadata files
            if os.path.exists(self.index_path):
                os.remove(self.index_path)
            if os.path.exists(self.metadata_path):
                os.remove(self.metadata_path)
        else:
            # Keep previous file's documents if they exist
            if self.current_file and self.metadatas:
                # Filter out current file's documents
                previous_docs = [(meta, i) for i, meta in enumerate(self.metadatas) 
                               if meta["source_file"] != self.current_file]
                
                if previous_docs:
                    new_metadatas = [meta for meta, _ in previous_docs]
                    self.metadatas = new_metadatas
                    self.files_in_index = {m["source_file"] for m in self.metadatas}
                    
                    # We need to rebuild the index from scratch since we can't reliably reconstruct vectors
                    if FAISS_AVAILABLE:
                        self.index = faiss.IndexFlatIP(self.dim)
                    else:
                        self.embeddings = np.zeros((0, self.dim), dtype="float32")
                        
                    # Save the current state so we can restore it later
                    if os.path.exists(self.index_path):
                        os.rename(self.index_path, self.index_path + '.bak')
                    if os.path.exists(self.metadata_path):
                        os.rename(self.metadata_path, self.metadata_path + '.bak')
                else:
                    # If no previous docs to keep, just clear everything
                    self.clear(keep_previous=False)

    def add_document(self, text: str, metadata: dict):
        """Add a document embedding and metadata to the store."""
        if not text or not text.strip():
            print(f"[WARN] Skipping empty text for {metadata}")
            return

        # Update current file and files in index
        source_file = metadata.get("source_file")
        if source_file:
            self.current_file = source_file
            self.files_in_index.add(source_file)

        emb = self.embedder.encode(text, convert_to_numpy=True).astype("float32").reshape(1, -1)
        emb = normalize(emb)

        if FAISS_AVAILABLE:
            if self.index is None:
                self.index = faiss.IndexFlatIP(self.dim)  # cosine similarity
            self.index.add(emb)
        else:
            if self.embeddings.size == 0:
                self.embeddings = emb
            else:
                self.embeddings = np.vstack([self.embeddings, emb])

        self.metadatas.append(metadata)

    def save(self):
        """Save index and metadata to disk."""
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        if FAISS_AVAILABLE and self.index is not None:
            faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            for m in self.metadatas:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")

    def search(self, query_text: str, top_k: int = 5, min_score: float = 0.5, nprobe: int = None):
        """Search for top_k similar documents to query_text.
        
        Args:
            query_text (str): The text to search for
            top_k (int): Maximum number of results to return
            min_score (float): Minimum similarity score (0-1) to include in results
            nprobe (int): Number of clusters to visit for IVF indices (higher = more accurate but slower)
        """
        if not query_text or not query_text.strip():
            return []

        # Check cache first
        cache_key = query_text.strip()
        if cache_key in self._cached_embeddings:
            q = self._cached_embeddings[cache_key]
        else:
            q = self.embedder.encode(query_text, convert_to_numpy=True).astype("float32").reshape(1, -1)
            q = normalize(q)
            self._cached_embeddings[cache_key] = q
            
            # Keep cache size reasonable
            if len(self._cached_embeddings) > 1000:
                self._cached_embeddings.pop(next(iter(self._cached_embeddings)))

        if FAISS_AVAILABLE and self.index is not None:
            # Optimize search parameters for IVF index
            if isinstance(self.index, faiss.IndexIVFFlat):
                if nprobe is None:
                    # Auto-adjust nprobe based on dataset size
                    nprobe = min(max(int(np.sqrt(self.index.ntotal)), 1), 50)
                self.index.nprobe = nprobe
            
            # Search for more results than needed to ensure we get enough after filtering
            D, I = self.index.search(q, min(top_k * 3, self.index.ntotal))
            results = []
            for score, idx in zip(D[0], I[0]):
                if idx < 0 or idx >= len(self.metadatas) or score < min_score:
                    continue
                    
                metadata = self.metadatas[idx]
                source_file = metadata.get("source_file", "")
                
                # Add source information to the result
                results.append({
                    "score": float(score),
                    "metadata": metadata,
                    "is_from_current": source_file == self.current_file,
                    "source_file": source_file
                })
                
                if len(results) >= top_k:
                    break
                    
            return results

        # Fallback: numpy search
        if self.embeddings is None or self.embeddings.size == 0:
            return []
            
        sims = (self.embeddings @ q.T).flatten()
        idxs = np.argsort(-sims)
        results = []
        
        for i in idxs:
            score = float(sims[i])
            if score < min_score:
                break
                
            metadata = self.metadatas[i]
            source_file = metadata.get("source_file", "")
            
            results.append({
                "score": score,
                "metadata": metadata,
                "is_from_current": source_file == self.current_file,
                "source_file": source_file
            })
            
            if len(results) >= top_k:
                break
                
        return results
