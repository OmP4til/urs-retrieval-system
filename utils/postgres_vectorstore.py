import os
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
from sqlalchemy import func, text
from .db_models import Document, Requirement, get_session

def normalize(vecs: np.ndarray) -> np.ndarray:
    """L2-normalize vectors row-wise."""
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.clip(norms, 1e-8, None)

class PostgresVectorStore:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        db_url: str = None,
    ):
        """Initialize PostgresVectorStore.
        
        Args:
            model_name: Name of the sentence-transformer model to use
            db_url: Database connection URL
        """
        self.embedder = SentenceTransformer(model_name)
        self.dim = self.embedder.get_sentence_embedding_dimension()
        self.db_url = db_url or os.getenv("DATABASE_URL")
        if not self.db_url:
            raise ValueError("Database URL must be provided")
            
        from .db_models import init_db
        self.engine = init_db(self.db_url)
        
        # Create pgvector extension if it doesn't exist
        with self.engine.connect() as conn:
            conn.execute(text('CREATE EXTENSION IF NOT EXISTS vector;'))
            conn.commit()
    
    def add_document(self, filename: str, requirements: List[Dict[str, Any]]):
        """Add a document and its requirements to the database.
        
        Args:
            filename: Name of the document
            requirements: List of requirements with text and metadata
        """
        session = get_session(self.engine)
        try:
            # Create document record
            doc = Document(filename=filename)
            session.add(doc)
            session.flush()  # Get the document ID
            
            # Process requirements in batches
            batch_size = 100
            for i in range(0, len(requirements), batch_size):
                batch = requirements[i:i + batch_size]
                
                # Generate embeddings for the batch
                texts = [r["text"] for r in batch]
                embeddings = self.embedder.encode(texts, convert_to_numpy=True)
                embeddings = normalize(embeddings)
                
                # Create requirement records
                for req, emb in zip(batch, embeddings):
                    requirement = Requirement(
                        document_id=doc.id,
                        requirement_id=req.get("id"),
                        text=req["text"],
                        page_number=req.get("page_number"),
                        embedding=emb.tolist(),
                        comments=req.get("comments", []),
                        responses=req.get("responses", [])
                    )
                    session.add(requirement)
                
                session.flush()
            
            session.commit()
            return doc.id
            
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def search(self, query_text: str, top_k: int = 5, min_score: float = 0.5) -> List[Dict]:
        """Search for similar requirements using cosine similarity.
        
        Args:
            query_text: Text to search for
            top_k: Maximum number of results to return
            min_score: Minimum similarity score threshold
        """
        if not query_text.strip():
            return []
            
        # Generate query embedding
        query_emb = self.embedder.encode(query_text, convert_to_numpy=True)
        query_emb = normalize(query_emb.reshape(1, -1))[0]
        
        session = get_session(self.engine)
        try:
            # Use pgvector's L2 distance and convert to cosine similarity
            # cos_sim = 1 - (L2_distance^2 / 2)
            results = session.execute(text("""
                SELECT 
                    r.requirement_id,
                    r.text,
                    r.page_number,
                    r.comments,
                    r.responses,
                    d.filename,
                    (1 - (r.embedding <-> :query)^2 / 2) as similarity
                FROM requirements r
                JOIN documents d ON r.document_id = d.id
                WHERE (1 - (r.embedding <-> :query)^2 / 2) >= :min_score
                ORDER BY r.embedding <-> :query
                LIMIT :top_k
            """), {
                "query": query_emb.tolist(),
                "min_score": min_score,
                "top_k": top_k
            })
            
            return [{
                "score": float(row.similarity),
                "metadata": {
                    "source_file": row.filename,
                    "page_number": row.page_number,
                    "requirement_id": row.requirement_id,
                    "comments": row.comments,
                    "responses": row.responses
                },
                "text": row.text
            } for row in results]
            
        finally:
            session.close()
    
    def get_all_documents(self) -> List[Dict]:
        """Get a list of all indexed documents."""
        session = get_session(self.engine)
        try:
            docs = session.query(Document).all()
            return [{
                "id": doc.id,
                "filename": doc.filename,
                "upload_date": doc.upload_date.isoformat(),
                "requirement_count": len(doc.requirements)
            } for doc in docs]
        finally:
            session.close()
    
    def delete_document(self, document_id: int):
        """Delete a document and all its requirements."""
        session = get_session(self.engine)
        try:
            doc = session.query(Document).get(document_id)
            if doc:
                session.delete(doc)
                session.commit()
        finally:
            session.close()