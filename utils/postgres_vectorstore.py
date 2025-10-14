
import os
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
from sqlalchemy import text
from .db_models import Document, Requirement, get_session, init_db

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
            raise ValueError("Database URL must be provided or set in DATABASE_URL environment variable")
        
        # Initialize database
        self.engine = init_db(self.db_url)
        self.current_file = None
        
        print(f"PostgresVectorStore initialized with model: {model_name}")
        print(f"Connected to database")
    
    def clear(self, keep_previous: bool = True):
        """Clear documents from the database.
        
        Args:
            keep_previous: If True, keeps documents from previous files.
                          If False, clears everything.
        """
        session = get_session(self.engine)
        try:
            if not keep_previous:
                # Clear all documents
                session.query(Requirement).delete()
                session.query(Document).delete()
                session.commit()
                print("Cleared all documents from database")
            else:
                # Keep previous file's documents
                if self.current_file:
                    doc = session.query(Document).filter_by(filename=self.current_file).first()
                    if doc:
                        session.delete(doc)  # Cascade will delete requirements
                        session.commit()
                        print(f"Cleared documents from: {self.current_file}")
        except Exception as e:
            session.rollback()
            print(f"Error clearing database: {e}")
            raise e
        finally:
            session.close()
    
    def _clean_metadata_for_json(self, data):
        """Convert any numpy/float32 values to regular Python types for JSON serialization."""
        if isinstance(data, list):
            return [self._clean_metadata_for_json(item) for item in data]
        elif isinstance(data, dict):
            return {key: self._clean_metadata_for_json(value) for key, value in data.items()}
        elif hasattr(data, 'item'):  # numpy scalars
            return data.item()
        elif isinstance(data, (float, int)):
            return float(data) if isinstance(data, float) else int(data)
        else:
            return data

    def add_document(self, text: str, metadata: Dict[str, Any]):
        """Add a single requirement to the database.
        
        Args:
            text: The requirement text
            metadata: Dictionary containing source_file, page_number, requirement_id, etc.
        """
        if not text or not text.strip():
            print(f"[WARN] Skipping empty text")
            return
        
        source_file = metadata.get("source_file")
        if not source_file:
            raise ValueError("source_file is required in metadata")
        
        self.current_file = source_file
        
        session = get_session(self.engine)
        try:
            # Get or create document
            doc = session.query(Document).filter_by(filename=source_file).first()
            if not doc:
                doc = Document(filename=source_file)
                session.add(doc)
                session.flush()
            
            # Generate embedding
            emb = self.embedder.encode(text, convert_to_numpy=True).astype("float32")
            emb = normalize(emb.reshape(1, -1))[0]
            
            # Clean metadata to ensure JSON serialization compatibility
            clean_comments = self._clean_metadata_for_json(metadata.get("comments", []))
            clean_responses = self._clean_metadata_for_json(metadata.get("responses", []))
            
            # Create requirement
            requirement = Requirement(
                document_id=doc.id,
                requirement_id=metadata.get("requirement_id"),
                text=text,
                page_number=metadata.get("page_number"),
                embedding=emb.tolist(),
                comments=clean_comments,
                responses=clean_responses
            )
            session.add(requirement)
            session.commit()
            
        except Exception as e:
            session.rollback()
            print(f"Error adding document: {e}")
            raise e
        finally:
            session.close()
    
    def save(self):
        """Save changes to database (no-op for PostgreSQL, commits happen per transaction)."""
        print("All changes saved to PostgreSQL")
    
    def search(self, query_text: str, top_k: int = 5, min_score: float = 0.5) -> List[Dict]:
        """Search for similar requirements using cosine similarity.
        
        Args:
            query_text: Text to search for
            top_k: Maximum number of results to return
            min_score: Minimum similarity score threshold (0-1)
        
        Returns:
            List of matching requirements with scores and metadata
        """
        if not query_text or not query_text.strip():
            return []
        
        # Generate query embedding
        query_emb = self.embedder.encode(query_text, convert_to_numpy=True).astype("float32")
        query_emb = normalize(query_emb.reshape(1, -1))[0]
        
        session = get_session(self.engine)
        try:
            # Use cosine similarity: 1 - cosine_distance
            # pgvector's <=> operator returns cosine distance (0 = identical, 2 = opposite)
            # cosine similarity = 1 - (cosine_distance / 2)
            query = text("""
                SELECT 
                    r.id,
                    r.requirement_id,
                    r.text,
                    r.page_number,
                    r.comments,
                    r.responses,
                    d.filename,
                    1 - (r.embedding <=> CAST(:query_vector AS vector)) as similarity
                FROM requirements r
                JOIN documents d ON r.document_id = d.id
                WHERE 1 - (r.embedding <=> CAST(:query_vector AS vector)) >= :min_score
                ORDER BY r.embedding <=> CAST(:query_vector AS vector)
                LIMIT :top_k
            """)
            
            results = session.execute(
                query,
                {
                    "query_vector": query_emb.tolist(),
                    "min_score": min_score,
                    "top_k": top_k
                }
            ).fetchall()
            
            return [{
                "score": float(row.similarity),
                "metadata": {
                    "source_file": row.filename,
                    "page_number": row.page_number,
                    "requirement_id": row.requirement_id,
                    "comments": row.comments or [],
                    "responses": row.responses or []
                },
                "text": row.text,
                "is_from_current": row.filename == self.current_file,
                "source_file": row.filename
            } for row in results]
            
        except Exception as e:
            print(f"Error searching: {e}")
            return []
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
                print(f"Deleted document ID: {document_id}")
        except Exception as e:
            session.rollback()
            print(f"Error deleting document: {e}")
            raise e
        finally:
            session.close()

