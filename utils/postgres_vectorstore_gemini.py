"""
PostgreSQL Vector Store for Gemini Branch - CORRECTED VERSION
Dedicated database connection for Gemini-only extraction and storage
"""

import os
import json
import numpy as np
from typing import List, Dict, Any, Optional
import psycopg2
from psycopg2.extras import Json
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

class PostgresVectorStoreGemini:
    def __init__(self):
        """Initialize connection to urs_gemini database"""
        self.connection_params = {
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'port': int(os.getenv('POSTGRES_PORT', '5433')),
            'database': os.getenv('POSTGRES_DB', 'urs_gemini'),
            'user': os.getenv('POSTGRES_USER', 'postgres'),
            'password': os.getenv('POSTGRES_PASSWORD', 'Patil1234')
        }
        
        # Initialize embedding model - using intfloat/e5-large-v2 for superior semantic understanding
        # This model provides state-of-the-art semantic embeddings with 1024 dimensions
        self.model = SentenceTransformer('intfloat/e5-large-v2')
        print("PostgreSQL vectorstore initialized with intfloat/e5-large-v2 for meaning-based matching")
        
        # Test connection
        self._test_connection()
    
    def _test_connection(self):
        """Test database connection"""
        try:
            conn = psycopg2.connect(**self.connection_params)
            conn.close()
            print("Connected to urs_gemini database successfully")
        except Exception as e:
            print(f"Database connection failed: {e}")
            raise
    
    def add_requirements(self, requirements: List[str], document_name: str, 
                        comments: Optional[str] = None, 
                        matched_document_name: Optional[str] = None) -> bool:
        """
        Add requirements to the vector store with enhanced metadata
        
        Args:
            requirements: List of requirement strings
            document_name: Name of the source document
            comments: Optional comments about the requirements
            matched_document_name: Optional name of matched document
            
        Returns:
            bool: Success status
        """
        try:
            conn = psycopg2.connect(**self.connection_params)
            cur = conn.cursor()
            
            for req in requirements:
                # Generate embedding with E5 passage prefix for document storage
                embedding = self.model.encode("passage: " + req)
                embedding_str = '[' + ','.join(map(str, embedding.tolist())) + ']'
                
                # Create metadata
                metadata = {
                    'source': document_name,
                    'extraction_method': 'gemini_holistic',
                    'processed_at': str(datetime.now())
                }
                
                # Insert requirement
                cur.execute("""
                    INSERT INTO requirements 
                    (requirement, embedding, metadata, document_name, comments, matched_document_name, extraction_type)
                    VALUES (%s, %s::vector, %s, %s, %s, %s, %s)
                """, (
                    req,
                    embedding_str,
                    Json(metadata),
                    document_name,
                    comments,
                    matched_document_name,
                    'holistic'
                ))
            
            conn.commit()
            cur.close()
            conn.close()
            
            print(f"Ô£à Added {len(requirements)} requirements to urs_gemini database")
            return True
            
        except Exception as e:
            print(f"ÔØî Error adding requirements: {e}")
            import traceback
            traceback.print_exc()
            return False

    def add_requirements_with_individual_comments(self, requirement_comment_pairs: List[Dict], document_name: str, 
                                                matched_document_name: Optional[str] = None) -> bool:
        """
        Add requirements with their individually associated comments
        
        Args:
            requirement_comment_pairs: List of dicts with 'requirement' and 'comments' keys
            document_name: Name of the source document
            matched_document_name: Optional name of matched document
            
        Returns:
            bool: Success status
        """
        try:
            conn = psycopg2.connect(**self.connection_params)
            cur = conn.cursor()
            
            # Debug: log first pair structure to file
            if requirement_comment_pairs:
                with open('c:\\vv\\db_debug.log', 'w', encoding='utf-8') as f:
                    f.write(f"Total pairs: {len(requirement_comment_pairs)}\n\n")
                    f.write(f"First pair:\n{requirement_comment_pairs[0]}\n\n")
                    f.write(f"First pair keys: {requirement_comment_pairs[0].keys()}\n\n")
                    f.write(f"First pair type: {type(requirement_comment_pairs[0])}\n")
                print(f"­ƒöì DEBUG: First pair structure logged to c:\\vv\\db_debug.log")
            
            for pair in requirement_comment_pairs:
                requirement_obj = pair.get('requirement', {})
                associated_comments = pair.get('comments', [])
                
                # Extract requirement text
                req_text = requirement_obj.get('text', '')
                if not req_text:
                    print(f"ÔÜá´©Å Skipping pair with no text: {pair}")
                    continue
                
                # Generate embedding with E5 passage prefix for document storage
                embedding = self.model.encode("passage: " + req_text)
                embedding_str = '[' + ','.join(map(str, embedding.tolist())) + ']'
                
                # Create metadata with requirement details
                metadata = {
                    'source': document_name,
                    'extraction_method': 'gemini_precise_pairing',
                    'processed_at': str(datetime.now()),
                    'category': requirement_obj.get('category', 'general'),
                    'priority': requirement_obj.get('priority', 'medium'),
                    'confidence': float(requirement_obj.get('confidence', 0.8)),  # Convert to Python float
                    'source_context': requirement_obj.get('source_context', '')
                }
                
                # Format individual comments for this specific requirement
                individual_comments = None
                if associated_comments:
                    # Clean comments to remove numpy types and ensure JSON serialization
                    cleaned_comments = []
                    for comment in associated_comments:
                        cleaned_comment = {}
                        for key, value in comment.items():
                            # Convert numpy types to Python native types
                            if hasattr(value, 'item'):  # numpy scalar
                                cleaned_comment[key] = float(value.item())
                            elif isinstance(value, (int, float, str, bool, type(None))):
                                cleaned_comment[key] = value
                            else:
                                cleaned_comment[key] = str(value)
                        cleaned_comments.append(cleaned_comment)
                    
                    comments_data = {
                        'count': len(cleaned_comments),
                        'comments': cleaned_comments,
                        'has_vendor_responses': any(c.get('comment_type') == 'vendor_response' for c in cleaned_comments),
                        'authors': list(set(c.get('author', 'Unknown') for c in cleaned_comments))
                    }
                    individual_comments = json.dumps(comments_data)
                
                # Insert requirement with its specific comments
                cur.execute("""
                    INSERT INTO requirements 
                    (requirement, embedding, metadata, document_name, comments, matched_document_name, extraction_type)
                    VALUES (%s, %s::vector, %s, %s, %s, %s, %s)
                """, (
                    req_text,
                    embedding_str,
                    Json(metadata),
                    document_name,
                    individual_comments,
                    matched_document_name,
                    'precise_pairing'
                ))
            
            conn.commit()
            cur.close()
            conn.close()
            
            print(f"Ô£à Added {len(requirement_comment_pairs)} requirements with individual comments to database")
            return True
            
        except Exception as e:
            print(f"ÔØî Error adding requirements with individual comments: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def search_similar_requirements(self, query: str, top_k: int = 5, threshold: float = 0.4) -> List[Dict[str, Any]]:
        """
        Search for similar requirements using pgvector's built-in similarity search.
        OPTIMIZED: Uses pgvector's cosine distance operator for efficient search.
        
        Args:
            query: Search query
            top_k: Number of results to return
            threshold: Minimum similarity score (default 0.4 for cross-document matching)
            
        Returns:
            List of similar requirements with semantic similarity scores
        """
        try:
            conn = psycopg2.connect(**self.connection_params)
            cur = conn.cursor()
            
            # First check if there are any requirements in the database
            cur.execute("SELECT COUNT(*) FROM requirements")
            total_count = cur.fetchone()[0]
            print(f"­ƒôè PostgreSQL database has {total_count} requirements")
            
            if total_count == 0:
                print("ÔÜá´©Å No historical requirements in PostgreSQL database yet")
                cur.close()
                conn.close()
                return []
            
            # Generate query embedding with E5 query prefix for search
            query_embedding = self.model.encode(["query: " + query])[0]
            embedding_str = '[' + ','.join(map(str, query_embedding.tolist())) + ']'
            
            # Use pgvector's <=> operator for cosine distance
            # <=> returns cosine distance (0 = identical, 2 = opposite)
            # We convert to similarity: similarity = 1 - distance
            # Limit to top_k*10 candidates to scan (reasonable performance)
            cur.execute("""
                SELECT 
                    requirement, 
                    metadata, 
                    document_name, 
                    comments, 
                    matched_document_name,
                    extraction_type,
                    created_at,
                    embedding,
                    embedding <=> %s::vector as distance
                FROM requirements
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (embedding_str, embedding_str, min(top_k * 10, 200)))
            
            results = cur.fetchall()
            print(f"­ƒöì Scanned {len(results)} candidate requirements from PostgreSQL")
            
            # Calculate similarity scores and filter by threshold
            formatted_results = []
            for row in results:
                distance = float(row[8])
                similarity = 1.0 - (distance / 2.0)  # Normalize distance to similarity
                
                if similarity >= threshold:
                    formatted_results.append({
                        'requirement': row[0],
                        'metadata': row[1],
                        'document_name': row[2],
                        'comments': row[3],
                        'matched_document_name': row[4],
                        'extraction_type': row[5],
                        'created_at': row[6],
                        'similarity_score': similarity
                    })
            
            # Sort by similarity (highest first) and limit to top_k
            formatted_results.sort(key=lambda x: x['similarity_score'], reverse=True)
            final_results = formatted_results[:top_k]
            
            print(f"Ô£à Found {len(final_results)} PostgreSQL matches above threshold {threshold}")
            
            cur.close()
            conn.close()
            
            return final_results
            
        except Exception as e:
            print(f"ÔØî Error searching requirements: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def get_all_requirements(self) -> List[Dict[str, Any]]:
        """Get all requirements from the database"""
        try:
            conn = psycopg2.connect(**self.connection_params)
            cur = conn.cursor()
            
            cur.execute("""
                SELECT 
                    id,
                    requirement, 
                    metadata, 
                    document_name, 
                    comments, 
                    matched_document_name,
                    extraction_type,
                    created_at
                FROM requirements 
                ORDER BY created_at DESC
            """)
            
            results = []
            for row in cur.fetchall():
                results.append({
                    'id': row[0],
                    'requirement': row[1],
                    'metadata': row[2],
                    'document_name': row[3],
                    'comments': row[4],
                    'matched_document_name': row[5],
                    'extraction_type': row[6],
                    'created_at': row[7]
                })
            
            cur.close()
            conn.close()
            
            return results
            
        except Exception as e:
            print(f"ÔØî Error getting requirements: {e}")
            return []
    
    def clear_database(self) -> bool:
        """Clear all requirements from the database"""
        try:
            conn = psycopg2.connect(**self.connection_params)
            cur = conn.cursor()
            
            cur.execute("DELETE FROM requirements")
            
            conn.commit()
            cur.close()
            conn.close()
            
            print("Ô£à Database cleared successfully")
            return True
            
        except Exception as e:
            print(f"ÔØî Error clearing database: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            conn = psycopg2.connect(**self.connection_params)
            cur = conn.cursor()
            
            # Get total count
            cur.execute("SELECT COUNT(*) FROM requirements")
            total_count = cur.fetchone()[0]
            
            # Get count by extraction type
            cur.execute("""
                SELECT extraction_type, COUNT(*) 
                FROM requirements 
                GROUP BY extraction_type
            """)
            extraction_stats = dict(cur.fetchall())
            
            # Get recent documents
            cur.execute("""
                SELECT document_name, COUNT(*) 
                FROM requirements 
                GROUP BY document_name 
                ORDER BY COUNT(*) DESC 
                LIMIT 10
            """)
            document_stats = dict(cur.fetchall())
            
            cur.close()
            conn.close()
            
            return {
                'total_requirements': total_count,
                'extraction_types': extraction_stats,
                'documents': document_stats
            }
            
        except Exception as e:
            print(f"Error getting stats: {e}")
            return {}
    
    def update_requirement_comments(self, requirement_id: int, comments_json: str, page_reference: str = None) -> bool:
        """
        Update comments for an existing requirement
        
        Args:
            requirement_id: ID of the requirement to update
            comments_json: JSON string containing comments data
            page_reference: Optional page reference (ignored - column doesn't exist)
            
        Returns:
            bool: Success status
        """
        try:
            conn = psycopg2.connect(**self.connection_params)
            cursor = conn.cursor()
            
            # Update the requirement with comments (only columns that exist)
            update_query = """
            UPDATE requirements 
            SET comments = %s
            WHERE id = %s
            """
            
            cursor.execute(update_query, (comments_json, requirement_id))
            conn.commit()
            
            cursor.close()
            conn.close()
            
            return True
            
        except Exception as e:
            print(f"Error updating requirement comments: {e}")
            return False
    
    def search_requirements_by_document(self, document_name: str) -> List[Dict[str, Any]]:
        """
        Get all requirements for a specific document
        
        Args:
            document_name: Name of the document
            
        Returns:
            List of requirements from the document
        """
        try:
            conn = psycopg2.connect(**self.connection_params)
            cursor = conn.cursor()
            
            # Only select columns that actually exist in the database
            query = """
            SELECT id, requirement, comments, document_name, metadata, extraction_type
            FROM requirements 
            WHERE document_name = %s OR document_name LIKE %s
            ORDER BY id
            """
            
            cursor.execute(query, (document_name, f"%{document_name}%"))
            results = cursor.fetchall()
            
            requirements = []
            for row in results:
                requirements.append({
                    'id': row[0],
                    'requirement': row[1],
                    'comments': row[2],
                    'document_name': row[3],
                    'metadata': row[4],
                    'extraction_type': row[5]
                })
            
            cursor.close()
            conn.close()
            
            return requirements
            
        except Exception as e:
            print(f"Error searching requirements by document: {e}")
            return []
    
    def get_all_documents(self) -> List[Dict[str, Any]]:
        """
        Get all unique documents in the database with their stats
        
        Returns:
            List of document information
        """
        try:
            conn = psycopg2.connect(**self.connection_params)
            cursor = conn.cursor()
            
            # Note: created_at might not exist, so use a safer query
            query = """
            SELECT document_name, 
                   COUNT(*) as requirement_count,
                   MIN(id) as min_id
            FROM requirements 
            GROUP BY document_name
            ORDER BY MIN(id) DESC
            """
            
            cursor.execute(query)
            results = cursor.fetchall()
            
            documents = []
            for row in results:
                documents.append({
                    'filename': row[0],
                    'requirement_count': row[1],
                    'id': row[2]
                })
            
            cursor.close()
            conn.close()
            
            return documents
            
        except Exception as e:
            print(f"Error getting all documents: {e}")
            return []

if __name__ == "__main__":
    # Test the vector store
    vs = PostgresVectorStoreGemini()
    
    # Test search by document
    docs = vs.get_all_documents()
    print(f"Found {len(docs)} documents")
    
    if docs:
        test_doc = docs[0]['filename']
        requirements = vs.search_requirements_by_document(test_doc)
        print(f"Found {len(requirements)} requirements for {test_doc}")
