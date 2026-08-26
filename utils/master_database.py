"""
Master Database Layer for checking pre-existing requirements and responses.
This module provides functionality to check the Excel master database before
querying the PostgreSQL historical database.
"""

import pandas as pd
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

class MasterDatabase:
    """
    Interface to the URS Response Automation Master Database Excel file.
    Provides semantic matching between new requirements and existing ones.
    """
    
    def __init__(self, excel_path: str = None):
        """
        Initialize the master database interface.

        Args:
            excel_path: Path to the Excel master database file. Defaults to
                config.MASTER_DB_PATH, which is anchored to the project root
                rather than the working directory.
        """
        if excel_path is None:
            from config import MASTER_DB_PATH
            excel_path = str(MASTER_DB_PATH)
        self.excel_path = excel_path
        self.df = None
        self.semantic_matcher = None
        self._load_database()
        self._init_semantic_matcher()
    
    def _load_database(self):
        """Load the Excel database into memory."""
        try:
            if not Path(self.excel_path).exists():
                logger.warning(f"Master database file not found: {self.excel_path}")
                self.df = pd.DataFrame()
                return
            
            # Read Master_DB sheet
            self.df = pd.read_excel(self.excel_path, sheet_name='Master_DB')
            
            # Clean column names
            self.df.columns = self.df.columns.str.strip()
            
            # Rename columns for clarity
            if 'Point' in self.df.columns:
                self.df.rename(columns={'Point': 'requirement'}, inplace=True)
            if 'Comment' in self.df.columns:
                self.df.rename(columns={'Comment': 'response'}, inplace=True)
            if 'Deviation Number' in self.df.columns:
                self.df.rename(columns={'Deviation Number': 'deviation_id'}, inplace=True)
            
            # Remove rows with empty requirements
            self.df = self.df[self.df['requirement'].notna()]
            self.df = self.df[self.df['requirement'].str.strip() != '']
            
            # Clean text fields
            self.df['requirement'] = self.df['requirement'].str.strip()
            self.df['response'] = self.df['response'].fillna('').str.strip()
            
            logger.info(f"Loaded master database with {len(self.df)} requirement-response pairs")
            
        except Exception as e:
            logger.error(f"Error loading master database: {e}")
            self.df = pd.DataFrame()
    
    # How many vector-search candidates get the full validated scoring.
    _SHORTLIST = 10

    def _init_semantic_matcher(self):
        """Initialize semantic matcher for intelligent requirement matching."""
        try:
            from utils.extractors import get_semantic_matcher
            self.semantic_matcher = get_semantic_matcher()
            logger.info("Semantic matcher initialized for master database")
        except Exception as e:
            logger.warning(f"Could not initialize semantic matcher: {e}")
            self.semantic_matcher = None
        self._matrix = None
        self._matrix_failed = False

    def _get_matrix(self):
        """
        Embeddings for every master requirement as one normalised matrix.

        Scoring a requirement against the master database used to be a Python
        loop calling the matcher once per row: 481 microseconds per row, so 112
        ms against 233 rows and 41 seconds for a 365-requirement document. At
        50,000 rows that loop would take hours.

        One matrix multiply replaces the loop and runs in about 0.08 ms, so the
        shortlist is effectively free and only those few candidates need the
        expensive validated scoring.

        Built once per instance and cached on disk, keyed by the workbook's
        size and modification time.
        """
        if self._matrix is not None or self._matrix_failed:
            return self._matrix

        try:
            import numpy as np

            cache_path = self._matrix_cache_path()
            if cache_path and cache_path.exists():
                self._matrix = np.load(cache_path)
                if self._matrix.shape[0] == len(self.df):
                    logger.info("Loaded master database embeddings from %s", cache_path.name)
                    return self._matrix
                self._matrix = None      # stale cache, rebuild

            # Match _get_embedding's normalisation exactly, so the shortlist
            # and the validated re-scoring see the same vectors.
            from utils.extractors import clean_extracted_text
            texts = ["query: " + clean_extracted_text(str(r)).lower()
                     for r in self.df['requirement'].tolist()]
            logger.info("Embedding %d master database requirements (one-off)...", len(texts))
            self._matrix = self.semantic_matcher.model.encode(
                texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)

            if cache_path:
                try:
                    np.save(cache_path, self._matrix)
                    logger.info("Cached master database embeddings to %s", cache_path.name)
                except OSError as e:
                    logger.warning("Could not cache master embeddings: %s", e)

            return self._matrix

        except Exception as e:
            logger.warning("Vectorised master matching unavailable (%s); using the row loop", e)
            self._matrix_failed = True
            return None

    # Bump when the embedding model or its prefix changes, so cached vectors
    # built under the old scheme are not silently reused.
    _EMBED_SCHEME = "e5q1"

    def _matrix_cache_path(self):
        """
        Cache file keyed by workbook size, mtime and embedding scheme.

        Size and mtime catch edits to the workbook; the scheme tag catches a
        change of model or prefix, which would otherwise leave vectors that no
        longer match how queries are encoded.
        """
        try:
            path = Path(self.excel_path)
            stat = path.stat()
            return path.with_name(
                f".{path.stem}.emb.{self._EMBED_SCHEME}.{stat.st_size}.{int(stat.st_mtime)}.npy")
        except OSError:
            return None
    
    def search_requirement(self, requirement_text: str, threshold: float = None) -> Optional[Dict[str, Any]]:
        """
        Search for a similar requirement in the master database using semantic matching.
        
        Args:
            requirement_text: The requirement to search for
            threshold: Minimum raw cosine similarity to consider a match.
                Defaults to config.MASTER_DB_THRESHOLD. Note e5-large-v2 has a
                high floor - unrelated requirements score around 0.80 - so a
                threshold below about 0.85 matches essentially everything.

        Returns:
            Dict with match info if found, None otherwise
        """
        if threshold is None:
            from config import MASTER_DB_THRESHOLD
            threshold = MASTER_DB_THRESHOLD

        if self.df is None or self.df.empty:
            return None
        
        if not requirement_text or not requirement_text.strip():
            return None
        
        requirement_text = requirement_text.strip()
        
        # First try exact match (case-insensitive)
        exact_match = self.df[self.df['requirement'].str.lower() == requirement_text.lower()]
        if not exact_match.empty:
            row = exact_match.iloc[0]
            return {
                'deviation_id': row.get('deviation_id', 'N/A'),
                'requirement': row['requirement'],
                'response': row['response'],
                'similarity': 1.0,
                'match_type': 'exact',
                'source': 'master_database'
            }
        
        # Semantic/fuzzy matching
        if self.semantic_matcher:
            best_match = None
            best_score = 0.0

            matrix = self._get_matrix()
            if matrix is not None:
                # One matmul shortlists the nearest rows, then only those get
                # the full validated scoring. Same answer as scanning every
                # row, without the per-row Python call.
                import numpy as np
                # Reuse the matcher's embedding cache rather than re-encoding:
                # a fresh e5-large encode costs ~180 ms on CPU and would swamp
                # the 0.08 ms matmul this optimisation exists for.
                query = self.semantic_matcher._get_embedding(requirement_text)
                if query.size == 0:
                    return None
                norm = np.linalg.norm(query)
                if norm:
                    query = query / norm
                sims = matrix @ query
                top_n = min(self._SHORTLIST, len(sims))
                candidates = np.argpartition(-sims, top_n - 1)[:top_n]

                for i in candidates:
                    row = self.df.iloc[int(i)]
                    similarity = self.semantic_matcher.calculate_semantic_similarity(
                        requirement_text, row['requirement'])
                    if similarity > best_score:
                        best_score = similarity
                        best_match = row
            else:
                for idx, row in self.df.iterrows():
                    master_req = row['requirement']
                    similarity = self.semantic_matcher.calculate_semantic_similarity(
                        requirement_text, master_req
                    )

                    if similarity > best_score:
                        best_score = similarity
                        best_match = row
            
            if best_score >= threshold:
                return {
                    'deviation_id': best_match.get('deviation_id', 'N/A'),
                    'requirement': best_match['requirement'],
                    'response': best_match['response'],
                    'similarity': float(best_score),
                    'match_type': 'semantic',
                    'source': 'master_database'
                }
        else:
            # Fallback to simple substring matching
            for idx, row in self.df.iterrows():
                master_req = row['requirement'].lower()
                req_lower = requirement_text.lower()
                
                # Check if significant overlap
                if len(requirement_text) > 20:
                    # For longer requirements, check substring
                    if req_lower in master_req or master_req in req_lower:
                        return {
                            'deviation_id': row.get('deviation_id', 'N/A'),
                            'requirement': row['requirement'],
                            'response': row['response'],
                            'similarity': 0.8,
                            'match_type': 'substring',
                            'source': 'master_database'
                        }
        
        return None
    
    def batch_search_requirements(self, requirements: List[str], threshold: float = 0.7) -> Dict[str, Optional[Dict[str, Any]]]:
        """
        Search for multiple requirements in batch.
        
        Args:
            requirements: List of requirement texts to search
            threshold: Minimum similarity threshold
            
        Returns:
            Dict mapping requirement text to match info (or None if no match)
        """
        results = {}
        
        for req_text in requirements:
            match = self.search_requirement(req_text, threshold)
            results[req_text] = match
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the master database."""
        if self.df is None or self.df.empty:
            return {
                'total_requirements': 0,
                'requirements_with_responses': 0,
                'requirements_without_responses': 0,
                'unique_deviation_ids': 0
            }
        
        return {
            'total_requirements': len(self.df),
            'requirements_with_responses': len(self.df[self.df['response'] != '']),
            'requirements_without_responses': len(self.df[self.df['response'] == '']),
            'unique_deviation_ids': self.df['deviation_id'].nunique() if 'deviation_id' in self.df.columns else 0
        }
    
    def reload(self):
        """Reload the database from file."""
        self._load_database()
        logger.info("Master database reloaded")


def test_master_database():
    """Test function for the master database."""
    db = MasterDatabase()
    
    print("Master Database Statistics:")
    stats = db.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test search
    test_req = "CIP System"
    print(f"\n\nSearching for: '{test_req}'")
    result = db.search_requirement(test_req)
    
    if result:
        print(f"✅ Found match!")
        print(f"  Deviation ID: {result['deviation_id']}")
        print(f"  Requirement: {result['requirement'][:100]}...")
        print(f"  Response: {result['response'][:100]}...")
        print(f"  Similarity: {result['similarity']:.2f}")
        print(f"  Match Type: {result['match_type']}")
    else:
        print("❌ No match found")


if __name__ == "__main__":
    test_master_database()
