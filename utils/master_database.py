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
    
    def _init_semantic_matcher(self):
        """Initialize semantic matcher for intelligent requirement matching."""
        try:
            from utils.extractors import get_semantic_matcher
            self.semantic_matcher = get_semantic_matcher()
            logger.info("Semantic matcher initialized for master database")
        except Exception as e:
            logger.warning(f"Could not initialize semantic matcher: {e}")
            self.semantic_matcher = None
    
    def search_requirement(self, requirement_text: str, threshold: float = 0.75) -> Optional[Dict[str, Any]]:
        """
        Search for a similar requirement in the master database using semantic matching.
        
        Args:
            requirement_text: The requirement to search for
            threshold: Minimum similarity score (0.0-1.0) to consider a match.
                      Default 0.75 for high-confidence semantic matches.
            
        Returns:
            Dict with match info if found, None otherwise
        """
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
