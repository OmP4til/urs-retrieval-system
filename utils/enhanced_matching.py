"""
Enhanced Semantic Matching with Multi-Layer Validation
Significantly improves matching accuracy for URS requirements
"""

from typing import List, Dict, Any, Tuple
import re
from sentence_transformers import SentenceTransformer
import numpy as np

class EnhancedSemanticMatcher:
    """
    Multi-layer semantic matching with domain-specific validation
    """
    
    def __init__(self):
        self.model = SentenceTransformer('all-mpnet-base-v2')
        self.domain_keywords = self._load_domain_keywords()
        self.technical_patterns = self._load_technical_patterns()
        
    def _load_domain_keywords(self) -> Dict[str, List[str]]:
        """Load pharmaceutical/manufacturing domain keywords"""
        return {
            'equipment': [
                'pump', 'valve', 'tank', 'vessel', 'motor', 'compressor',
                'mixer', 'agitator', 'filter', 'heat exchanger', 'reactor',
                'dryer', 'granulator', 'tablet press', 'coating', 'mill'
            ],
            'process': [
                'cip', 'clean in place', 'sterilization', 'sanitization',
                'washing', 'rinsing', 'drying', 'mixing', 'blending',
                'granulation', 'coating', 'milling', 'sieving'
            ],
            'specifications': [
                'temperature', 'pressure', 'flow', 'speed', 'capacity',
                'volume', 'level', 'ph', 'conductivity', 'viscosity',
                'rpm', 'bar', 'psi', 'liter', 'kg', 'meter'
            ],
            'materials': [
                'stainless steel', '316l', '304', 'ptfe', 'epdm', 'viton',
                'silicone', 'gasket', 'seal', 'o-ring', 'metallic',
                'non-metallic', 'product contact', 'surface finish'
            ],
            'safety': [
                'emergency', 'stop', 'interlock', 'alarm', 'safety',
                'protection', 'guarding', 'earthing', 'grounding',
                'explosion proof', 'atex', 'hazard'
            ],
            'automation': [
                'plc', 'hmi', 'scada', 'control', 'sensor', 'transmitter',
                'actuator', 'solenoid', 'automation', 'monitoring',
                'data logging', 'recipe', 'batch'
            ],
            'compliance': [
                'gmp', 'cgmp', 'fda', '21 cfr', 'eu gmp', 'validation',
                'qualification', 'iq', 'oq', 'pq', 'audit trail',
                'electronic signature', 'data integrity'
            ]
        }
    
    def _load_technical_patterns(self) -> List[str]:
        """Regex patterns for technical specifications"""
        return [
            r'\d+\s*(?:bar|psi|mpa)',  # Pressure
            r'\d+\s*(?:°c|°f|celsius|fahrenheit)',  # Temperature
            r'\d+\s*(?:rpm|hz)',  # Speed/frequency
            r'\d+\s*(?:l|liter|m3|gallon)',  # Volume
            r'\d+\s*(?:kg|g|ton|lb)',  # Weight
            r'\d+\s*(?:kw|hp|watt)',  # Power
            r'\d+\s*(?:mm|cm|m|inch)',  # Dimensions
            r'\d+\s*(?:%|percent)',  # Percentage
            r'ph\s*\d+',  # pH value
            r'\d+\s*(?:phase|step|stage)',  # Process phases
        ]
    
    def enhanced_match(self, query: str, candidate: str, 
                      category_hint: str = None) -> Tuple[float, Dict[str, Any]]:
        """
        Multi-layer matching with comprehensive validation
        
        Returns:
            (final_score, match_details)
        """
        # Layer 1: Semantic similarity (base score)
        semantic_score = self._calculate_semantic_similarity(query, candidate)
        
        # Layer 2: Keyword overlap validation
        keyword_score, keyword_details = self._calculate_keyword_overlap(query, candidate)
        
        # Layer 3: Technical pattern matching
        technical_score, technical_details = self._calculate_technical_similarity(query, candidate)
        
        # Layer 4: Structural similarity
        structural_score = self._calculate_structural_similarity(query, candidate)
        
        # Layer 5: Domain-specific validation
        domain_score, domain_details = self._calculate_domain_relevance(
            query, candidate, category_hint
        )
        
        # Weighted combination
        weights = {
            'semantic': 0.30,      # Base semantic understanding
            'keyword': 0.25,       # Keyword overlap critical for relevance
            'technical': 0.20,     # Technical specifications
            'structural': 0.10,    # Similar structure/format
            'domain': 0.15         # Domain relevance
        }
        
        final_score = (
            weights['semantic'] * semantic_score +
            weights['keyword'] * keyword_score +
            weights['technical'] * technical_score +
            weights['structural'] * structural_score +
            weights['domain'] * domain_score
        )
        
        # Apply penalties for mismatches
        if self._has_contradictory_terms(query, candidate):
            final_score *= 0.3  # Heavy penalty
        
        if keyword_score < 0.2 and semantic_score > 0.7:
            # High semantic but no keyword overlap = false positive
            final_score *= 0.5
        
        match_details = {
            'semantic_score': semantic_score,
            'keyword_score': keyword_score,
            'keyword_details': keyword_details,
            'technical_score': technical_score,
            'technical_details': technical_details,
            'structural_score': structural_score,
            'domain_score': domain_score,
            'domain_details': domain_details,
            'final_score': final_score,
            'confidence': self._calculate_confidence(
                semantic_score, keyword_score, technical_score
            )
        }
        
        return final_score, match_details
    
    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Base semantic similarity using embeddings"""
        emb1 = self.model.encode(text1)
        emb2 = self.model.encode(text2)
        
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return max(0.0, min(1.0, float(similarity)))
    
    def _calculate_keyword_overlap(self, text1: str, text2: str) -> Tuple[float, Dict]:
        """Calculate meaningful keyword overlap"""
        words1 = set(self._extract_keywords(text1.lower()))
        words2 = set(self._extract_keywords(text2.lower()))
        
        if not words1 or not words2:
            return 0.0, {'common_words': [], 'word_count': 0}
        
        common = words1 & words2
        union = words1 | words2
        
        # Jaccard similarity
        jaccard = len(common) / len(union) if union else 0.0
        
        # Bonus for important keywords
        important_matches = []
        for category, keywords in self.domain_keywords.items():
            category_common = [w for w in common if any(kw in w for kw in keywords)]
            if category_common:
                important_matches.extend(category_common)
        
        importance_bonus = min(0.3, len(important_matches) * 0.1)
        
        return min(1.0, jaccard + importance_bonus), {
            'common_words': list(common)[:10],
            'word_count': len(common),
            'important_matches': important_matches[:5],
            'jaccard': jaccard
        }
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords (remove stopwords)"""
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are',
            'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'should', 'could', 'may', 'might', 'must',
            'can', 'shall', 'this', 'that', 'these', 'those', 'it', 'its'
        }
        
        words = re.findall(r'\b[a-z]{3,}\b', text)  # 3+ letter words
        return [w for w in words if w not in stopwords]
    
    def _calculate_technical_similarity(self, text1: str, text2: str) -> Tuple[float, Dict]:
        """Match technical specifications and parameters"""
        matches = []
        
        for pattern in self.technical_patterns:
            values1 = set(re.findall(pattern, text1.lower()))
            values2 = set(re.findall(pattern, text2.lower()))
            
            if values1 and values2:
                common = values1 & values2
                if common:
                    matches.extend(list(common))
        
        # Check for specific equipment/component mentions
        equipment_matches = []
        for category, keywords in self.domain_keywords.items():
            if category in ['equipment', 'process', 'specifications']:
                for kw in keywords:
                    if kw in text1.lower() and kw in text2.lower():
                        equipment_matches.append(kw)
        
        score = 0.0
        if matches:
            score += 0.5
        if equipment_matches:
            score += min(0.5, len(equipment_matches) * 0.15)
        
        return min(1.0, score), {
            'technical_matches': matches[:5],
            'equipment_matches': equipment_matches[:5]
        }
    
    def _calculate_structural_similarity(self, text1: str, text2: str) -> float:
        """Compare structural patterns (length, format, etc.)"""
        len1, len2 = len(text1), len(text2)
        
        # Similar length bonus
        len_ratio = min(len1, len2) / max(len1, len2) if max(len1, len2) > 0 else 0
        
        # Both have numbers
        has_numbers1 = bool(re.search(r'\d', text1))
        has_numbers2 = bool(re.search(r'\d', text2))
        numbers_match = 0.3 if (has_numbers1 == has_numbers2) else 0.0
        
        return len_ratio * 0.7 + numbers_match
    
    def _calculate_domain_relevance(self, text1: str, text2: str, 
                                   category: str = None) -> Tuple[float, Dict]:
        """Check domain-specific relevance"""
        text1_lower = text1.lower()
        text2_lower = text2.lower()
        
        domain_matches = {}
        total_score = 0.0
        
        for category, keywords in self.domain_keywords.items():
            matches1 = [kw for kw in keywords if kw in text1_lower]
            matches2 = [kw for kw in keywords if kw in text2_lower]
            
            common = set(matches1) & set(matches2)
            if common:
                domain_matches[category] = list(common)
                total_score += len(common) * 0.15
        
        return min(1.0, total_score), domain_matches
    
    def _has_contradictory_terms(self, text1: str, text2: str) -> bool:
        """Detect contradictory or opposite terms"""
        opposites = [
            ('metallic', 'non-metallic'),
            ('manual', 'automatic'),
            ('required', 'optional'),
            ('included', 'excluded'),
            ('approved', 'rejected'),
            ('pass', 'fail')
        ]
        
        text1_lower = text1.lower()
        text2_lower = text2.lower()
        
        for term1, term2 in opposites:
            if (term1 in text1_lower and term2 in text2_lower) or \
               (term2 in text1_lower and term1 in text2_lower):
                return True
        
        return False
    
    def _calculate_confidence(self, semantic: float, keyword: float, 
                            technical: float) -> str:
        """Calculate confidence level"""
        avg = (semantic + keyword + technical) / 3
        
        if avg >= 0.75 and keyword >= 0.3:
            return "high"
        elif avg >= 0.5 and keyword >= 0.2:
            return "medium"
        else:
            return "low"


# Integration with your existing code
def get_enhanced_matcher():
    """Singleton instance of enhanced matcher"""
    global _ENHANCED_MATCHER
    try:
        _ENHANCED_MATCHER
    except NameError:
        _ENHANCED_MATCHER = EnhancedSemanticMatcher()
    return _ENHANCED_MATCHER


def match_requirements_enhanced(new_requirement: str, 
                               historical_requirements: List[Dict],
                               min_score: float = 0.6) -> List[Dict]:
    """
    Enhanced matching function for requirements
    
    Args:
        new_requirement: New requirement text
        historical_requirements: List of historical requirement dicts
        min_score: Minimum matching score (default 0.6 for high quality)
    
    Returns:
        List of matches sorted by score
    """
    matcher = get_enhanced_matcher()
    matches = []
    
    for hist_req in historical_requirements:
        hist_text = hist_req.get('requirement', '')
        
        if not hist_text:
            continue
        
        score, details = matcher.enhanced_match(
            new_requirement, 
            hist_text,
            category_hint=hist_req.get('category')
        )
        
        if score >= min_score:
            matches.append({
                'requirement': hist_text,
                'document_name': hist_req.get('document_name', ''),
                'comments': hist_req.get('comments', ''),
                'match_score': score,
                'match_details': details,
                'confidence': details['confidence'],
                'keyword_overlap': details['keyword_details']['word_count'],
                'common_keywords': details['keyword_details']['common_words']
            })
    
    # Sort by score (highest first)
    matches.sort(key=lambda x: x['match_score'], reverse=True)
    
    return matches
