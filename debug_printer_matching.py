#!/usr/bin/env python3
"""
Debug specific matching issue: Printer comment not matching audit trail requirement
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.extractors import get_semantic_matcher

def debug_printer_matching():
    """Debug the specific printer matching issue"""
    
    print("🔧 Debugging Printer Comment Matching")
    print("=" * 50)
    
    matcher = get_semantic_matcher()
    
    comment = "Printer in customer scope, integration of customer provided printer with the system will be in GLATT scope"
    requirement = "The system must be able to print an audit trail report in human readable non-modifiable format on demand."
    
    print(f"Comment: {comment}")
    print(f"Requirement: {requirement}")
    print()
    
    # Test direct similarity
    similarity = matcher.calculate_semantic_similarity(comment, requirement)
    print(f"Direct Semantic Similarity: {similarity:.3f}")
    
    # Test keyword similarity fallback
    keyword_sim = matcher._keyword_similarity(comment, requirement)
    print(f"Keyword Similarity: {keyword_sim:.3f}")
    
    # Test with different thresholds
    thresholds = [0.1, 0.15, 0.2, 0.25, 0.3]
    
    print(f"\nTesting different thresholds:")
    for threshold in thresholds:
        matches = matcher.find_best_semantic_matches(comment, [requirement], threshold=threshold)
        if matches:
            print(f"  Threshold {threshold:.2f}: ✅ MATCH - Score: {matches[0]['similarity_score']:.3f}")
        else:
            print(f"  Threshold {threshold:.2f}: ❌ No match")
    
    # Analyze words
    print(f"\nWord Analysis:")
    comment_words = set(comment.lower().split())
    req_words = set(requirement.lower().split())
    common_words = comment_words.intersection(req_words)
    
    print(f"Comment words: {sorted(comment_words)}")
    print(f"Requirement words: {sorted(req_words)}")
    print(f"Common words: {sorted(common_words)}")
    
    # Check terminology mappings
    print(f"\nTerminology Analysis:")
    for concept, synonyms in matcher._terminology_mappings.items():
        comment_has = any(syn in comment.lower() for syn in synonyms)
        req_has = any(syn in requirement.lower() for syn in synonyms)
        
        if comment_has or req_has:
            status = "✅ BOTH" if (comment_has and req_has) else "⚠️ ONE"
            print(f"  {concept}: {status} - Comment: {comment_has}, Requirement: {req_has}")
    
    print(f"\n🎯 Recommendation:")
    if similarity < 0.25:
        print("The semantic similarity is quite low. This suggests:")
        print("1. The embedding model doesn't see strong semantic connection")
        print("2. We should rely more on keyword/terminology matching")
        print("3. Consider adding more specific printer-related terms to terminology")
    else:
        print("Semantic similarity is reasonable, consider lowering threshold")

if __name__ == "__main__":
    debug_printer_matching()