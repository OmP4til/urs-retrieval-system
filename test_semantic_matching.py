#!/usr/bin/env python3
"""
Test script for intelligent semantic text matching
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.extractors import IntelligentTextMatcher, get_semantic_matcher

def test_semantic_matching():
    """Test the intelligent semantic matching functionality"""
    
    print("🧠 Testing Intelligent Semantic Text Matching")
    print("=" * 60)
    
    # Initialize the semantic matcher
    matcher = get_semantic_matcher()
    
    # Test cases demonstrating different terminology matching
    test_cases = [
        {
            "comment": "Test report 2.2 will be provided",
            "requirements": [
                "Material certificates for all metallic/non-metallic product contact parts.",
                "The system must validate user credentials before access.",
                "Documentation shall be submitted for all components.",
                "Verification reports must be available for review.",
                "All metallic components require certification documents."
            ],
            "description": "Your specific use case - test report comment"
        },
        {
            "comment": "Printer in customer scope, integration of customer provided printer with the system will be in GLATT scope",
            "requirements": [
                "The system must be able to prevent unauthorized user access.",
                "The system must be able to print an audit trail report in human readable non-modifiable format on demand.",
                "User authentication shall be implemented with multi-factor verification.",
                "All system events must be logged for audit purposes.",
                "Material certificates for all metallic/non-metallic product contact parts."
            ],
            "description": "Printer scope comment - should match audit trail printing requirement"
        },
        {
            "comment": "Compliance with FDA regulations required",
            "requirements": [
                "The system shall adhere to all applicable regulatory standards.",
                "Performance testing must be conducted annually.",
                "User training documentation must be provided.",
                "All activities must conform to quality standards.",
                "Regulatory compliance verification is mandatory."
            ],
            "description": "Regulatory compliance - different terminology test"
        },
        {
            "comment": "Authentication mechanism needs improvement",
            "requirements": [
                "User login credentials must be verified securely.",
                "Access control shall prevent unauthorized entry.",
                "The system must validate user identity before access.",
                "Password policies must enforce security standards.",
                "Multi-factor verification is required for admin access."
            ],
            "description": "Authentication - synonym matching test"
        }
    ]
    
    print("🔍 Testing Cases:")
    print("-" * 40)
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{i}. {case['description']}")
        print(f"   Comment: '{case['comment']}'")
        print(f"   Requirements to match against:")
        for j, req in enumerate(case['requirements']):
            print(f"     {j+1}. {req}")
        
        # Find semantic matches
        matches = matcher.find_best_semantic_matches(
            case['comment'], 
            case['requirements'],
            threshold=0.2  # Lowered threshold to catch more matches
        )
        
        print(f"\n   🎯 Semantic Matches Found:")
        if matches:
            for match in matches:
                score = match['similarity_score']
                match_type = match['match_type']
                explanation = match['explanation']
                req_text = match['requirement_text'][:80] + "..." if len(match['requirement_text']) > 80 else match['requirement_text']
                
                print(f"     ✅ Score: {score:.3f} | Type: {match_type}")
                print(f"        Requirement: {req_text}")
                print(f"        Explanation: {explanation}")
        else:
            print(f"     ❌ No matches found above threshold")
        
        print("-" * 40)
    
    print("\n" + "=" * 60)
    print("🚀 Key Features of Intelligent Matching:")
    print("• Semantic Understanding: Matches concepts, not just words")
    print("• Domain Terminology: Understands URS-specific vocabulary")
    print("• Synonym Recognition: Maps different terms with same meaning")
    print("• Context Awareness: Considers document context and relationships")
    print("• Confidence Scoring: Provides similarity scores for ranking")
    print("• Multi-Stage Matching: Combines spatial, exact, and semantic matching")
    
    print("\n📊 Matching Stages:")
    print("1. Exact Text Match (Highest Priority)")
    print("2. Sentence-Level Match")
    print("3. Spatial Proximity (PDF annotations)")
    print("4. Semantic Similarity (NEW!)")
    print("5. Broad Semantic Search (Fallback)")
    
    print("\n🎯 Benefits for Your Use Case:")
    print("• 'Test report' → Matches 'verification documents', 'validation reports'")
    print("• 'Certificate' → Matches 'documentation', 'proof', 'validation'")
    print("• 'Material' → Matches 'component', 'part', 'substance'")
    print("• 'Printer scope' → Matches 'print audit trail', 'reporting functionality'")
    print("• 'Compliance' → Matches 'regulatory', 'standard', 'conformity'")
    
    print("\n✨ The system now understands meaning, not just exact words!")

if __name__ == "__main__":
    test_semantic_matching()