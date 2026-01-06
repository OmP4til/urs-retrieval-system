"""
Deep Semantic Matcher using Gemini for true meaning-based matching
"""

import google.generativeai as genai
from typing import Tuple, Dict, Any
import os

class DeepSemanticMatcher:
    """
    Uses Gemini to understand the deep semantic meaning of requirements
    and determine if they truly mean the same thing, even with different wording
    """
    
    def __init__(self, api_key: str = None):
        if not api_key:
            api_key = os.environ.get("GEMINI_API_KEY")
        
        if not api_key:
            raise ValueError("GEMINI_API_KEY not provided")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        
    def compare_requirements(self, req1: str, req2: str) -> Tuple[float, Dict[str, Any]]:
        """
        Use Gemini to deeply understand if two requirements have the same meaning
        
        Returns:
            (similarity_score, analysis_details)
            similarity_score: 0.0 to 1.0
        """
        
        prompt = f"""You are an expert in pharmaceutical manufacturing requirements analysis.

Compare these two requirements and determine if they have the SAME FUNCTIONAL MEANING, even if worded differently.

Requirement 1: {req1}

Requirement 2: {req2}

Analysis criteria:
1. Do they describe the same functional requirement?
2. Do they have the same intent/purpose?
3. Would satisfying one automatically satisfy the other?
4. Are they just domain-related or truly equivalent?

Respond in this EXACT format:
SIMILARITY_SCORE: [0.0 to 1.0, where 1.0 = identical meaning, 0.0 = completely different]
SAME_MEANING: [YES or NO]
REASONING: [One sentence explaining why they are or aren't the same]

Scoring guide:
- 0.95-1.0: Identical meaning, just different wording
- 0.85-0.94: Very similar, same core requirement with minor variations
- 0.70-0.84: Related but different specific requirements
- 0.50-0.69: Same domain/topic but different requirements
- 0.0-0.49: Different requirements entirely

Be strict - only score high if they truly mean the same thing functionally."""

        try:
            response = self.model.generate_content(prompt)
            text = response.text
            
            # Parse response
            score_line = [l for l in text.split('\n') if 'SIMILARITY_SCORE:' in l]
            same_line = [l for l in text.split('\n') if 'SAME_MEANING:' in l]
            reason_line = [l for l in text.split('\n') if 'REASONING:' in l]
            
            score = 0.0
            same_meaning = "NO"
            reasoning = "Parse error"
            
            if score_line:
                score_text = score_line[0].split('SIMILARITY_SCORE:')[1].strip()
                score = float(score_text)
            
            if same_line:
                same_meaning = same_line[0].split('SAME_MEANING:')[1].strip()
            
            if reason_line:
                reasoning = reason_line[0].split('REASONING:')[1].strip()
            
            return score, {
                'same_meaning': same_meaning,
                'reasoning': reasoning,
                'confidence': 'high',
                'method': 'gemini_deep_semantic'
            }
            
        except Exception as e:
            print(f"Gemini comparison error: {e}")
            return 0.0, {
                'same_meaning': 'UNKNOWN',
                'reasoning': f'Error: {str(e)}',
                'confidence': 'none',
                'method': 'error'
            }
