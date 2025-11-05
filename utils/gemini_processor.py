"""
Gemini Pro integration for intelligent requirement extraction and preprocessing.
This module uses Google's Gemini Pro model to analyze documents and extract
high-quality requirements with better context understanding.
"""

import google.generativeai as genai
import re
import json
from typing import List, Dict, Any, Optional
import logging

# Configure logging to avoid Windows console encoding issues
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', 
                   handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

class GeminiProcessor:
    """
    Processes documents using Gemini Pro for intelligent requirement extraction.
    """
    
    def __init__(self, api_key: str, model_name: str = "models/gemini-2.5-flash"):
        """
        Initialize the Gemini processor with API key.
        
        Args:
            api_key: Google AI API key for Gemini Pro
            model_name: Gemini model to use (default: models/gemini-2.5-flash)
        """
        self.api_key = api_key
        self.model_name = model_name
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        
        logger.info(f"Initialized Gemini processor with model: {model_name}")
    
    def extract_requirements_from_text(self, text: str, document_type: str = "technical_specification") -> List[Dict[str, Any]]:
        """
        Extract requirements from text using Gemini Pro's advanced understanding.
        
        Args:
            text: Input text to analyze
            document_type: Type of document for context
            
        Returns:
            List of requirement dictionaries with text, category, confidence, etc.
        """
        if not text or len(text.strip()) < 20:
            return []
        
        # Create a comprehensive prompt for requirement extraction
        prompt = self._create_extraction_prompt(text, document_type)
        
        try:
            # Generate response using Gemini
            response = self.model.generate_content(prompt)
            
            if response.text:
                return self._parse_gemini_response(response.text)
            else:
                logger.warning("Empty response from Gemini")
                return []
                
        except Exception as e:
            logger.error(f"Error calling Gemini API: {str(e)}")
            return []
    
    def _create_extraction_prompt(self, text: str, document_type: str) -> str:
        """Create a detailed prompt for Gemini to extract requirements."""
        
        prompt = f"""
You are an expert technical document analyst specializing in requirement extraction from {document_type} documents.

TASK: Extract ALL requirements, specifications, constraints, and technical details from the following text.

INSTRUCTIONS:
1. Identify requirements using these indicators:
   - Modal verbs: shall, must, should, will, require, need
   - Technical specifications: parameters, limits, ranges, capacities
   - Safety requirements: safety, emergency, protection, compliance
   - Process requirements: procedures, steps, conditions
   - Equipment specifications: pumps, valves, controls, electrical
   - Installation requirements: mounting, connections, access
   - Performance criteria: efficiency, accuracy, response time

2. For each requirement found, extract:
   - The complete requirement text (clean and well-formatted)
   - Category (safety, technical, process, equipment, installation, performance, compliance)
   - Section context (if numbered like 6.1, 6.2, etc.)
   - Priority level (critical, important, standard)

3. Clean up the text by:
   - Removing table prefixes like "3.0 | OVERVIEW |"
   - Fixing formatting issues and line breaks
   - Expanding abbreviations where clear
   - Maintaining technical precision

4. Include requirements that are:
   - Explicit statements with modal verbs
   - Technical parameters and specifications
   - Safety and compliance standards
   - Equipment and system requirements
   - Process and operational procedures

IMPORTANT: 
- Extract even brief technical specifications
- Include numbered requirements (6.1, 6.2, etc.) with their context
- Capture bullet points and sub-requirements
- Don't miss CIP cleaning phases, electrical specifications, or safety requirements

Return the results in this JSON format:
{{
  "requirements": [
    {{
      "text": "Complete requirement text",
      "category": "safety|technical|process|equipment|installation|performance|compliance",
      "section": "section number or context",
      "priority": "critical|important|standard",
      "confidence": 0.95
    }}
  ],
  "total_found": <number>,
  "document_summary": "Brief summary of document content"
}}

TEXT TO ANALYZE:
{text[:4000]}  
"""
        return prompt
    
    def _parse_gemini_response(self, response_text: str) -> List[Dict[str, Any]]:
        """Parse Gemini's JSON response into requirement objects."""
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed = json.loads(json_str)
                
                if "requirements" in parsed and isinstance(parsed["requirements"], list):
                    requirements = []
                    for req in parsed["requirements"]:
                        if isinstance(req, dict) and "text" in req:
                            # Ensure all required fields are present
                            requirement = {
                                "text": req.get("text", "").strip(),
                                "category": req.get("category", "technical"),
                                "section": req.get("section", ""),
                                "priority": req.get("priority", "standard"),
                                "confidence": req.get("confidence", 0.8),
                                "source": "gemini_pro"
                            }
                            if requirement["text"] and len(requirement["text"]) > 8:
                                requirements.append(requirement)
                    
                    logger.info(f"Extracted {len(requirements)} requirements via Gemini")
                    return requirements
            
            # Fallback: try to extract requirements from unstructured text
            return self._extract_from_unstructured_response(response_text)
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            return self._extract_from_unstructured_response(response_text)
        except Exception as e:
            logger.error(f"Error parsing Gemini response: {e}")
            return []
    
    def _extract_from_unstructured_response(self, response_text: str) -> List[Dict[str, Any]]:
        """Extract requirements from unstructured Gemini response as fallback."""
        requirements = []
        
        # Look for numbered requirements or bullet points
        patterns = [
            r'(?:^|\n)\s*(\d+\.?\d*\.?\s+.+?)(?=\n\s*\d+\.|\n\n|$)',  # Numbered requirements
            r'(?:^|\n)\s*[•\-\*]\s*(.+?)(?=\n\s*[•\-\*]|\n\n|$)',     # Bullet points
            r'(?:^|\n)\s*([A-Z][^.\n]*(?:shall|must|should|will|require)[^.\n]*\.?)',  # Modal verb sentences
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response_text, re.MULTILINE | re.IGNORECASE)
            for match in matches:
                text = match.strip()
                if len(text) > 15 and any(kw in text.lower() for kw in ['shall', 'must', 'should', 'will', 'require', 'specification']):
                    requirements.append({
                        "text": text,
                        "category": "technical",
                        "section": "",
                        "priority": "standard",
                        "confidence": 0.7,
                        "source": "gemini_pro_fallback"
                    })
        
        logger.info(f"Fallback extraction found {len(requirements)} requirements")
        return requirements
    
    def enhance_requirement_text(self, requirement_text: str) -> str:
        """
        Use Gemini to enhance and clean up requirement text.
        
        Args:
            requirement_text: Raw requirement text to enhance
            
        Returns:
            Enhanced and cleaned requirement text
        """
        if not requirement_text or len(requirement_text.strip()) < 10:
            return requirement_text
        
        prompt = f"""
Clean and enhance this requirement text for better clarity and completeness:

ORIGINAL: {requirement_text}

INSTRUCTIONS:
1. Remove any table prefixes like "3.0 | OVERVIEW |"
2. Fix formatting and grammar issues
3. Ensure technical terms are properly formatted
4. Make the requirement statement clear and complete
5. Preserve all technical details and specifications
6. Return only the cleaned requirement text, nothing else

ENHANCED REQUIREMENT:
"""
        
        try:
            response = self.model.generate_content(prompt)
            if response.text:
                enhanced = response.text.strip()
                # Basic validation - ensure the enhanced text is reasonable
                if len(enhanced) > 5 and len(enhanced) < len(requirement_text) * 3:
                    return enhanced
            
            return requirement_text  # Return original if enhancement fails
            
        except Exception as e:
            logger.warning(f"Failed to enhance requirement text: {e}")
            return requirement_text
    
    def categorize_requirements(self, requirements: List[str]) -> List[Dict[str, Any]]:
        """
        Use Gemini to categorize a list of requirements.
        
        Args:
            requirements: List of requirement texts
            
        Returns:
            List of categorized requirements with metadata
        """
        if not requirements:
            return []
        
        # Process in batches to avoid token limits
        batch_size = 10
        categorized = []
        
        for i in range(0, len(requirements), batch_size):
            batch = requirements[i:i + batch_size]
            batch_result = self._categorize_batch(batch)
            categorized.extend(batch_result)
        
        return categorized
    
    def _categorize_batch(self, requirement_batch: List[str]) -> List[Dict[str, Any]]:
        """Categorize a batch of requirements."""
        requirements_text = "\n".join([f"{i+1}. {req}" for i, req in enumerate(requirement_batch)])
        
        prompt = f"""
Categorize these requirements into appropriate categories and assign priority levels:

REQUIREMENTS:
{requirements_text}

CATEGORIES:
- safety: Safety, emergency, protection requirements
- technical: Technical specifications, parameters, performance
- process: Process steps, procedures, operations  
- equipment: Equipment specifications, components
- installation: Installation, mounting, connections
- compliance: Standards, regulations, validation
- performance: Performance criteria, efficiency, accuracy

PRIORITIES:
- critical: Safety-critical, mandatory compliance
- important: Significant technical requirements
- standard: Normal operational requirements

Return JSON format:
{{
  "categorized": [
    {{
      "index": 1,
      "text": "requirement text",
      "category": "category",
      "priority": "priority",
      "confidence": 0.95
    }}
  ]
}}
"""
        
        try:
            response = self.model.generate_content(prompt)
            if response.text:
                json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group(0))
                    if "categorized" in parsed:
                        return [{
                            "text": item.get("text", requirement_batch[item.get("index", 1) - 1]),
                            "category": item.get("category", "technical"),
                            "priority": item.get("priority", "standard"),
                            "confidence": item.get("confidence", 0.8),
                            "source": "gemini_pro_categorized"
                        } for item in parsed["categorized"] if "text" in item or "index" in item]
        
        except Exception as e:
            logger.warning(f"Failed to categorize batch: {e}")
        
        # Fallback: return with default categorization
        return [{
            "text": req,
            "category": "technical",
            "priority": "standard", 
            "confidence": 0.6,
            "source": "gemini_pro_fallback"
        } for req in requirement_batch]


def test_gemini_processor(api_key: str, sample_text: str = None):
    """Test function for Gemini processor."""
    if not sample_text:
        sample_text = """
        6.1 CIP unit shall follow the 4 cleaning phases steps as follows:
        - Pre-rinse phase
        - Caustic wash phase  
        - Intermediate rinse phase
        - Sanitization phase

        6.2 All electrical wiring shall be concealed and with proper earthing arrangement.

        7.1 CIP return pump shall have the following specifications:
        Flow rate: 150 L/min minimum
        Head: 25 meters minimum
        Material: 316L stainless steel
        """
    
    processor = GeminiProcessor(api_key)
    requirements = processor.extract_requirements_from_text(sample_text)
    
    print(f"Found {len(requirements)} requirements:")
    for i, req in enumerate(requirements, 1):
        print(f"{i}. [{req['category']}] {req['text']}")
    
    return requirements