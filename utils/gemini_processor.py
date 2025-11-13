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
    
    def extract_requirements_holistically(self, full_document_text: str, document_name: str = "Technical Document") -> List[Dict[str, Any]]:
        """
        Let Gemini 2.5 Flash analyze the entire document holistically and extract requirements 
        in its own intelligent way without structural restrictions.
        
        Args:
            full_document_text: Complete document text
            document_name: Name of the document for context
            
        Returns:
            List of intelligently extracted requirements with rich metadata
        """
        
        prompt = f"""
        You are an expert technical document analyst. Please analyze this complete {document_name} and extract ALL requirements in the most intelligent and comprehensive way possible.

        DOCUMENT TO ANALYZE:
        {full_document_text}

        INSTRUCTIONS:
        1. Read and understand the ENTIRE document context
        2. Extract EVERY requirement, specification, constraint, or obligation
        3. For each requirement, provide:
           - Clear, standalone requirement text
           - Intelligent categorization (safety, performance, functional, design, compliance, etc.)
           - Confidence level (0.0-1.0)
           - Source context/section where found
           - Priority level (critical, high, medium, low)
           - Dependencies or relationships to other requirements
           - Any specific technical parameters or values
           
        4. Don't limit yourself to obvious "shall" statements - include:
           - Design specifications
           - Performance criteria
           - Safety requirements
           - Compliance standards
           - Operational requirements
           - Maintenance requirements
           - Environmental conditions
           - Quality standards
           
        5. Use your intelligence to understand implicit requirements from context
        6. Group related requirements logically
        7. Identify any conflicts or ambiguities

        OUTPUT FORMAT: JSON array with this structure for each requirement:
        {{
            "id": "auto-generated unique ID",
            "text": "Clear, standalone requirement statement",
            "category": "intelligent category",
            "subcategory": "more specific classification",
            "confidence": 0.95,
            "source_context": "where in document this was found",
            "priority": "critical/high/medium/low",
            "technical_parameters": {{"any specific values or ranges"}},
            "dependencies": ["list of related requirement IDs"],
            "compliance_standards": ["relevant standards mentioned"],
            "notes": "any additional context or interpretation"
        }}

        
        Be as comprehensive and intelligent as possible. Extract everything that could be considered a requirement or specification.
        """
        
        if not full_document_text or len(full_document_text.strip()) < 50:
            logger.warning("Document text too short for analysis")
            return []
        
        try:
            # Use Gemini's full potential with a large context window
            logger.info(f"Analyzing document with Gemini 2.5 Flash - {len(full_document_text)} characters")
            
            response = self.model.generate_content(prompt)
            
            if response.text:
                logger.info("Received response from Gemini, parsing requirements...")
                return self._parse_holistic_response(response.text)
            else:
                logger.warning("Empty response from Gemini")
                return []
                
        except Exception as e:
            logger.error(f"Error in holistic extraction: {str(e)}")
            return []
    
    def _parse_holistic_response(self, response_text: str) -> List[Dict[str, Any]]:
        """
        Parse Gemini's holistic response into structured requirements.
        """
        try:
            # Try to extract JSON from the response
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                requirements = json.loads(json_str)
                
                # Validate and enhance each requirement
                processed_requirements = []
                for idx, req in enumerate(requirements):
                    if isinstance(req, dict) and 'text' in req:
                        # Ensure all required fields exist
                        processed_req = {
                            'id': req.get('id', f'GEMINI_REQ_{idx+1:03d}'),
                            'text': req.get('text', '').strip(),
                            'category': req.get('category', 'general'),
                            'subcategory': req.get('subcategory', ''),
                            'confidence': float(req.get('confidence', 0.8)),
                            'source_context': req.get('source_context', ''),
                            'priority': req.get('priority', 'medium'),
                            'technical_parameters': req.get('technical_parameters', {}),
                            'dependencies': req.get('dependencies', []),
                            'compliance_standards': req.get('compliance_standards', []),
                            'notes': req.get('notes', ''),
                            'extraction_method': 'gemini_holistic'
                        }
                        
                        if processed_req['text']:  # Only add if text is not empty
                            processed_requirements.append(processed_req)
                
                logger.info(f"Successfully parsed {len(processed_requirements)} requirements from Gemini response")
                return processed_requirements
            
            else:
                logger.warning("No JSON found in Gemini response, attempting text parsing...")
                return self._fallback_text_parsing(response_text)
                
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parsing failed: {e}, attempting fallback parsing...")
            return self._fallback_text_parsing(response_text)
        except Exception as e:
            logger.error(f"Error parsing holistic response: {e}")
            return []
    
    def _fallback_text_parsing(self, response_text: str) -> List[Dict[str, Any]]:
        """
        Fallback method to extract requirements from text response if JSON parsing fails.
        """
        requirements = []
        lines = response_text.split('\n')
        current_req = {}
        req_counter = 1
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Look for requirement patterns
            if line.lower().startswith(('requirement:', 'req:', '-')) or 'shall' in line.lower():
                if current_req.get('text'):
                    # Save previous requirement
                    current_req['id'] = f'GEMINI_REQ_{req_counter:03d}'
                    requirements.append(current_req)
                    req_counter += 1
                
                # Start new requirement
                current_req = {
                    'text': line.replace('Requirement:', '').replace('Req:', '').replace('-', '').strip(),
                    'category': 'general',
                    'confidence': 0.7,
                    'extraction_method': 'gemini_fallback'
                }
        
        # Add the last requirement
        if current_req.get('text'):
            current_req['id'] = f'GEMINI_REQ_{req_counter:03d}'
            requirements.append(current_req)
        
        logger.info(f"Fallback parsing extracted {len(requirements)} requirements")
        return requirements
    
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
    
    def extract_comments_and_responses(self, full_document_text: str, document_name: str = "Document") -> List[Dict[str, Any]]:
        """
        Extract comments and responses from documents where users have marked up requirements.
        This identifies original requirements and associated comments/responses with their authors.
        
        Args:
            full_document_text: Complete document text
            document_name: Name of the document for context
            
        Returns:
            List of requirements with their associated comments and responses
        """
        
        # Enhanced prompt for comment extraction
        comment_extraction_prompt = f"""
        You are analyzing a technical document that contains requirements and comments/responses from reviewers.
        
        Document: {document_name}
        
        Your task is to identify:
        1. Original requirements (technical specifications, functional needs, etc.)
        2. Comments/responses/replies associated with each requirement
        3. Author/source of each comment (company name, person name, role, etc.)
        
        Common patterns to look for:
        - Requirements followed by responses in different formatting
        - Comments in margins, callouts, or highlighted text
        - Tracked changes or revision marks
        - Author signatures like "GLATT:", "John Smith:", "Engineering Team:", etc.
        - Response patterns like "will provide", "offering", "suggests", "recommends"
        
        For each requirement with comments, extract:
        - requirement_text: The original requirement
        - comments: List of comments/responses with their details
        
        Return ONLY a valid JSON array in this exact format:
        [
            {{
                "id": "REQ_001",
                "requirement_text": "Original requirement text",
                "comments": [
                    {{
                        "comment_text": "The actual comment or response text",
                        "author": "Who made the comment (name, company, role)",
                        "comment_type": "response|suggestion|clarification|objection"
                    }}
                ],
                "page_reference": "Page number or section if identifiable"
            }}
        ]
        
        If no comments are found, return an empty array: []
        
        Document text:
        {full_document_text[:15000]}  # Limit to avoid token limits
        """
        
        try:
            logger.info(f"Extracting comments and responses from document: {document_name}")
            
            # Generate response using Gemini
            response = self.model.generate_content(comment_extraction_prompt)
            
            if not response or not response.text:
                logger.warning("No response from Gemini for comment extraction")
                return []
            
            # Clean and parse JSON
            response_text = response.text.strip()
            
            # Remove markdown code blocks if present
            if response_text.startswith('```'):
                response_text = re.sub(r'^```(?:json)?\s*', '', response_text)
                response_text = re.sub(r'\s*```$', '', response_text)
            
            # Parse JSON
            try:
                comments_data = json.loads(response_text)
                
                if not isinstance(comments_data, list):
                    logger.error("Response is not a list")
                    return []
                
                logger.info(f"Successfully extracted {len(comments_data)} requirements with comments")
                
                # Validate and enhance each entry
                validated_comments = []
                for i, item in enumerate(comments_data):
                    if isinstance(item, dict) and 'requirement_text' in item:
                        # Ensure required fields
                        validated_item = {
                            'id': item.get('id', f"COMMENT_REQ_{i+1}"),
                            'requirement_text': item.get('requirement_text', '').strip(),
                            'comments': item.get('comments', []),
                            'page_reference': item.get('page_reference', 'Unknown'),
                            'extracted_by': 'gemini_comment_extractor',
                            'document_name': document_name
                        }
                        
                        # Validate comments structure
                        validated_comments_list = []
                        for comment in validated_item.get('comments', []):
                            if isinstance(comment, dict):
                                validated_comment = {
                                    'comment_text': comment.get('comment_text', '').strip(),
                                    'author': comment.get('author', 'Unknown').strip(),
                                    'comment_type': comment.get('comment_type', 'response').strip()
                                }
                                if validated_comment['comment_text']:  # Only add if has actual comment text
                                    validated_comments_list.append(validated_comment)
                        
                        validated_item['comments'] = validated_comments_list
                        
                        # Only add if has requirement text
                        if validated_item['requirement_text']:
                            validated_comments.append(validated_item)
                
                return validated_comments
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error: {e}")
                logger.error(f"Raw response: {response_text[:500]}")
                return []
                
        except Exception as e:
            logger.error(f"Error extracting comments with Gemini: {e}")
            return []

    def extract_requirements_with_comments_holistically(self, full_document_text: str, document_name: str = "Document", file_bytes: bytes = None) -> Dict[str, Any]:
        """
        Enhanced holistic extraction that extracts both requirements AND their associated comments.
        This integrates both requirement extraction and comment extraction in one comprehensive analysis.
        
        Args:
            full_document_text: Complete document text
            document_name: Name of the document
            file_bytes: Original file bytes for comment parsing (optional)
            
        Returns:
            Dict containing both requirements and comment mappings
        """
        logger.info(f"Starting comprehensive extraction (requirements + comments) for: {document_name}")
        
        # Step 1: Extract structured comments from DOCX if file_bytes provided
        structured_comments = {}
        if file_bytes and document_name.lower().endswith('.docx'):
            try:
                from utils.extractors import get_docx_comments_with_text_mapping
                structured_comments = get_docx_comments_with_text_mapping(file_bytes)
                logger.info(f"Found {len(structured_comments)} text segments with comments in DOCX")
            except Exception as e:
                logger.warning(f"Could not extract structured comments: {e}")
        
        # Step 2: Use Gemini for comprehensive analysis including comment detection
        enhanced_prompt = f"""
        You are an expert technical document analyst. Analyze this complete {document_name} and perform precise requirement-comment extraction with EXACT PAIRING.

        CRITICAL TASK: Extract requirements and their SPECIFIC associated comments with precise linking.

        DOCUMENT TO ANALYZE:
        {full_document_text}

        EXTRACTION RULES:
        1. **REQUIREMENT IDENTIFICATION**: Look for specifications, constraints, obligations, functional needs
        2. **COMMENT IDENTIFICATION**: Find responses, feedback, clarifications that relate to specific requirements
        3. **PRECISE LINKING**: Each comment must be linked to its exact associated requirement
        4. **AUTHOR DETECTION**: Identify who made each comment (look for signatures like "GLATT:", "Engineering:", etc.)
        5. **SPATIAL PROXIMITY**: Comments usually appear near their related requirements in the document

        COMMENT PATTERNS TO LOOK FOR:
        - Company responses: "GLATT will provide...", "Engineering confirms..."  
        - Vendor feedback: "We offer...", "Available as standard...", "Optional feature..."
        - Technical clarifications: "This means...", "Specification details..."
        - Status updates: "Completed", "In progress", "Not applicable"
        - Questions/concerns: "Need clarification on...", "Issue with..."

        OUTPUT FORMAT: Return a JSON object with EXACT requirement-comment pairing:
        {{
            "requirements": [
                {{
                    "id": "REQ_001",
                    "text": "Exact requirement text from document", 
                    "category": "functional/safety/performance/interface/data/etc",
                    "confidence": 0.95,
                    "source_context": "Document section where this requirement appears",
                    "priority": "critical/high/medium/low",
                    "specific_comments": [
                        {{
                            "comment_text": "Exact comment text that relates to THIS requirement",
                            "author": "Who made this comment",  
                            "comment_type": "vendor_response/clarification/confirmation/objection",
                            "confidence": 0.9
                        }}
                    ]
                }},
                {{
                    "id": "REQ_002", 
                    "text": "Another requirement",
                    "category": "performance",
                    "confidence": 0.92,
                    "source_context": "Section B.2",
                    "priority": "high",
                    "specific_comments": []  // This requirement has no associated comments
                }}
            ]
        }}

        CRITICAL SUCCESS FACTORS:
        - Each requirement gets ONLY its own associated comments (not all comments)
        - If a requirement has no comments, its "specific_comments" array should be empty []
        - Comments must be precisely matched to their requirements based on document context and proximity
        - Don't assign the same comment to multiple requirements unless it truly applies to both
        - Be very careful about comment-requirement relationships - accuracy is more important than completeness

        Analyze the document carefully and provide precise requirement-comment pairing.
        """
        
        try:
            logger.info(f"Performing comprehensive Gemini analysis - {len(full_document_text)} characters")
            
            response = self.model.generate_content(enhanced_prompt)
            
            if not response or not response.text:
                logger.warning("No response from Gemini for comprehensive extraction")
                return {"requirements": [], "comments": [], "comment_mappings": {}}
            
            # Parse the comprehensive response
            response_text = response.text.strip()
            
            # Clean markdown if present
            if response_text.startswith('```'):
                response_text = re.sub(r'^```(?:json)?\s*', '', response_text)
                response_text = re.sub(r'\s*```$', '', response_text)
            
            try:
                comprehensive_data = json.loads(response_text)
                
                # Validate structure and extract data
                requirements = comprehensive_data.get('requirements', [])
                
                # Count total comments from all requirements
                total_comments = 0
                requirement_comment_pairs = []
                
                for req in requirements:
                    req_comments = req.get('specific_comments', [])
                    total_comments += len(req_comments)
                    
                    # Create requirement-comment pairs for database storage
                    requirement_comment_pairs.append({
                        'requirement': req,
                        'comments': req_comments
                    })
                
                logger.info(f"Extracted {len(requirements)} requirements with {total_comments} total associated comments")
                
                # Process structured comments from DOCX if available  
                if structured_comments:
                    logger.info("Merging with structured DOCX comments...")
                    # Try to match structured comments to requirements based on text proximity
                    for req_pair in requirement_comment_pairs:
                        req_text = req_pair['requirement'].get('text', '').lower()
                        
                        # Find DOCX comments that might relate to this requirement
                        for text, docx_comments in structured_comments.items():
                            # Simple text matching - could be improved with better NLP
                            if any(word in text.lower() for word in req_text.split()[:5]):  # Check first 5 words
                                for docx_comment in docx_comments:
                                    req_pair['comments'].append({
                                        'comment_text': docx_comment['text'],
                                        'author': docx_comment.get('author', 'Unknown'),
                                        'comment_type': 'docx_structured',
                                        'confidence': 0.7,  # Lower confidence for auto-matched
                                        'source': 'docx_structured'
                                    })
                
                return {
                    "requirements": requirements,  # Keep original for compatibility
                    "requirement_comment_pairs": requirement_comment_pairs,  # New structure with precise pairing
                    "total_comments": total_comments,
                    "comment_mappings": structured_comments,
                    "extraction_method": "gemini_comprehensive_with_precise_pairing",
                    "document_name": document_name
                }
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error in comprehensive extraction: {e}")
                logger.error(f"Raw response: {response_text[:500]}...")
                
                # Fallback to regular requirement extraction
                logger.info("Falling back to regular requirement extraction")
                requirements = self.extract_requirements_holistically(full_document_text, document_name)
                return {
                    "requirements": requirements,
                    "comments": [],
                    "comment_mappings": structured_comments,
                    "extraction_method": "fallback_requirements_only"
                }
                
        except Exception as e:
            logger.error(f"Error in comprehensive extraction: {e}")
            # Fallback to regular extraction
            requirements = self.extract_requirements_holistically(full_document_text, document_name)
            return {
                "requirements": requirements,
                "comments": [],
                "comment_mappings": {},
                "extraction_method": "error_fallback"
            }


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