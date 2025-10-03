import json
import re

def extract_and_parse_json(text: str, is_batch: bool = True) -> dict:
    """
    Extract and parse JSON from text that might contain other content.
    Handles common JSON parsing issues and cleans the input.
    
    Args:
        text (str): The text to parse
        is_batch (bool): Whether to expect an array of requirements for multiple pages
    """
    if not text:
        return {"requirements": [], "comments": [], "responses": []}
    
    def clean_json_text(json_text: str) -> str:
        """Clean up common JSON formatting issues."""
        # Remove any text before the first { or [ and after the last } or ]
        start = min(
            (json_text.find('{') if '{' in json_text else len(json_text)),
            (json_text.find('[') if '[' in json_text else len(json_text))
        )
        end = max(json_text.rfind('}'), json_text.rfind(']')) + 1
        if start < end:
            json_text = json_text[start:end]
            
        # Handle line breaks and whitespace
        json_text = re.sub(r'\s+', ' ', json_text)
        # Fix common quote issues
        json_text = json_text.replace('"', '"').replace('"', '"').replace("'", '"')
        # Fix trailing commas
        json_text = re.sub(r',\s*([\]}])', r'\1', json_text)
        # Fix missing commas between objects
        json_text = re.sub(r'}\s*{', '},{', json_text)
        json_text = re.sub(r']\s*{', '],{', json_text)
        json_text = re.sub(r'}\s*\[', '},[', json_text)
        return json_text.strip()
    
    # First try to parse as-is
    try:
        cleaned = clean_json_text(text)
        result = json.loads(cleaned)
        if isinstance(result, dict) and "requirements" in result:
            return result
        if isinstance(result, list) and is_batch:
            # Convert batch format to single response format
            all_reqs = []
            for page_reqs in result:
                if isinstance(page_reqs, dict) and "requirements" in page_reqs:
                    all_reqs.extend(page_reqs["requirements"])
                elif isinstance(page_reqs, dict):
                    # Try to convert single requirement into proper format
                    if "text" in page_reqs:
                        all_reqs.append(page_reqs)
            return {"requirements": all_reqs, "comments": [], "responses": []}
    except json.JSONDecodeError as e:
        # Try to find and fix the specific issue
        if "Extra data" in str(e):
            # Try to parse just the first complete JSON structure
            try:
                cleaned = re.search(r'(\{.*?\}|\[.*?\])', text, re.DOTALL)
                if cleaned:
                    return extract_and_parse_json(cleaned.group(1), is_batch)
            except:
                pass
        
        # Try to find any JSON-like structures
        json_patterns = [
            r"\{(?:[^{}]|(?R))*\}",  # Nested JSON object
            r"\[(?:[^\[\]]|(?R))*\]",  # Nested JSON array
            r"\{[^{]*\}",  # Simple JSON object
            r"\[[^\[]*\]"  # Simple JSON array
        ]
        
        for pattern in json_patterns:
            matches = re.finditer(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    potential_json = clean_json_text(match.group(0))
                    result = json.loads(potential_json)
                    
                    if isinstance(result, dict):
                        if "requirements" in result:
                            return result
                        elif "text" in result:
                            # Single requirement format
                            return {"requirements": [result], "comments": [], "responses": []}
                            
                    if isinstance(result, list) and is_batch:
                        all_reqs = []
                        for item in result:
                            if isinstance(item, dict):
                                if "requirements" in item:
                                    all_reqs.extend(item["requirements"])
                                elif "text" in item:
                                    all_reqs.append(item)
                        if all_reqs:
                            return {"requirements": all_reqs, "comments": [], "responses": []}
                except:
                    continue
    
    # If we get here, we couldn't find valid JSON
    return {
        "requirements": [],
        "comments": [],
        "responses": []
    }