
import re
from typing import List
from concurrent.futures import ThreadPoolExecutor
import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
from nltk.tokenize import sent_tokenize

# More specific patterns to reduce false positives
REQUIREMENT_PATTERNS = [
    r'\b(shall|must|should|will)\b',  # Core modal verbs
    r'\b(is\s+required\s+to|are\s+required\s+to)\b', # "is required to"
    r'\b(needs\s+to)\b', # "needs to"
]

# Patterns for lines that are likely requirements (e.g., in lists)
LINE_PATTERNS = [
    r'^\s*[\*\-]\s+',  # Starts with * or - (bullet points)
    r'^\s*\d+\.\s+',  # Starts with "1.", "2.", etc.
]

def is_requirement(text: str) -> bool:
    """Check if a string contains requirement keywords."""
    lower_text = text.lower()
    for pattern in REQUIREMENT_PATTERNS:
        if re.search(pattern, lower_text):
            return True
    return False

def process_chunk(chunk: str) -> str:
    """Process a single sentence or line to check if it's a requirement."""
    chunk = chunk.strip()
    if len(chunk) < 15:  # Skip very short lines
        return None
        
    # Check for requirement patterns
    if is_requirement(chunk):
        return chunk
        
    return None

def rule_based_requirements(text: str, max_workers: int = 4) -> List[str]:
    """
    Extract requirements from text using improved logic for sentences and lists.
    """
    if not text:
        return []
        
    requirements = set() # Use a set to avoid duplicates

    # 1. Split text into lines to handle lists effectively
    lines = text.split('\n')
    
    # 2. Check individual lines, especially for list items
    for line in lines:
        line = line.strip()
        is_list_item = any(re.search(p, line) for p in LINE_PATTERNS)
        
        if is_list_item and is_requirement(line):
            requirements.add(line)

    # 3. Use NLTK for sentence tokenization on the whole text for prose
    sentences = sent_tokenize(text)
    
    # 4. Process sentences in parallel to find more requirements
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        found_reqs = list(filter(None, executor.map(process_chunk, sentences)))
        for req in found_reqs:
            requirements.add(req)
    
    return sorted(list(requirements))