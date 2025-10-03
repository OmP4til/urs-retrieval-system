# utils/preprocess.py
import re
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
from nltk.tokenize import sent_tokenize

STOPWORDS = {
    "the","and","for","with","that","this","these","those","are","is","a","an","to","of","in","on","by","be","as","or","it","from","at"
}

REQUIREMENT_PATTERNS = [
    r'\b(shall|must|should|require|required|needs to|will)\b',
    r'\b(mandatory|necessary|essential)\b',
    r'\b(specification|requirement|feature)\b'
]

def extract_keywords(text: str, top_n: int = 20) -> List[str]:
    toks = re.findall(r"\w+", (text or "").lower())
    toks = [t for t in toks if t not in STOPWORDS and len(t) > 2]
    freq = {}
    for t in toks:
        freq[t] = freq.get(t, 0) + 1
    sorted_terms = sorted(freq.keys(), key=lambda k: -freq[k])
    return sorted_terms[:top_n]

def process_sentence(sent: str) -> str:
    """Process a single sentence to check if it's a requirement."""
    sent = sent.strip()
    if len(sent) < 10:  # Skip very short sentences
        return None
        
    # Check for requirement patterns
    for pattern in REQUIREMENT_PATTERNS:
        if re.search(pattern, sent, re.I):
            return sent
    return None

def rule_based_requirements(text: str, max_workers: int = 4) -> List[str]:
    """Extract requirements from text using parallel processing."""
    if not text:
        return []
        
    # Use NLTK for better sentence tokenization
    sentences = sent_tokenize(text)
    
    # Process sentences in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        requirements = list(filter(None, executor.map(process_sentence, sentences)))
    
    return requirements

def intelligent_chunker(page_text: str, structure_hints: Dict = None) -> List[Dict]:
    # Split text into logical sections based on headers or spacing
    sections = re.split(r'\n\s*\n', page_text)
    
    chunks = []
    for section in sections:
        if len(section.strip()) > 0:
            reqs = rule_based_requirements(section)
            if reqs:  # Only include sections with requirements
                chunks.append({
                    "type": "section",
                    "text": section.strip(),
                    "requirements": reqs
                })
    
    return chunks if chunks else [{"type": "page", "text": page_text, "requirements": rule_based_requirements(page_text)}]
