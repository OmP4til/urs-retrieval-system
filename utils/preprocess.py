
import re
from typing import List, Dict
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
    """Check if a string contains requirement keywords or technical specifications."""
    lower_text = text.lower()
    
    # Modal verbs (strong indicators)
    for pattern in REQUIREMENT_PATTERNS:
        if re.search(pattern, lower_text):
            return True
    
    # Technical specifications and parameters
    tech_terms = [
        'temperature', 'pressure', 'speed', 'flow', 'capacity', 'voltage', 'frequency',
        'bar', '°c', 'rpm', 'm³/h', 'kg', 'amperage', 'kw', 'level', 'alarm', 'trip',
        'control', 'pump', 'valve', 'filter', 'exhaust', 'inlet', 'outlet', 'spray',
        'air', 'water', 'steam', 'electrical', 'wiring', 'phase', 'volts', 'hz'
    ]
    if any(term in lower_text for term in tech_terms):
        return True
    
    # Safety and compliance terms
    safety_terms = [
        'safety', 'emergency', 'stop', 'e-stop', 'earthing', 'grounding', 'interlocking',
        'guarded', 'concealed', 'noise', 'ergonomic', 'training', 'compliance', 'cgmp',
        'validation', 'moving parts', 'mechanism', 'provided by vendor'
    ]
    if any(term in lower_text for term in safety_terms):
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
    """Extract requirements from text: supports bullets/numbered lists and sentences with modal keywords."""
    if not text:
        return []

    results: List[str] = []
    seen = set()

    # Pattern 1: Bullet points or numbered lists (multiline)
    bullet_pattern = r'(?:^|\n)[\s]*(?:[•\-*]|\d+\.)\s*(.+?)(?=(?:\n[\s]*(?:[•\-*]|\d+\.))|$)'
    bullets = re.findall(bullet_pattern, text, re.DOTALL | re.MULTILINE)
    for b in bullets:
        cleaned = b.strip()
        if len(cleaned) > 15 and any(kw in cleaned.lower() for kw in ['shall','must','should','will','require','required']):
            n = cleaned.lower()
            if n not in seen:
                seen.add(n)
                results.append(cleaned)

    # Pattern 2: Sentences with requirement keywords
    sentences = sent_tokenize(text)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        found = list(filter(None, executor.map(process_chunk, sentences)))
    for s in found:
        n = s.lower()
        if n not in seen:
            seen.add(n)
            results.append(s)

    return results


# ---------------- Enhanced Normalization and Comprehensive Extraction ----------------

_PREFIX_RE = re.compile(
    r'^(?:\s*\d+(?:\.\d+)*\s*\|\s*[A-Z0-9 /&\-]+(?:\s*\|\s*[A-Z0-9 /&\-]+)*)\s+',
)

def _strip_table_prefixes(line: str) -> str:
    """Remove leading table section prefixes like '3.0 | OVERVIEW | OVERVIEW | ' from a line."""
    s = line
    # Remove at most once (patterns typically encompass repeated pipes already)
    m = _PREFIX_RE.match(s)
    if m:
        s = s[m.end():]
    return s.strip()


def _split_list_like(text: str) -> List[str]:
    """Split long requirement lines on strong separators into smaller atomic items.

    Splits on semicolons and em/en dashes with surrounding spaces. Avoid splitting hyphenated words.
    """
    if not text or len(text) < 120:
        return [text.strip()]
    parts = re.split(r'\s*;\s+|\s*[\u2013\u2014]\s+', text)
    parts = [p.strip() for p in parts if p and len(p.strip()) >= 8]
    # If splitting yields too few or no modal verbs, keep original
    has_modal = lambda s: any(k in s.lower() for k in [' shall ', ' must ', ' should ', ' will ', ' required', ' require '])
    if parts and (sum(1 for p in parts if has_modal(' '+p+' ')) >= 1):
        return parts
    return [text.strip()]


def _normalize_text_lines(text: str) -> str:
    """Normalize each line: strip table prefixes; keep original line breaks."""
    lines = text.splitlines()
    norm = []
    for ln in lines:
        ln = ln.rstrip()
        if not ln.strip():
            norm.append(ln)
            continue
        norm.append(_strip_table_prefixes(ln))
    return "\n".join(norm)


def enhanced_rule_based_requirements(text: str, max_workers: int = 4) -> List[str]:
    """Stronger extractor: normalize prefixes, extract bullets and sentences, split long list-like items."""
    if not text:
        return []

    text = _normalize_text_lines(text)

    results: List[str] = []
    seen = set()

    # 1) Bullets/numbered items (more variants, including parentheses like 1) )
    bullet_pattern = r'(?:^|\n)[\s]*(?:[•\-*]|\d+\.|\d+\))\s*(.+?)(?=(?:\n[\s]*(?:[•\-*]|\d+\.|\d+\)))|$)'
    bullets = re.findall(bullet_pattern, text, re.DOTALL | re.MULTILINE)
    for b in bullets:
        items = _split_list_like(b.strip())
        for it in items:
            if len(it) < 12:
                continue
            if is_requirement(' '+it+' '):
                key = it.lower()
                if key not in seen:
                    seen.add(key)
                    results.append(it)

    # 2) Sentences with modal verbs
    sentences = sent_tokenize(text)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        candidate_sentences = list(filter(None, executor.map(process_chunk, sentences)))
    for s in candidate_sentences:
        # Further split long sentences into atomic sub-reqs
        for it in _split_list_like(s):
            if not it:
                continue
            key = it.lower()
            if key not in seen and is_requirement(' '+it+' '):
                seen.add(key)
                results.append(it)

    return results


def extract_all_requirements(text: str, max_workers: int = 4) -> List[str]:
    """Comprehensive extraction combining normalization, bullets, sentences, and list-like splitting."""
    if not text:
        return []

    text = _normalize_text_lines(text)
    results: List[str] = []
    seen = set()

    # Bullets first (high precision)
    bullet_pattern = r'(?:^|\n)[\s]*(?:[•\-*]|\d+\.|\d+\))\s*(.+?)(?=(?:\n[\s]*(?:[•\-*]|\d+\.|\d+\)))|$)'
    bullets = re.findall(bullet_pattern, text, re.DOTALL | re.MULTILINE)
    for b in bullets:
        for it in _split_list_like(b.strip()):
            if len(it) >= 12 and is_requirement(' '+it+' '):
                k = it.lower()
                if k not in seen:
                    seen.add(k)
                    results.append(it)

    # Sentences next
    sentences = sent_tokenize(text)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        candidate_sentences = list(filter(None, executor.map(process_chunk, sentences)))
    for s in candidate_sentences:
        for it in _split_list_like(s):
            if len(it) >= 12 and is_requirement(' '+it+' '):
                k = it.lower()
                if k not in seen:
                    seen.add(k)
                    results.append(it)

    return results


def count_all_requirements(pages: List[Dict]) -> int:
    """Count potential requirements across a list of extracted page dicts."""
    if not pages:
        return 0
    total = 0
    for p in pages:
        txt = p.get('content', '') or ''
        if not txt.strip():
            continue
        total += len(extract_all_requirements(txt))
    return total