
import io
import re
from collections import defaultdict
from typing import List, Dict, Any
from datetime import datetime
import pdfplumber
from docx import Document
from lxml import etree
import zipfile
import openpyxl

def _extract_table_rows(table, _depth: int = 0) -> List[str]:
    """
    Flatten a python-docx table into "cell | cell | cell" lines.

    Recurses into tables nested inside cells, which python-docx does not expose
    through doc.tables and does not include in cell.text.

    Horizontally merged cells appear once per underlying grid column in
    row.cells, so consecutive repeats of the same text are collapsed.
    """
    if _depth > 10:          # cyclic or pathological nesting guard
        return []

    lines = []
    for row in table.rows:
        try:
            cells = list(row.cells)
        except (IndexError, ValueError):
            # Malformed grid (irregular spans); skip the row rather than fail.
            continue

        row_text = []
        nested_lines = []
        seen_tc = set()
        for cell in cells:
            # A merged cell is returned once per grid column it spans, and each
            # repeat is the same underlying <w:tc>. Process it once, or its
            # nested tables get emitted once per span.
            tc_id = id(cell._tc)
            if tc_id in seen_tc:
                continue
            seen_tc.add(tc_id)

            text = cell.text.strip()
            if text and (not row_text or row_text[-1] != text):
                row_text.append(text)
            for nested in cell.tables:
                nested_lines.extend(_extract_table_rows(nested, _depth + 1))

        if row_text:
            lines.append(" | ".join(row_text))
        lines.extend(nested_lines)

    return lines


def extract_text_from_docx(uploaded_file) -> str:
    """
    Extract complete text content from a DOCX file for holistic analysis.
    
    Args:
        uploaded_file: Streamlit uploaded file object or file-like object
        
    Returns:
        Complete text content of the document
    """
    try:
        # Read file bytes
        if hasattr(uploaded_file, 'read'):
            file_bytes = uploaded_file.read()
            uploaded_file.seek(0)
        else:
            with open(uploaded_file, 'rb') as f:
                file_bytes = f.read()
        
        # Load document
        doc = Document(io.BytesIO(file_bytes))

        full_text = []

        # Extract all paragraph text
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                full_text.append(paragraph.text.strip())

        # Extract all table text, recursing into nested tables.
        #
        # doc.tables only lists top-level tables, and cell.text covers the
        # cell's own paragraphs but not tables nested inside it. URS documents
        # are frequently one outer table per section with the real requirements
        # in an inner table, so without this recursion a large share of the
        # document is silently dropped.
        for table in doc.tables:
            full_text.extend(_extract_table_rows(table))

        return "\n\n".join(full_text)
        
    except Exception as e:
        print(f"Error extracting text from DOCX: {e}")
        return ""


OOXML_NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}

# Note: The following imports have external dependencies that must be installed.
# For OCR (PDFs with images):
# - pytesseract: pip install pytesseract
# - pdf2image: pip install pdf2image
# - Google Tesseract: Must be installed on the system (e.g., via brew, apt-get)
# - Poppler: Required by pdf2image (e.g., via brew, apt-get)
try:
    import pytesseract
    from pdf2image import convert_from_bytes
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# For PDF annotations and comments:
# - PyMuPDF: pip install PyMuPDF
try:
    import fitz  # PyMuPDF
    PDF_ANNOTATIONS_AVAILABLE = True
except ImportError:
    PDF_ANNOTATIONS_AVAILABLE = False

# For semantic matching and intelligent text understanding:
# - sentence-transformers: pip install sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    SEMANTIC_MATCHING_AVAILABLE = True
except ImportError:
    SEMANTIC_MATCHING_AVAILABLE = False


def clean_extracted_text(text: str) -> str:
    """
    Clean extracted text by removing unwanted characters and normalizing whitespace.
    
    Args:
        text (str): Raw extracted text
        
    Returns:
        str: Cleaned text
    """
    if not text:
        return ""
    
    # Remove backslashes and pipe symbols, but preserve word boundaries
    cleaned = text.replace("\\", " ").replace("|", " ")
    
    # Remove excessive whitespace and normalize
    cleaned = " ".join(cleaned.split())
    
    # Remove tabs and normalize spacing
    cleaned = cleaned.replace("\t", " ")
    
    return cleaned.strip()


class IntelligentTextMatcher:
    """
    Intelligent text matching using semantic embeddings to understand different terminologies.
    Can match concepts even when different words are used.
    """
    
    def __init__(self, model_name="intfloat/e5-large-v2"):
        """Initialize the semantic matcher with a pre-trained model.
        
        Uses intfloat/e5-large-v2 by default - a more powerful semantic model that:
        - Better understands meaning vs word overlap
        - Can distinguish between similar words with different meanings
        - Provides more accurate semantic similarity scores
        """
        self.model = None
        self.model_name = model_name
        self._cache = {}  # Cache for embeddings
        self._terminology_mappings = self._load_domain_terminology()
        self._meaning_validators = self._load_meaning_validators()
        
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity
            self.model = SentenceTransformer(model_name)
            print(f"Semantic matcher initialized with {model_name} for meaning-based matching")
        except ImportError as e:
            print(f"Required packages not available: {e}")
            print("Install with: pip install sentence-transformers scikit-learn")
        except Exception as e:
            print(f"Failed to load semantic model: {e}")
    
    def _load_meaning_validators(self) -> Dict[str, Dict[str, Any]]:
        """Load validators to detect when words are similar but meanings differ.
        
        Returns validators that help distinguish:
        - Same words, different context (e.g., "system validation" vs "validation system")
        - Similar words, different meaning (e.g., "print report" vs "report findings")
        """
        return {
            "context_validators": {
                # Words that change meaning based on context
                "system": {
                    "system validation": "process of validating a system",
                    "validation system": "a system that performs validation",
                    "system requirement": "requirement for a system",
                    "requirement system": "system managing requirements"
                },
                "print": {
                    "print report": "physical printing of documents",
                    "report print": "same as print report",
                    "print system": "printing hardware/software",
                    "system print": "printing system configuration"
                },
                "control": {
                    "control system": "system for controlling processes",
                    "system control": "control over system behavior",
                    "access control": "managing user access",
                    "control access": "same as access control"
                }
            },
            "semantic_opposites": [
                # Word pairs that look similar but mean different things
                ("manual", "automatic"),
                ("required", "optional"),
                ("metallic", "non-metallic"),
                ("contact", "non-contact"),
                ("approved", "rejected"),
                ("compliant", "non-compliant"),
                ("included", "excluded")
            ]
        }
    
    def _load_domain_terminology(self) -> Dict[str, List[str]]:
        """Load domain-specific terminology mappings for URS documents."""
        return {
            # Certificate/Documentation terms
            "certificate": ["cert", "certification", "documentation", "proof", "validation", "verification"],
            "test_report": ["test report", "testing document", "validation report", "verification document", "test result", "test protocol"],
            "material": ["material", "substance", "component", "part", "element"],
            
            # Technical terms
            "metallic": ["metal", "metallic", "steel", "aluminum", "alloy"],
            "non_metallic": ["non-metal", "non-metallic", "plastic", "polymer", "ceramic", "rubber"],
            "contact": ["contact", "touching", "interface", "surface", "connection"],
            
            # System terms
            "system": ["system", "application", "software", "platform", "solution"],
            "access": ["access", "login", "authentication", "authorization", "entry"],
            "security": ["security", "protection", "safety", "secure", "protected"],
            "audit": ["audit", "log", "trail", "record", "tracking", "monitoring"],
            
            # Hardware/Output terms
            "printer": ["printer", "print", "printing", "output", "report generation", "document output"],
            "scope": ["scope", "responsibility", "coverage", "boundary", "domain"],
            "integration": ["integration", "interface", "connection", "linking", "combining"],
            
            # Process terms
            "provide": ["provide", "supply", "deliver", "give", "furnish", "submit"],
            "required": ["required", "needed", "necessary", "mandatory", "must have"],
            "validate": ["validate", "verify", "confirm", "check", "test", "ensure"],
            
            # Quality terms
            "compliance": ["compliance", "adherence", "conformity", "accordance", "standard"],
            "specification": ["specification", "spec", "requirement", "standard", "criteria"],
            "performance": ["performance", "operation", "functioning", "behavior", "execution"],
            
            # Format terms
            "readable": ["readable", "human readable", "legible", "clear", "understandable"],
            "format": ["format", "formatting", "layout", "structure", "presentation"],
            "demand": ["demand", "request", "on-demand", "when needed", "as required"]
        }
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text with caching."""
        if not self.model or not text:
            return np.array([])
        
        # Clean and normalize text
        text = clean_extracted_text(text).lower()
        
        if text in self._cache:
            return self._cache[text]
        
        try:
            embedding = self.model.encode([text])[0]
            self._cache[text] = embedding
            
            # Limit cache size
            if len(self._cache) > 1000:
                # Remove oldest entry
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
            
            return embedding
        except Exception as e:
            print(f"Error generating embedding for '{text}': {e}")
            return np.array([])
    
    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity based on MEANING, not just word overlap.
        
        This method:
        1. Uses semantic embeddings to understand meaning
        2. Validates that similar words have similar meanings
        3. Penalizes matches where words are same but meanings differ
        """
        if not self.model:
            # Fallback to simple keyword matching
            return self._keyword_similarity(text1, text2)
        
        # Get semantic embeddings
        emb1 = self._get_embedding(text1)
        emb2 = self._get_embedding(text2)
        
        if emb1.size == 0 or emb2.size == 0:
            return self._keyword_similarity(text1, text2)
        
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            # Calculate cosine similarity (raw semantic score)
            raw_similarity = cosine_similarity([emb1], [emb2])[0][0]
            raw_similarity = max(0.0, min(1.0, raw_similarity))
            
            # Apply semantic validation to adjust score based on meaning
            validated_similarity = self._validate_semantic_match(text1, text2, raw_similarity)
            
            return validated_similarity
        except Exception as e:
            print(f"Error calculating similarity: {e}")
            return self._keyword_similarity(text1, text2)
    
    def _keyword_similarity(self, text1: str, text2: str) -> float:
        """Fallback keyword-based similarity with enhanced terminology understanding."""
        text1_clean = clean_extracted_text(text1).lower()
        text2_clean = clean_extracted_text(text2).lower()

        # Direct substring matching
        if text1_clean in text2_clean or text2_clean in text1_clean:
            return 0.8

        # Terminology mapping based similarity
        words1 = set(text1_clean.split())
        words2 = set(text2_clean.split())

        # Check for direct word overlap
        common_words = words1.intersection(words2)
        if common_words:
            overlap_ratio = len(common_words) / max(len(words1), len(words2))
            if overlap_ratio > 0.3:
                return overlap_ratio * 0.7  # Scale down keyword-only matches

        # Enhanced terminology mappings with scoring
        concept_matches = []
        for concept, synonyms in self._terminology_mappings.items():
            concept_in_1 = any(syn in text1_clean for syn in synonyms)
            concept_in_2 = any(syn in text2_clean for syn in synonyms)

            if concept_in_1 and concept_in_2:
                concept_matches.append(concept)

        # Score based on number of concept matches
        if concept_matches:
            base_score = 0.5
            bonus_score = min(0.3, len(concept_matches) * 0.1)  # Bonus for multiple concept matches
            return base_score + bonus_score

        # Special case handling for printer/print relationship
        printer_terms = ["printer", "print", "printing"]
        output_terms = ["output", "report", "document", "trail"]

        has_printer_1 = any(term in text1_clean for term in printer_terms)
        has_printer_2 = any(term in text2_clean for term in printer_terms)
        has_output_1 = any(term in text1_clean for term in output_terms)
        has_output_2 = any(term in text2_clean for term in output_terms)

        if (has_printer_1 and has_output_2) or (has_output_1 and has_printer_2):
            return 0.55  # Medium-high confidence for printer-output relationships

        return 0.0
    
    def _validate_semantic_match(self, text1: str, text2: str, raw_score: float) -> float:
        """Validate that high similarity scores represent true semantic matches.
        
        Args:
            text1: First text
            text2: Second text  
            raw_score: Raw similarity score from embeddings
            
        Returns:
            Adjusted score that accounts for meaning differences
        """
        text1_lower = text1.lower()
        text2_lower = text2.lower()
        
        # Check for semantic opposites (same domain, opposite meaning)
        for word1, word2 in self._meaning_validators["semantic_opposites"]:
            has_word1_in_text1 = word1 in text1_lower
            has_word2_in_text1 = word2 in text1_lower
            has_word1_in_text2 = word1 in text2_lower
            has_word2_in_text2 = word2 in text2_lower
            
            # If one text has word1 and other has word2, they're opposite meanings
            if (has_word1_in_text1 and has_word2_in_text2) or (has_word2_in_text1 and has_word1_in_text2):
                # Penalize heavily - same domain but opposite meaning
                return raw_score * 0.3
        
        # Check for context-dependent meaning differences
        context_validators = self._meaning_validators["context_validators"]
        
        # Look for key context words that change meaning
        for keyword, contexts in context_validators.items():
            if keyword in text1_lower and keyword in text2_lower:
                # Both texts have the keyword - check if context is different
                text1_contexts = [ctx for ctx in contexts.keys() if ctx in text1_lower]
                text2_contexts = [ctx for ctx in contexts.keys() if ctx in text2_lower]
                
                if text1_contexts and text2_contexts:
                    # Check if they're talking about different things
                    text1_meanings = {contexts[ctx] for ctx in text1_contexts}
                    text2_meanings = {contexts[ctx] for ctx in text2_contexts}
                    
                    # If meanings don't overlap, penalize the score
                    if not text1_meanings.intersection(text2_meanings):
                        # Same words, different meanings
                        return raw_score * 0.5
        
        # Check word order and structure for context
        # "system validation" vs "validation system" - different meanings
        words1 = text1_lower.split()
        words2 = text2_lower.split()
        
        # If high word overlap but different order, verify it's not a meaning change
        common_words = set(words1).intersection(set(words2))
        if len(common_words) >= 2:  # At least 2 words in common
            # Check if key noun-adjective pairs are reversed
            key_pairs = [("system", "validation"), ("print", "report"), ("access", "control")]
            for word1, word2 in key_pairs:
                if word1 in common_words and word2 in common_words:
                    # Check order in each text
                    try:
                        idx1_text1 = words1.index(word1)
                        idx2_text1 = words1.index(word2)
                        idx1_text2 = words2.index(word1)
                        idx2_text2 = words2.index(word2)
                        
                        # If order is reversed (e.g., "system validation" vs "validation system")
                        order_text1 = idx1_text1 < idx2_text1
                        order_text2 = idx1_text2 < idx2_text2
                        
                        if order_text1 != order_text2:
                            # Word order reversed - might mean different things
                            # Moderate penalty
                            return raw_score * 0.7
                    except ValueError:
                        pass
        
        # If all validation passed, return original score
        return raw_score

    def find_best_semantic_matches(self, query: str, candidates: List[str], threshold: float = 0.3, top_k: int = 5) -> List[Dict[str, Any]]:
        """Find best matching candidate lines for a query using semantic/keyword similarity.

        Returns a list of dicts: {requirement_text, similarity_score, match_type, explanation}
        """
        results: List[Dict[str, Any]] = []
        if not candidates:
            return results

        use_semantic = self.model is not None
        for cand in candidates:
            if not cand or not cand.strip():
                continue
            score = self.calculate_semantic_similarity(query, cand)
            if score >= threshold:
                explanation = "semantic" if use_semantic else "keyword"
                results.append({
                    "requirement_text": cand,
                    "similarity_score": float(score),
                    "match_type": explanation,
                    "explanation": f"{explanation} match, score={score:.2f}"
                })

        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return results[:top_k]


def get_semantic_matcher() -> "IntelligentTextMatcher":
    """Return a singleton instance of IntelligentTextMatcher (no emoji in logs)."""
    global _SEMANTIC_MATCHER_SINGLETON
    try:
        _SEMANTIC_MATCHER_SINGLETON
    except NameError:
        _SEMANTIC_MATCHER_SINGLETON = None
    if _SEMANTIC_MATCHER_SINGLETON is None:
        _SEMANTIC_MATCHER_SINGLETON = IntelligentTextMatcher()
    return _SEMANTIC_MATCHER_SINGLETON


def get_docx_comments_with_text_mapping(file_bytes: bytes) -> Dict[str, List[Dict[str, Any]]]:
    """Parse DOCX comments and map them to exact text spans using OOXML comment ranges.

    Returns a mapping: exact_text -> list of comment dicts {id, text, author, initials, date}
    """
    try:
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as z:
            # If no comments.xml, return empty mapping
            if "word/comments.xml" not in z.namelist():
                return {}

            comments_xml = z.read("word/comments.xml")
            root = etree.fromstring(comments_xml)

            comments_dict: Dict[str, Dict[str, Any]] = {}
            for c in root.xpath("//w:comment", namespaces=OOXML_NS):
                cid = c.get(f"{{{OOXML_NS['w']}}}id")
                # Gather visible text within the comment node
                text_nodes = c.xpath('.//w:t', namespaces=OOXML_NS)
                text = clean_extracted_text(' '.join([t.text for t in text_nodes if t is not None and t.text]))
                author = (c.xpath("@w:author", namespaces=OOXML_NS) or [None])[0]
                initials = (c.xpath("@w:initials", namespaces=OOXML_NS) or [None])[0]
                date = (c.xpath("@w:date", namespaces=OOXML_NS) or [None])[0]
                comments_dict[cid] = {
                    "id": cid,
                    "text": text,
                    "author": author,
                    "initials": initials,
                    "date": date,
                }

            # Parse document to map comments to EXACT text positions
            document_xml = z.read("word/document.xml")
            doc_root = etree.fromstring(document_xml)

            text_to_comments: Dict[str, List[Dict[str, Any]]] = {}
            handled_comment_ids = set()

            # Primary mapping using commentRangeStart/commentRangeEnd
            comment_text_buffers: Dict[str, List[str]] = defaultdict(list)
            active_comment_ids: List[str] = []

            for elem in doc_root.iter():
                try:
                    tag = etree.QName(elem.tag).localname
                except Exception:
                    tag = elem.tag.split('}')[-1]

                if tag == "commentRangeStart":
                    cid = elem.get(f"{{{OOXML_NS['w']}}}id")
                    if cid and cid in comments_dict and cid not in active_comment_ids:
                        active_comment_ids.append(cid)
                        comment_text_buffers.setdefault(cid, [])

                elif tag == "commentRangeEnd":
                    cid = elem.get(f"{{{OOXML_NS['w']}}}id")
                    if cid in active_comment_ids:
                        active_comment_ids = [c for c in active_comment_ids if c != cid]
                    if cid and cid in comment_text_buffers and cid in comments_dict:
                        parts = comment_text_buffers.get(cid, [])
                        text = clean_extracted_text(' '.join(parts))
                        if text:
                            existing_ids = {c.get('id') for c in text_to_comments.setdefault(text, [])}
                            if cid not in existing_ids:
                                text_to_comments[text].append(comments_dict[cid])
                                handled_comment_ids.add(cid)
                        comment_text_buffers.pop(cid, None)

                elif tag == "t":
                    if active_comment_ids:
                        value = elem.text or ''
                        if value:
                            for cid in active_comment_ids:
                                comment_text_buffers[cid].append(value)

                elif tag in {"tab", "cr", "br"}:
                    if active_comment_ids:
                        for cid in active_comment_ids:
                            comment_text_buffers[cid].append(' ')

            # Flush any buffers in case of malformed ranges
            for cid, parts in list(comment_text_buffers.items()):
                if cid in comments_dict and parts:
                    text = clean_extracted_text(' '.join(parts))
                    if text:
                        existing_ids = {c.get('id') for c in text_to_comments.setdefault(text, [])}
                        if cid not in existing_ids:
                            text_to_comments[text].append(comments_dict[cid])
                            handled_comment_ids.add(cid)
                comment_text_buffers.pop(cid, None)

            # Fallback: handle commentReference elements for any unmapped comments
            comment_refs = doc_root.xpath('.//w:commentReference', namespaces=OOXML_NS)

            for ref in comment_refs:
                comment_id = ref.get(f"{{{OOXML_NS['w']}}}id")

                if comment_id not in comments_dict or comment_id in handled_comment_ids:
                    continue

                comment_mapped = False

                # Method 1: Get the EXACT text run that contains the comment reference
                parent_run = ref.xpath('ancestor::w:r', namespaces=OOXML_NS)
                
                if parent_run:
                    run_text_nodes = parent_run[0].xpath('.//w:t', namespaces=OOXML_NS)
                    run_text = clean_extracted_text(''.join([node.text for node in run_text_nodes if node.text]))
                    
                    if run_text and len(run_text) > 3:
                        existing_ids = {c.get('id') for c in text_to_comments.setdefault(run_text, [])}
                        if comment_id not in existing_ids:
                            text_to_comments[run_text].append(comments_dict[comment_id])
                        handled_comment_ids.add(comment_id)
                        comment_mapped = True
                        continue

                # Method 2: If in a table, try table cell context
                if not comment_mapped:
                    parent_cell = ref.xpath('ancestor::w:tc', namespaces=OOXML_NS)
                    if parent_cell:
                        cell_text_nodes = parent_cell[0].xpath('.//w:t', namespaces=OOXML_NS)
                        cell_text = clean_extracted_text(''.join([node.text for node in cell_text_nodes if node.text]))
                        
                        if cell_text and len(cell_text) > 3:
                            parts = re.split(r'[:\s]{2,}', cell_text)
                            best_part = None
                            for part in parts:
                                part = clean_extracted_text(part)
                                if 3 < len(part) < 50:
                                    if not best_part or len(part) < len(best_part):
                                        best_part = part
                            
                            target_text = best_part or cell_text
                            existing_ids = {c.get('id') for c in text_to_comments.setdefault(target_text, [])}
                            if comment_id not in existing_ids:
                                text_to_comments[target_text].append(comments_dict[comment_id])
                            handled_comment_ids.add(comment_id)
                            comment_mapped = True

                # Method 3: Enhanced paragraph-level mapping with better precision
                if not comment_mapped:
                    parent_para = ref.xpath('ancestor::w:p', namespaces=OOXML_NS)
                    
                    if parent_para:
                        para_text_nodes = parent_para[0].xpath('.//w:t', namespaces=OOXML_NS)
                        para_text = clean_extracted_text(''.join([node.text for node in para_text_nodes if node.text]))
                        
                        if para_text and len(para_text) > 3:
                            if len(para_text) < 100:
                                existing_ids = {c.get('id') for c in text_to_comments.setdefault(para_text, [])}
                                if comment_id not in existing_ids:
                                    text_to_comments[para_text].append(comments_dict[comment_id])
                                handled_comment_ids.add(comment_id)
                                comment_mapped = True
                            else:
                                sentences = re.split(r'[.!?]+\s+', para_text)
                                best_sentence = None
                                for sentence in sentences:
                                    sentence = clean_extracted_text(sentence)
                                    if 10 < len(sentence) < 80:
                                        if any(keyword in sentence.lower() for keyword in ['training', 'specification', 'requirement', 'shall', 'must']):
                                            best_sentence = sentence
                                            break
                                        elif not best_sentence:
                                            best_sentence = sentence
                                
                                if best_sentence:
                                    existing_ids = {c.get('id') for c in text_to_comments.setdefault(best_sentence, [])}
                                    if comment_id not in existing_ids:
                                        text_to_comments[best_sentence].append(comments_dict[comment_id])
                                    handled_comment_ids.add(comment_id)
                                    comment_mapped = True

            return text_to_comments

    except Exception as e:
        # Log the error for debugging but don't crash
        print(f"Warning: Error parsing DOCX comments: {e}")
        import traceback
        traceback.print_exc()
    return {}


def detect_document_structure(doc) -> Dict[str, Any]:
    """Detect a simple section structure from a DOCX document.

    Returns dict: {"sections": {section_title: {"title": section_title, "section_number": idx}}}
    """
    sections: Dict[str, Dict[str, Any]] = {}
    idx = 1

    # Heuristic 1: Numbered headings like "1.0 Title" or "2 Title"
    for p in doc.paragraphs:
        txt = clean_extracted_text(p.text)
        if not txt or len(txt) < 3:
            continue
        m = re.match(r"^(\d+(?:\.\d+)*)\s+(.{3,})$", txt)
        if m and len(m.group(2).strip()) >= 3:
            title = txt.strip()
            if title not in sections:
                sections[title] = {"title": title, "section_number": idx}
                idx += 1

    # Heuristic 2: Fallback to common section keywords if none detected
    if not sections:
        common_sections = [
            "Approval", "Introduction", "Overview", "Statutory Regulations", "Process Requirements",
            "Cleaning Requirements", "General Requirement Functions", "Installation Requirements",
            "Health Safety Environment", "Documentation Training", "References Definitions",
            "Abbreviations", "Revision History", "Vendor Acceptance"
        ]
        for title in common_sections:
            if title not in sections:
                sections[title] = {"title": title, "section_number": idx}
                idx += 1

    return {"sections": sections}
def get_paragraph_comments(paragraph, comments_dict: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return list of comment dicts attached to a paragraph.

    Each item contains: {id, text, author, initials, date}
    """
    comments: List[Dict[str, Any]] = []
    try:
        # Get the XML element for this paragraph
        p_element = paragraph._element
        
        # Find all comment reference elements using lxml etree
        from lxml import etree
        # Convert paragraph element to lxml element if needed
        if hasattr(p_element, 'xml'):
            # This is a python-docx element, get raw XML
            xml_str = p_element.xml
            element = etree.fromstring(xml_str)
        else:
            element = p_element
            
        refs = element.xpath(".//w:commentReference", namespaces=OOXML_NS)
        for ref in refs:
            cid = ref.get(f"{{{OOXML_NS['w']}}}id")
            if cid and cid in comments_dict:
                comments.append(comments_dict[cid])
    except Exception:
        pass
    return comments


def categorize_requirements_by_section(pages: List[Dict[str, Any]], structure: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Categorize requirements by document sections."""
    
    # If no clear structure detected, return as-is
    if not structure["sections"]:
        return pages
    
    # Create section mapping
    section_mapping = {}
    sections_list = list(structure["sections"].items())
    
    for i, page in enumerate(pages):
        content = page.get("content", "")
        page_type = page.get("content_type", "")
        
        # Skip non-table content for now (focus on requirements in tables)
        if page_type != "docx-table-row":
            continue
            
        # Try to determine which section this requirement belongs to
        detected_section = None
        
        # Method 1: Look for section numbers in the content
        for section_title, section_info in sections_list:
            # Extract section number from title (e.g., "1.0", "2.1", "3.2")
            section_match = re.search(r'^(\d+(?:\.\d+)?)', section_title)
            if section_match:
                section_num = section_match.group(1)
                # Look for this section number in the table row content
                if section_num in content:
                    detected_section = section_title
                    break
        
        # Method 2: Look for keywords from section titles
        if not detected_section:
            for section_title, section_info in sections_list:
                # Extract key words from section title
                title_words = re.findall(r'\b[A-Z][a-z]+\b', section_title)
                if title_words:
                    for word in title_words:
                        if word.lower() in content.lower() and len(word) > 3:
                            detected_section = section_title
                            break
                if detected_section:
                    break
        
        # Method 3: Sequential assignment based on position
        if not detected_section and sections_list:
            # For table rows, try to assign based on order
            section_index = min(i // 10, len(sections_list) - 1)  # Rough grouping
            detected_section = sections_list[section_index][0]
        
        # Add section information to the page
        if detected_section:
            page["section"] = detected_section
            page["section_number"] = structure["sections"][detected_section].get("section_number", 0)
        else:
            page["section"] = "Unspecified"
            page["section_number"] = 999
    
    return pages


def extract_section_based_requirements(doc, file_bytes: bytes) -> List[Dict[str, Any]]:
    """Extract requirements organized by document sections."""
    
    # First, detect document structure
    structure = detect_document_structure(doc)
    text_to_comments = get_docx_comments_with_text_mapping(file_bytes)
    
    pages = []
    content_buffer = []
    comments_buffer = []
    page_num = 1
    current_section = "Introduction"
    
    # DEDUPLICATION: Track seen content to prevent duplicates
    seen_content = set()
    
    # Create a lookup for section detection by keywords
    section_keywords = {}
    for section_name, section_info in structure["sections"].items():
        words = section_name.lower().split()
        for word in words:
            if len(word) > 3:  # Only meaningful words
                if word not in section_keywords:
                    section_keywords[word] = []
                section_keywords[word].append(section_name)
    
    # Track repeated content to filter out headers/footers
    content_frequency = {}
    for p in doc.paragraphs:
        text = clean_extracted_text(p.text).strip()
        if text:
            content_frequency[text] = content_frequency.get(text, 0) + 1
    
    # Identify repeated header/footer text (appears more than 2 times)
    repeated_headers = {text for text, count in content_frequency.items() if count > 2}
    
    # Iterate through document elements
    for element in doc.element.body:
        if element.tag.endswith('p'):
            # Find the corresponding paragraph object
            para = None
            for p in doc.paragraphs:
                if p._p == element:
                    para = p
                    break
            
            if para and para.text.strip():
                para_text = clean_extracted_text(para.text)
                
                # Skip repeated headers/footers (like document titles repeated on every page)
                if para_text in repeated_headers:
                    # Allow if it has comments or if it's the first occurrence
                    has_comments = para_text in text_to_comments and text_to_comments[para_text]
                    is_substantial = len(para_text.split()) > 10  # More than 10 words
                    
                    if not has_comments and not is_substantial:
                        continue
                
                # Enhanced section detection
                detected_section = None
                
                # Method 1: Direct section title match
                for section_name in structure["sections"]:
                    section_title = structure["sections"][section_name]["title"]
                    if section_title.lower() in para_text.lower() or para_text.lower() in section_title.lower():
                        detected_section = section_name
                        break
                
                # Method 2: Check for numbered sections in the paragraph
                if not detected_section:
                    numbered_match = re.search(r'(\d+\.?\d*)\s+([A-Z][A-Za-z\s]+)', para_text)
                    if numbered_match:
                        section_text = clean_extracted_text(numbered_match.group(2))
                        # Look for this section text in our detected sections
                        for section_name in structure["sections"]:
                            if section_text.lower() in section_name.lower():
                                detected_section = section_name
                                break
                
                # Method 3: Keyword-based detection
                if not detected_section:
                    para_words = para_text.lower().split()
                    for word in para_words:
                        if word in section_keywords:
                            # Use the first matching section
                            detected_section = section_keywords[word][0]
                            break
                
                if detected_section:
                    current_section = detected_section
                
                content_buffer.append(para_text)
                para_comments = text_to_comments.get(para_text, [])
                comments_buffer.extend(para_comments)
        
        elif element.tag.endswith('tbl'):
            # Find the corresponding table object
            table = None
            for t in doc.tables:
                if t._tbl == element:
                    table = t
                    break
            
            if not table:
                continue
            
            # Add any preceding text paragraphs as a page
            if content_buffer:
                pages.append({
                    "page_number": page_num,
                    "content": "\n".join(content_buffer).strip(),
                    "tables": [],
                    "content_type": "docx",
                    "comments": comments_buffer,
                    "section": current_section,
                })
                page_num += 1
                content_buffer, comments_buffer = [], []
            
            # Process table with enhanced section awareness, splitting cells into individual requirements
            table_header = None

            for row_idx, row in enumerate(table.rows):
                if row_idx == 0:
                    table_header = " | ".join([clean_extracted_text(c.text) for c in row.cells])
                    continue

                row_section = current_section

                for cell_idx, cell in enumerate(row.cells):
                    cell_text = clean_extracted_text(cell.text)
                    if not cell_text or len(cell_text) < 15:
                        continue

                    # Remove table prefixes like "3.0 | OVERVIEW | ..." from cell text before processing
                    _PREFIX_RE = re.compile(r'^(?:\s*\d+(?:\.\d+)*\s*\|\s*[A-Z0-9 /&\-]+(?:\s*\|\s*[A-Z0-9 /&\-]+)*)\s+')
                    clean_cell_text = cell_text
                    m = _PREFIX_RE.match(clean_cell_text)
                    if m:
                        clean_cell_text = clean_cell_text[m.end():].strip()

                    # Skip empty or duplicate rows that are just section headers
                    if not clean_cell_text or len(clean_cell_text) < 8:
                        continue
                    if re.match(r'^[A-Z\s/]+$', clean_cell_text):
                        continue

                    # Determine section from numbers/keywords inside this cell
                    section_number_match = re.search(r'(\d+\.\d+(?:\.\d+){0,2}|\d+\.0|\d+)\b', cell_text)
                    if section_number_match:
                        section_num = section_number_match.group(1)
                        if section_num.startswith('1'):
                            row_section = "Approval" if '1.0' in section_num else "Introduction"
                        elif section_num.startswith('2'):
                            row_section = "Introduction"
                        elif section_num.startswith('3'):
                            row_section = "Overview"
                        elif section_num.startswith('4'):
                            row_section = "Statutory Regulations"
                        elif section_num.startswith('5'):
                            row_section = "Process Requirements"
                        elif section_num.startswith('6'):
                            row_section = "Cleaning Requirements"
                        elif section_num.startswith('7'):
                            row_section = "General Requirement Functions"
                        elif section_num.startswith('8'):
                            row_section = "Installation Requirements"
                        elif section_num.startswith('9'):
                            row_section = "Health Safety Environment"
                        elif section_num.startswith('10'):
                            row_section = "Documentation Training"
                        elif section_num.startswith('11'):
                            row_section = "References Definitions"
                        elif section_num.startswith('12'):
                            row_section = "Abbreviations"
                        elif section_num.startswith('13'):
                            row_section = "Revision History"
                        elif section_num.startswith('14'):
                            row_section = "Vendor Acceptance"
                        else:
                            row_section = f"Section {section_num}"
                    else:
                        cl = clean_cell_text.lower()
                        if any(k in cl for k in ["control","operational","parameter"]):
                            row_section = "General Requirement Functions"
                        elif any(k in cl for k in ["process","granulation","flow"]):
                            row_section = "Process Requirements"
                        elif any(k in cl for k in ["cleaning","cip","wash"]):
                            row_section = "Cleaning Requirements"
                        elif any(k in cl for k in ["installation","utility","connection"]):
                            row_section = "Installation Requirements"
                        elif any(k in cl for k in ["training","documentation","manual"]):
                            row_section = "Documentation Training"
                        elif any(k in cl for k in ["safety","health","environment","emergency"]):
                            row_section = "Health Safety Environment"

                    # Use clean_cell_text for processing (without table prefixes)
                    # Enhanced requirement detection - don't just look for modal verbs
                    
                    # Function to check if text is a meaningful requirement/specification
                    def is_meaningful_requirement(text):
                        text_lower = text.lower()
                        # Modal verbs (strong indicators)
                        if any(kw in text_lower for kw in ['shall','must','should','will','require','required']):
                            return True
                        # Technical specifications and parameters
                        if any(kw in text_lower for kw in ['temperature','pressure','speed','flow','capacity','voltage',
                                                          'frequency','bar','°c','rpm','m³/h','kg','amperage','kw',
                                                          'level','alarm','trip','control','pump','valve','filter',
                                                          'exhaust','inlet','outlet','spray','air','water']):
                            return True
                        # Safety and compliance terms
                        if any(kw in text_lower for kw in ['safety','emergency','stop','e-stop','earthing','grounding',
                                                          'interlocking','guarded','concealed','noise','ergonomic',
                                                          'training','compliance','cgmp','validation']):
                            return True
                        # Equipment and design specifications
                        if any(kw in text_lower for kw in ['electrical','wiring','moving parts','mechanism','equipment',
                                                          'provided by vendor','design','installation','phase','volts']):
                            return True
                        return False
                    
                    # Split into bullet items first
                    bullet_pattern = r'(?:^|\n)[\s]*(?:[•\-*]|\d+\.)\s*(.+?)(?=(?:\n[\s]*(?:[•\-*]|\d+\.))|$)'
                    bullets = re.findall(bullet_pattern, clean_cell_text, re.DOTALL)
                    items = [clean_extracted_text(b) for b in bullets if len(b.strip()) > 8] if bullets else []
                    
                    if not items:
                        # Try splitting on tabs (common in specification tables)
                        if '\t' in clean_cell_text:
                            tab_parts = [p.strip() for p in clean_cell_text.split('\t') if p.strip() and len(p.strip()) >= 8]
                            if len(tab_parts) > 1:
                                items = tab_parts
                        
                        # Try splitting on semicolons and em-dashes for long requirement lists
                        if not items and len(clean_cell_text) > 80 and (';' in clean_cell_text or '–' in clean_cell_text or '—' in clean_cell_text):
                            parts = re.split(r'\s*;\s+|\s*[\u2013\u2014]\s+', clean_cell_text)
                            parts = [p.strip() for p in parts if p and len(p.strip()) >= 8]
                            if len(parts) > 1:
                                items = parts
                        
                        # Try splitting on newlines for multi-line specifications
                        if not items and '\n' in clean_cell_text:
                            line_parts = [p.strip() for p in clean_cell_text.split('\n') if p.strip() and len(p.strip()) >= 8]
                            if len(line_parts) > 1:
                                items = line_parts
                        
                        # Fallback to sentence splitting but be more inclusive
                        if not items:
                            parts = re.split(r'[.!?]+\s+', clean_cell_text)
                            items = [
                                (clean_extracted_text(p) + ".") if not p.strip().endswith(('.', '!', '?')) else clean_extracted_text(p)
                                for p in parts
                                if len(p.strip()) > 8
                            ]
                    
                    # Filter items to keep only meaningful requirements/specifications
                    filtered_items = []
                    for item in items:
                        if is_meaningful_requirement(item):
                            filtered_items.append(item)
                    
                    # If no meaningful items found but original text seems like a specification, keep it
                    if not filtered_items and len(clean_cell_text) > 8 and is_meaningful_requirement(clean_cell_text):
                        filtered_items = [clean_cell_text]
                    
                    items = filtered_items

                    if not items:
                        continue

                    # Collect exact text comments at cell/paragraph level
                    cell_comments = []
                    if cell_text and cell_text in text_to_comments:
                        cell_comments.extend(text_to_comments[cell_text])
                    for para in cell.paragraphs:
                        ptx = clean_extracted_text(para.text)
                        if ptx and ptx in text_to_comments:
                            for c in text_to_comments[ptx]:
                                if c not in cell_comments and len(cell_comments) < 5:
                                    cell_comments.append(c)

                    # Emit a page per requirement item
                    for req_idx, item in enumerate(items):
                        if not item.strip():
                            continue
                        norm = " ".join(item.split()).lower()
                        if norm in seen_content:
                            continue
                        seen_content.add(norm)

                        # Use the clean item text without table prefixes for display
                        pages.append({
                            "page_number": f"Table-{page_num}-R{row_idx}-C{cell_idx}-{req_idx}",
                            "content": item,  # Clean requirement text without table prefix
                            "tables": [item],
                            "content_type": "docx-table-row",
                            "comments": cell_comments,
                            "section": row_section,
                            "requirement_id": f"{row_section.replace(' ', '')}-{page_num}-{row_idx}-{req_idx}",
                        })
                        page_num += 1
    
    # Add any remaining content
    if content_buffer:
        pages.append({
            "page_number": page_num,
            "content": "\n".join(content_buffer).strip(),
            "tables": [],
            "content_type": "docx",
            "comments": comments_buffer,
            "section": current_section,
        })
    
    return pages


def extract_pdf_annotations_and_text(file_bytes: bytes) -> Dict[str, Any]:
    """
    Extract text and annotations from PDF with precise line-by-line mapping.
    Returns dict with text content and mapped annotations.
    """
    if not PDF_ANNOTATIONS_AVAILABLE:
        print("PyMuPDF not available, falling back to basic PDF extraction")
        return {"pages": [], "annotations": []}
    
    try:
        pdf_doc = fitz.open(stream=file_bytes, filetype="pdf")
        pages_data = []
        all_annotations = []
        
        for page_num in range(len(pdf_doc)):
            page = pdf_doc.load_page(page_num)
            
            # Extract text with line-level precision
            text_dict = page.get_text("dict")
            page_text = ""
            text_lines = []
            line_positions = []
            
            # Build text line by line with position tracking
            for block in text_dict["blocks"]:
                if "lines" in block:
                    for line in block["lines"]:
                        line_text = ""
                        line_bbox = None
                        
                        for span in line["spans"]:
                            line_text += span["text"]
                            if line_bbox is None:
                                line_bbox = span["bbox"]
                            else:
                                # Expand bbox to include this span
                                line_bbox = [
                                    min(line_bbox[0], span["bbox"][0]),
                                    min(line_bbox[1], span["bbox"][1]),
                                    max(line_bbox[2], span["bbox"][2]),
                                    max(line_bbox[3], span["bbox"][3])
                                ]
                        
                        if line_text.strip():
                            cleaned_line_text = clean_extracted_text(line_text)
                            if cleaned_line_text:  # Only add if there's content after cleaning
                                text_lines.append(cleaned_line_text)
                                line_positions.append(line_bbox)
                                page_text += cleaned_line_text + "\n"
            
            # Extract annotations for this page
            page_annotations = []
            annotations = page.annots()
            
            for annot in annotations:
                try:
                    annot_dict = annot.info
                    annot_rect = annot.rect
                    
                    # Get annotation content
                    content = clean_extracted_text(annot_dict.get("content", ""))
                    if not content:
                        # Try to get subject or title as fallback
                        content = clean_extracted_text(annot_dict.get("subject", "") or annot_dict.get("title", ""))
                    
                    if content:
                        # Find which text line(s) this annotation is closest to
                        closest_lines = []
                        annot_center_y = (annot_rect.y0 + annot_rect.y1) / 2
                        
                        # Find overlapping or nearby text lines
                        for i, line_bbox in enumerate(line_positions):
                            if line_bbox:
                                line_center_y = (line_bbox[1] + line_bbox[3]) / 2
                                vertical_distance = abs(annot_center_y - line_center_y)
                                
                                # Consider lines within reasonable vertical proximity
                                if vertical_distance < 50:  # Adjust threshold as needed
                                    closest_lines.append({
                                        "line_index": i,
                                        "line_text": text_lines[i],
                                        "distance": vertical_distance
                                    })
                        
                        # Sort by distance and take the closest
                        closest_lines.sort(key=lambda x: x["distance"])
                        
                        annotation_data = {
                            "id": f"page_{page_num + 1}_annot_{len(page_annotations)}",
                            "type": annot_dict.get("name", "Comment"),
                            "text": content,
                            "author": annot_dict.get("author", "Unknown"),
                            "page": page_num + 1,
                            "position": {
                                "x0": annot_rect.x0,
                                "y0": annot_rect.y0,
                                "x1": annot_rect.x1,
                                "y1": annot_rect.y1
                            },
                            "associated_lines": closest_lines[:3]  # Keep top 3 closest lines
                        }
                        
                        page_annotations.append(annotation_data)
                        all_annotations.append(annotation_data)
                        
                except Exception as e:
                    print(f"Error processing annotation on page {page_num + 1}: {e}")
                    continue
            
            # Extract tables using fitz
            tables = []
            try:
                tabs = page.find_tables()
                for tab in tabs:
                    table_data = tab.extract()
                    if table_data:
                        table_text = []
                        for row in table_data:
                            row_text = " | ".join(str(cell) if cell else "" for cell in row)
                            table_text.append(row_text)
                        tables.append("\n".join(table_text))
            except Exception as e:
                print(f"Error extracting tables from page {page_num + 1}: {e}")
            
            page_data = {
                "page_number": page_num + 1,
                "content": page_text.strip(),
                "text_lines": text_lines,
                "line_positions": line_positions,
                "tables": tables,
                "annotations": page_annotations,
                "content_type": "pdf"
            }
            
            pages_data.append(page_data)
        
        pdf_doc.close()
        
        return {
            "pages": pages_data,
            "annotations": all_annotations
        }
        
    except Exception as e:
        print(f"Error extracting PDF annotations: {e}")
        return {"pages": [], "annotations": []}


def map_pdf_annotations_to_requirements(pages_data: List[Dict], annotations: List[Dict]) -> List[Dict]:
    """
    Map PDF annotations to specific requirements with intelligent semantic matching.
    Uses both spatial proximity and semantic understanding.
    """
    semantic_matcher = get_semantic_matcher()
    
    for page_data in pages_data:
        page_annotations = page_data.get("annotations", [])
        page_content = page_data.get("content", "")
        text_lines = page_data.get("text_lines", [])
        
        # Create a mapping from annotations to specific text segments
        mapped_comments = []
        
        for annotation in page_annotations:
            comment_text = annotation.get("text", "")
            associated_lines = annotation.get("associated_lines", [])
            
            # Stage 1: Spatial proximity matching (existing logic)
            spatial_matches = []
            for line_info in associated_lines:
                line_text = line_info.get("line_text", "")
                
                # Comprehensive URS requirement detection keywords (much more inclusive)
                requirement_keywords = [
                    # Obligation keywords (strong indicators)
                    "shall", "must", "should", "will", "require", "required", "necessary", "need", "needs", "needed",
                    "expect", "expected", "demand", "specify", "specified", "ensure", "provide", "maintain",
                    
                    # Technical specification keywords
                    "specification", "standard", "compliance", "certificate", "certification", "spec", "specs",
                    "test", "testing", "report", "validation", "verification", "qualification", "analysis", "analytical",
                    "measure", "measurement", "monitor", "monitoring", "control", "controlled", "verify",
                    
                    # Material and construction keywords
                    "material", "materials", "metallic", "non-metallic", "stainless steel", "aisi", "316l", "304l",
                    "product contact", "contact parts", "surface roughness", "mirror polished", "construction",
                    "built", "made", "fabricated", "manufactured", "design", "designed", "finish", "coating",
                    
                    # Process and operational keywords
                    "process", "processing", "operation", "operational", "control", "parameter", "parameters",
                    "temperature", "pressure", "flow", "speed", "capacity", "range", "limit", "limits",
                    "alarm", "trip", "setpoint", "operating", "performance", "efficiency", "function",
                    
                    # Documentation and training keywords
                    "documentation", "document", "training", "manual", "procedure", "procedures", "protocol",
                    "audit", "trail", "security", "access", "authorization", "backup", "record", "records",
                    "certificate", "certificates", "approval", "approved", "qualification", "qualified",
                    
                    # Equipment and system keywords
                    "equipment", "system", "systems", "machine", "device", "devices", "component", "components",
                    "assembly", "installation", "maintenance", "service", "support", "unit", "units", "module",
                    "instrument", "instruments", "tool", "tools", "apparatus", "machinery",
                    
                    # Quality and safety keywords
                    "quality", "safety", "gmp", "cgmp", "fda", "cfr", "part 11", "gamp", "iso", "astm",
                    "containment", "atex", "hazard", "hazards", "risk", "risks", "emergency", "protection",
                    "safe", "safer", "secure", "security", "reliable", "reliability",
                    
                    # Capacity and measurement keywords
                    "capacity", "volume", "size", "dimension", "dimensions", "area", "height", "width", "length",
                    "diameter", "weight", "mass", "density", "thickness", "level", "amount", "quantity",
                    
                    # Utility and infrastructure keywords
                    "utility", "utilities", "power", "electricity", "electrical", "water", "steam", "compressed air",
                    "nitrogen", "vacuum", "drainage", "ventilation", "hvac", "lighting", "communication",
                    
                    # Location and facility keywords
                    "location", "area", "room", "zone", "space", "facility", "building", "floor", "clean room",
                    "warehouse", "storage", "environment", "environmental", "atmosphere", "conditions",
                    
                    # Personnel and training keywords
                    "personnel", "staff", "operator", "operators", "technician", "manager", "supervisor",
                    "qualified", "trained", "competent", "authorized", "responsible", "user", "users",
                    
                    # Time and scheduling keywords
                    "time", "duration", "schedule", "frequency", "interval", "cycle", "batch", "continuous",
                    "periodic", "regular", "annual", "daily", "weekly", "monthly", "shift",
                    
                    # General action keywords
                    "provide", "supply", "deliver", "install", "configure", "setup", "implement", "execute",
                    "perform", "conduct", "carry out", "achieve", "accomplish", "complete", "finish",
                    
                    # Comparative and range keywords
                    "minimum", "maximum", "min", "max", "range", "between", "from", "to", "up to", "down to",
                    "less than", "greater than", "equal", "approximately", "about", "around", "typical",
                    
                    # Technical units and measurements
                    "kg", "g", "mg", "ton", "l", "ml", "m3", "m", "cm", "mm", "inch", "ft", "°c", "°f",
                    "bar", "psi", "mpa", "pascal", "rpm", "hz", "v", "volt", "amp", "watt", "kw", "mw",
                    "ph", "ppm", "ppb", "percent", "ratio", "conductivity", "viscosity",
                    
                    # Manufacturing and pharmaceutical specific
                    "batch", "lot", "campaign", "product", "intermediate", "api", "excipient", "tablet",
                    "capsule", "liquid", "solid", "powder", "granule", "coating", "formulation", "recipe"
                ]
                
                # URS-specific patterns for requirement identification
                line_lower = line_text.lower()
                line_stripped = line_text.strip()
                
                # Pattern 1: Numbered requirements (very strong indicator)
                numbered_requirement = (
                    re.search(r'^\d+\.\d+(\.\d+)?\s+', line_stripped) or  # 7.2.1.1, 10.2.3
                    re.search(r'^[•\-\*]\s*\d+\.\d+', line_stripped) or   # • 7.2.1
                    re.search(r'^\d+\)\s+', line_stripped) or             # 1) Item
                    re.search(r'^req\s*no\.?\s*\d+', line_lower) or       # Req No 7.1.1
                    re.search(r'^sr\.?\s*no\.?\s*\d+', line_lower)        # Sr. No. 1
                )
                
                # Pattern 2: Table-based requirements (common in URS) - much more inclusive
                table_requirement = (
                    '|' in line_text and len(line_text.split('|')) >= 2 and
                    len(line_stripped) > 15 and  # Any meaningful table content
                    not any(header in line_lower for header in ["sr. no", "sr.no", "page no", "topics"])
                )
                
                # Pattern 3: Specification lines with technical details - more inclusive
                specification_pattern = (
                    re.search(r'(temperature|pressure|flow|speed|capacity|voltage|frequency|ph|conductivity|viscosity).*[:=]\s*[\d\w]', line_lower) or
                    re.search(r'(min|max|minimum|maximum|range|limit|between|from|to)\.?\s*[:=]?\s*\d+', line_lower) or
                    re.search(r'\d+\s*(bar|°c|°f|rpm|m³/h|kg|g|mg|l|ml|mm|µm|inch|ft|v|amp|watt|hz|psi|mpa|ra\s*≤)', line_lower) or
                    re.search(r'(supply|operating|design|working|storage|ambient|process)\s+(pressure|temperature|conditions|range)', line_lower) or
                    re.search(r'\d+.*\s*(percent|%|ppm|ppb|ratio)', line_lower) or
                    re.search(r'(accuracy|precision|tolerance)\s*[:=±]\s*[\d.]+', line_lower)
                )
                
                # Pattern 4: Compliance and standard references - more inclusive
                compliance_pattern = (
                    re.search(r'(comply|compliance|conform|conformance|accordance|conformity)\s+with', line_lower) or
                    re.search(r'(21\s*cfr|part\s*11|gmp|cgmp|fda|eu|iso\s*\d+|astm|ansi|din|bs|en\s*\d+)', line_lower) or
                    re.search(r'(standard|standards|guideline|guidelines|regulation|regulations|directive|code|norm)', line_lower) or
                    re.search(r'(certificate|certification|qualified|approved|validated|verified)', line_lower)
                )
                
                # Pattern 5: Equipment specifications and requirements - much more inclusive
                equipment_pattern = (
                    re.search(r'(equipment|system|machine|device|unit|component|instrument|tool|apparatus)', line_lower) or
                    re.search(r'(material|construction|design|fabrication|manufacturing|installation)', line_lower) or
                    re.search(r'(provide|supply|deliver|install|configure|setup|implement|include)', line_lower) or
                    re.search(r'(capacity|volume|size|dimension|area|height|width|length|weight)', line_lower) or
                    re.search(r'(utility|utilities|power|electrical|water|steam|air|gas|vacuum|drainage)', line_lower)
                )
                
                # Pattern 6: Safety and operational requirements - more inclusive
                safety_pattern = (
                    re.search(r'(safety|emergency|alarm|interlock|guard|protection|containment|hazard|risk)', line_lower) or
                    re.search(r'(training|personnel|operator|qualified|authorized|competent)', line_lower) or
                    re.search(r'(maintenance|service|cleaning|calibration|repair|replacement)', line_lower) or
                    re.search(r'(documentation|record|report|manual|procedure|protocol)', line_lower)
                )
                
                # Pattern 7: Process and manufacturing requirements
                process_pattern = (
                    re.search(r'(process|processing|operation|batch|lot|campaign|production)', line_lower) or
                    re.search(r'(control|monitor|measure|test|analyze|verify|validate)', line_lower) or
                    re.search(r'(pharmaceutical|api|excipient|tablet|capsule|liquid|solid|powder)', line_lower)
                )
                
                # Pattern 8: Any line with technical specifications or numbers
                technical_pattern = (
                    re.search(r'\d+.*(?:kg|g|mg|l|ml|m|cm|mm|°c|°f|bar|psi|rpm|v|amp|watt|%|ppm)', line_lower) or
                    re.search(r'(?:range|between|from|to|up to|down to|approximately|about)\s*\d+', line_lower) or
                    re.search(r'\d+\s*(?:x|by|×)\s*\d+', line_lower)  # Dimensions like 100 x 200
                )
                
                # Combined requirement detection logic - much more inclusive
                is_requirement = (
                    len(line_text) > 8 and  # Reduced minimum length
                    (
                        # Strong indicators (high confidence)
                        numbered_requirement or
                        table_requirement or
                        specification_pattern or
                        compliance_pattern or
                        equipment_pattern or
                        safety_pattern or
                        process_pattern or
                        technical_pattern or
                        
                        # Keyword-based detection (medium confidence) - much more inclusive
                        (len(line_text) > 15 and 
                         any(keyword in line_lower for keyword in requirement_keywords) and
                         not any(exclusion in line_lower for exclusion in [
                             "table of contents", "page no.", "page no", "revision history", 
                             "abbreviation", "definition", "reference list", "approval signature",
                             "document control", "distribution list", "change control"
                         ])) or
                        
                        # Very inclusive catch-all for potential requirements
                        (len(line_text) > 20 and 
                         (
                             # Any line mentioning specifications, requirements, or technical details
                             any(term in line_lower for term in [
                                 "requirement", "specification", "standard", "certificate", "test", "report",
                                 "material", "equipment", "system", "process", "control", "parameter",
                                 "capacity", "temperature", "pressure", "flow", "safety", "training",
                                 "documentation", "maintenance", "utility", "design", "construction"
                             ]) or
                             
                             # Any line with technical measurements or ranges
                             re.search(r'\d+', line_text) and any(unit in line_lower for unit in [
                                 "kg", "g", "l", "ml", "m", "cm", "mm", "°c", "°f", "bar", "psi", "rpm", "%"
                             ]) or
                             
                             # Any line describing what should be provided/done
                             any(action in line_lower for action in [
                                 "provide", "supply", "deliver", "install", "ensure", "maintain", "control",
                                 "monitor", "verify", "validate", "comply", "conform", "include", "contain"
                             ])
                         ) and
                         not any(exclusion in line_lower for exclusion in [
                             "table of contents", "page no", "revision", "version", "date",
                             "author", "approved by", "reviewed by", "signature"
                         ]))
                    )
                )
                
                if is_requirement:
                    spatial_matches.append({
                        "text": line_text,
                        "match_type": "spatial",
                        "distance": line_info.get("distance", 0),
                        "keywords": [kw for kw in requirement_keywords if kw in line_lower]
                    })
            
            # Stage 2: Semantic matching for all potential requirements on the page - much more inclusive
            requirement_lines = [
                line for line in text_lines 
                if len(line) > 12 and (
                    # Strong requirement indicators
                    any(keyword in line.lower() for keyword in [
                        "shall", "must", "should", "will", "requirement", "required", "necessary",
                        "certificate", "test", "report", "material", "specification", "compliance",
                        "equipment", "system", "process", "control", "parameter", "capacity",
                        "temperature", "pressure", "flow", "safety", "training", "documentation",
                        "maintenance", "utility", "design", "construction", "provide", "supply",
                        "deliver", "install", "ensure", "maintain", "monitor", "verify", "validate"
                    ]) or
                    # Any line with technical measurements
                    (re.search(r'\d+', line) and any(unit in line.lower() for unit in [
                        "kg", "g", "mg", "l", "ml", "m3", "m", "cm", "mm", "°c", "°f", "bar", "psi", 
                        "rpm", "v", "amp", "watt", "hz", "%", "ppm", "ppb"
                    ])) or
                    # Table content (if contains |)
                    ('|' in line and len(line.split('|')) >= 2 and 
                     not any(header in line.lower() for header in ["sr. no", "page no", "topics"]))
                )
            ]
            
            semantic_matches = semantic_matcher.find_best_semantic_matches(
                comment_text, 
                requirement_lines, 
                threshold=0.3  # Lower threshold for semantic matching
            )
            
            # Stage 3: Combine and rank all matches
            all_matches = []
            
            # Add spatial matches with bonus score
            for spatial_match in spatial_matches:
                all_matches.append({
                    "requirement_text": spatial_match["text"],
                    "similarity_score": 0.9,  # High score for spatial proximity
                    "match_type": "spatial+keyword",
                    "explanation": f"Spatially close, Keywords: {', '.join(spatial_match['keywords'])}",
                    "distance": spatial_match["distance"]
                })
            
            # Add semantic matches
            for semantic_match in semantic_matches:
                # Check if this requirement was already matched spatially
                already_matched = any(
                    match["requirement_text"] == semantic_match["requirement_text"] 
                    for match in all_matches
                )
                
                if not already_matched:
                    all_matches.append(semantic_match)
                else:
                    # Enhance existing spatial match with semantic info
                    for match in all_matches:
                        if match["requirement_text"] == semantic_match["requirement_text"]:
                            match["similarity_score"] = min(1.0, match["similarity_score"] + 0.1)
                            match["match_type"] = "spatial+semantic"
                            match["explanation"] += f" | {semantic_match['explanation']}"
            
            # Stage 4: Select the best match(es)
            if all_matches:
                # Sort by similarity score and take the best match
                all_matches.sort(key=lambda x: x["similarity_score"], reverse=True)
                best_match = all_matches[0]
                
                # Only include matches above a reasonable threshold
                if best_match["similarity_score"] >= 0.3:
                    mapped_comments.append({
                        "id": annotation["id"],
                        "text": comment_text,
                        "author": annotation.get("author", "Unknown"),
                        "type": annotation.get("type", "Comment"),
                        "associated_text": best_match["requirement_text"],
                        "page": annotation["page"],
                        "precision": best_match["match_type"],
                        "similarity_score": best_match["similarity_score"],
                        "match_explanation": best_match["explanation"]
                    })
            
            # Stage 5: Special handling for unmatched high-value annotations
            if not all_matches and len(comment_text) > 10:
                # Try broader semantic search with lower threshold
                broader_matches = semantic_matcher.find_best_semantic_matches(
                    comment_text, 
                    text_lines,  # Search all lines, not just requirements
                    threshold=0.2
                )
                
                if broader_matches:
                    best_broad_match = broader_matches[0]
                    mapped_comments.append({
                        "id": annotation["id"],
                        "text": comment_text,
                        "author": annotation.get("author", "Unknown"),
                        "type": annotation.get("type", "Comment"),
                        "associated_text": best_broad_match["requirement_text"],
                        "page": annotation["page"],
                        "precision": "semantic-broad",
                        "similarity_score": best_broad_match["similarity_score"],
                        "match_explanation": f"Broad semantic match: {best_broad_match['explanation']}"
                    })
        
        # Update page data with mapped comments
        page_data["comments"] = mapped_comments
    
    return pages_data


def extract_structured_content(uploaded_file) -> List[Dict[str, Any]]:
    """
    Returns a list of page/chunk dicts for PDF/DOCX/XLSX/TXT.
    """
    try:
        file_bytes = uploaded_file.read()
        uploaded_file.seek(0)
    except Exception as e:
        print(f"Error reading file: {e}")
        return []

    name = getattr(uploaded_file, "name", "").lower()
    pages = []

    try:
        # ---------------- PDF ----------------
        if name.endswith(".pdf"):
            # Try enhanced annotation extraction first
            if PDF_ANNOTATIONS_AVAILABLE:
                print("Using enhanced PDF extraction with annotation support...")
                pdf_data = extract_pdf_annotations_and_text(file_bytes)
                enhanced_pages = map_pdf_annotations_to_requirements(
                    pdf_data["pages"], 
                    pdf_data["annotations"]
                )
                
                # Convert to the expected format and add section detection
                for page_data in enhanced_pages:
                    # Add OCR fallback if text is sparse
                    content = page_data.get("content", "")
                    if len(content.strip()) < 100 and OCR_AVAILABLE:
                        print(f"PDF page {page_data['page_number']}: Low text content, attempting OCR.")
                        try:
                            images = convert_from_bytes(
                                file_bytes, 
                                first_page=page_data['page_number'], 
                                last_page=page_data['page_number'], 
                                dpi=200
                            )
                            if images:
                                ocr_text = pytesseract.image_to_string(images[0]) or ""
                                content += "\n\n" + ocr_text
                                page_data["content"] = content
                        except Exception as ocr_error:
                            print(f"OCR failed for page {page_data['page_number']}: {ocr_error}")
                    
                    # Add tables to content if present
                    tables = page_data.get("tables", [])
                    if tables:
                        page_data["content"] += "\n\n--- TABLES ---\n" + "\n\n".join(tables)
                    
                    pages.append(page_data)
            
            else:
                # Fallback to basic extraction without annotations
                print("PyMuPDF not available, using basic PDF extraction...")
                with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                    for i, page in enumerate(pdf.pages, start=1):
                        # Improved text extraction to handle line breaks better
                        text = clean_extracted_text(page.extract_text(x_tolerance=2, y_tolerance=2) or "")
                        
                        # Fallback to OCR if text is sparse and dependencies are available
                        if len(text.strip()) < 100 and OCR_AVAILABLE:
                            print(f"PDF page {i}: Low text content, attempting OCR.")
                            try:
                                images = convert_from_bytes(
                                    file_bytes, first_page=i, last_page=i, dpi=200
                                )
                                if images:
                                    ocr_text = clean_extracted_text(pytesseract.image_to_string(images[0]) or "")
                                    text += "\n\n" + ocr_text # Append OCR text
                            except Exception as ocr_error:
                                print(f"OCR failed for page {i}: {ocr_error}")
                                print("Ensure Tesseract and Poppler are installed and in your PATH.")

                        # Extract tables separately
                        tables = []
                        try:
                            raw_tables = page.extract_tables()
                            for tbl in raw_tables:
                                rows = [clean_extracted_text(" | ".join(map(str, row))) for row in tbl]
                                cleaned_rows = [row for row in rows if row]  # Remove empty rows after cleaning
                                if cleaned_rows:
                                    tables.append("\n".join(cleaned_rows))
                        except Exception:
                            pass
                        
                        content = text
                        if tables:
                            content += "\n\n--- TABLES ---\n" + "\n\n".join(tables)

                        pages.append({
                            "page_number": i,
                            "content": content.strip(),
                            "tables": tables,
                            "content_type": "pdf",
                            "comments": [],
                        })

        # ---------------- DOCX ----------------
        elif name.endswith(".docx"):
            doc = Document(io.BytesIO(file_bytes))
            
            # Use section-based extraction for better organization
            pages = extract_section_based_requirements(doc, file_bytes)

        # ---------------- XLSX ----------------
        elif name.endswith((".xlsx", ".xls")):
            wb = openpyxl.load_workbook(io.BytesIO(file_bytes), read_only=True, data_only=True)
            for sheetname in wb.sheetnames:
                ws = wb[sheetname]
                rows_text = [clean_extracted_text(" | ".join(str(c) if c is not None else "" for c in row)) for row in ws.iter_rows(values_only=True)]
                cleaned_rows = [row for row in rows_text if row]  # Remove empty rows after cleaning
                content = "\n".join(cleaned_rows)
                if content.strip():
                    pages.append({
                        "page_number": sheetname,
                        "content": content.strip(),
                        "tables": [content],
                        "content_type": "xlsx",
                        "comments": [],
                    })

        # ---------------- TXT ----------------
        elif name.endswith(".txt"):
            try:
                text = file_bytes.decode("utf-8")
            except UnicodeDecodeError:
                text = file_bytes.decode("latin-1", errors="ignore")
            pages.append({
                "page_number": 1,
                "content": text.strip(),
                "tables": [],
                "content_type": "txt",
                "comments": [],
            })

    except Exception as e:
        print(f"[extract_structured_content] error extracting '{name}': {e}")
        return []

    return pages


def extract_text_from_pdf(uploaded_file) -> str:
    """
    Extract complete text content from a PDF file including comments and annotations.
    
    Args:
        uploaded_file: Streamlit uploaded file object or file-like object
        
    Returns:
        Complete text content of the PDF including annotations
    """
    try:
        # Read file bytes
        if hasattr(uploaded_file, 'read'):
            file_bytes = uploaded_file.read()
            uploaded_file.seek(0)
        else:
            with open(uploaded_file, 'rb') as f:
                file_bytes = f.read()
        
        full_text = []
        
        # Use pdfplumber for main text extraction
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                # Extract main text
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    full_text.append(f"=== PAGE {page_num} ===")
                    full_text.append(page_text.strip())
        
        # Try to extract annotations/comments if PyMuPDF is available
        if PDF_ANNOTATIONS_AVAILABLE:
            try:
                pdf_doc = fitz.open(stream=file_bytes, filetype="pdf")
                
                for page_num in range(pdf_doc.page_count):
                    page = pdf_doc[page_num]
                    annotations = page.annots()
                    
                    if annotations:
                        full_text.append(f"\n=== COMMENTS/ANNOTATIONS PAGE {page_num + 1} ===")
                        
                        for annot in annotations:
                            annot_dict = annot.info
                            annot_type = annot_dict.get('type', 'Unknown')
                            content = annot_dict.get('content', '')
                            author = annot_dict.get('title', 'Unknown Author')
                            
                            if content and content.strip():
                                full_text.append(f"[{annot_type}] {author}: {content.strip()}")
                
                pdf_doc.close()
                
            except Exception as e:
                print(f"Warning: Could not extract PDF annotations: {e}")
        
        return "\n\n".join(full_text)
        
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""


def extract_text_from_excel(uploaded_file) -> str:
    """
    Extract complete text content from an Excel file including cell comments.
    
    Args:
        uploaded_file: Streamlit uploaded file object or file-like object
        
    Returns:
        Complete text content of the Excel file including comments
    """
    try:
        # Read file bytes
        if hasattr(uploaded_file, 'read'):
            file_bytes = uploaded_file.read()
            uploaded_file.seek(0)
        else:
            with open(uploaded_file, 'rb') as f:
                file_bytes = f.read()
        
        # Load workbook
        workbook = openpyxl.load_workbook(io.BytesIO(file_bytes), data_only=False)
        
        full_text = []
        
        for sheet_name in workbook.sheetnames:
            sheet = workbook[sheet_name]
            full_text.append(f"=== SHEET: {sheet_name} ===")
            
            # Extract cell values
            sheet_content = []
            for row in sheet.iter_rows():
                row_data = []
                for cell in row:
                    if cell.value is not None:
                        row_data.append(str(cell.value).strip())
                if row_data and any(row_data):  # Only add non-empty rows
                    sheet_content.append(" | ".join(row_data))
            
            if sheet_content:
                full_text.extend(sheet_content)
            
            # Extract cell comments
            comments_found = False
            for row in sheet.iter_rows():
                for cell in row:
                    if cell.comment:
                        if not comments_found:
                            full_text.append(f"\n=== COMMENTS IN {sheet_name} ===")
                            comments_found = True
                        
                        cell_ref = cell.coordinate
                        comment_text = cell.comment.text if hasattr(cell.comment, 'text') else str(cell.comment)
                        author = getattr(cell.comment, 'author', 'Unknown Author') if hasattr(cell.comment, 'author') else 'Unknown Author'
                        
                        full_text.append(f"[{cell_ref}] {author}: {comment_text}")
        
        workbook.close()
        return "\n\n".join(full_text)
        
    except Exception as e:
        print(f"Error extracting text from Excel: {e}")
        return ""


def extract_text_from_file(uploaded_file, filename: str) -> str:
    """
    Universal text extractor that handles different file types.
    
    Args:
        uploaded_file: Streamlit uploaded file object or file-like object
        filename: Name of the file to determine the type
        
    Returns:
        Complete text content including comments/annotations
    """
    file_extension = filename.lower().split('.')[-1]
    
    if file_extension in ['docx', 'doc']:
        return extract_text_from_docx(uploaded_file)
    elif file_extension == 'pdf':
        return extract_text_from_pdf(uploaded_file)
    elif file_extension in ['xlsx', 'xls']:
        return extract_text_from_excel(uploaded_file)
    else:
        print(f"Unsupported file type: {file_extension}")
        return ""
