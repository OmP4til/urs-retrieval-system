"""
Unlimited-OCR integration for requirement extraction and preprocessing.

Runs Baidu's Unlimited-OCR vision-language model locally - no API, no key, no
data leaving the machine.

  code:    https://github.com/baidu/Unlimited-OCR
  weights: https://huggingface.co/baidu/Unlimited-OCR

Two stages:
  1. Document parsing - the OCR VLM turns page images into structured markdown.
  2. Requirement structuring - the markdown is turned into the same requirement
     dicts the Gemini path produced, so the rest of the pipeline is unchanged.

Stage 2 is deterministic/rule-based: Unlimited-OCR is a document *parsing* model,
it does not do free-form reasoning, so the semantic judgement Gemini used to
provide is reconstructed from document structure and requirement language.
"""

import os
import re
import glob
import shutil
import tempfile
import logging
from typing import List, Dict, Any, Optional, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                   handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "baidu/Unlimited-OCR"

# Prompts from the upstream README.
SINGLE_IMAGE_PROMPT = "<image>document parsing."
MULTI_PAGE_PROMPT = "<image>Multi page parsing."

# Layout markers emitted by the model, stripped during post-processing.
DET_RE = re.compile(r'<\|det\|>([^<\s]+)(?:\s*\[[^\]]*\])?\s*<\|/det\|>(.*)', re.DOTALL)
SPECIAL_TOKEN_RE = re.compile(r'<\|(?:begin|end)_of_[a-z_]+\|>|<\|/?(?:ref|det|grounding)\|>')

IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff')


def remove_det(raw: str) -> str:
    """
    Strip <|det|>type [bbox]<|/det|> markers, group lines belonging to the
    same block with a newline, and separate different blocks with a blank line.

    Taken from the post-processing snippet in the Unlimited-OCR README.
    """
    blocks: List[List[str]] = []
    cur: Optional[List[str]] = None
    for line in raw.splitlines():
        line = line.rstrip()
        if not line:
            continue
        m = DET_RE.match(line)
        if m:
            category, content = m.group(1).strip(), m.group(2).strip()
            if category == 'image':
                continue
            if cur is not None:
                blocks.append(cur)
            cur = [content] if content else []
            continue
        if cur is None:
            cur = []
        cur.append(line)
    if cur is not None:
        blocks.append(cur)
    return '\n\n'.join('\n'.join(b) for b in blocks).strip()


def pdf_to_images(pdf_path: str, dpi: int = 300, out_dir: Optional[str] = None) -> List[str]:
    """Rasterise every PDF page to PNG. Returns the page image paths in order."""
    import fitz  # PyMuPDF

    doc = fitz.open(pdf_path)
    tmp_dir = out_dir or tempfile.mkdtemp(prefix='pdf_ocr_')
    os.makedirs(tmp_dir, exist_ok=True)
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    paths = []
    try:
        for i, page in enumerate(doc):
            out = os.path.join(tmp_dir, 'page_{:04d}.png'.format(i + 1))
            page.get_pixmap(matrix=mat).save(out)
            paths.append(out)
    finally:
        doc.close()
    return paths


def convert_doc_to_docx(src_path: str, out_dir: Optional[str] = None) -> str:
    """
    Convert a legacy Word 97-2003 .doc (OLE2) to .docx.

    python-docx only reads OOXML, so a real .doc yields an empty string and the
    document looks unreadable. Word is driven over COM through PowerShell, which
    avoids a pywin32 dependency. Converting to .docx rather than to PDF keeps the
    exact text layer and preserves tracked comments, so the comment-pairing path
    keeps working.

    Returns the path to the converted .docx.

    Raises:
        RuntimeError: Word is unavailable or the conversion failed.
    """
    import subprocess

    src = os.path.abspath(src_path)
    if not os.path.exists(src):
        raise RuntimeError("File not found: " + src)

    target_dir = out_dir or tempfile.mkdtemp(prefix='doc2docx_')
    os.makedirs(target_dir, exist_ok=True)
    dst = os.path.join(target_dir, os.path.splitext(os.path.basename(src))[0] + '.docx')

    # wdFormatDocumentDefault = 16 (.docx). Open read-only so the source is untouched.
    script = (
        "$ErrorActionPreference='Stop';"
        "$w=New-Object -ComObject Word.Application;"
        "$w.Visible=$false;$w.DisplayAlerts=0;"
        "try{"
        f"$d=$w.Documents.Open('{src}',$false,$true);"
        f"$d.SaveAs2('{dst}',16);"
        "$d.Close($false);"
        "Write-Output 'OK'"
        "}catch{Write-Output ('ERR: '+$_.Exception.Message)}"
        "finally{$w.Quit()}"
    )

    logger.info("Converting legacy .doc via Word COM: %s", os.path.basename(src))
    try:
        proc = subprocess.run(['powershell', '-NoProfile', '-Command', script],
                              capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise RuntimeError("Could not run Word for .doc conversion: {}".format(e))

    output = (proc.stdout or '').strip()
    if 'OK' not in output or not os.path.exists(dst):
        raise RuntimeError(
            "Failed to convert '{}' from legacy .doc. Word reported: {}. "
            "Re-save the file as .docx in Word and upload that instead.".format(
                os.path.basename(src), output or (proc.stderr or '').strip()[:200])
        )

    logger.info("Converted to %s", dst)
    return dst


def _select_device_and_dtype() -> Tuple[str, Any]:
    """
    Pick the best device/dtype this machine can actually run.

    bfloat16 needs Ampere (SM 8.0+); Turing and older get float16 on GPU and
    float32 on CPU. Falls back to CPU when CUDA is missing or VRAM is too small
    for the ~3.4 GB of weights plus activations.
    """
    import torch

    if not torch.cuda.is_available():
        logger.warning("CUDA not available - running Unlimited-OCR on CPU (float32). "
                       "This works but is slow; expect minutes per page.")
        return 'cpu', torch.float32

    props = torch.cuda.get_device_properties(0)
    total_gb = props.total_memory / 1024 ** 3
    free_bytes, _ = torch.cuda.mem_get_info(0)
    free_gb = free_bytes / 1024 ** 3

    # 3.34 GB of BF16 weights + activations/KV cache at 32k context.
    if free_gb < 6.0:
        logger.warning(
            "GPU '%s' has %.1f GB free of %.1f GB total - not enough for Unlimited-OCR "
            "(needs ~6 GB+). Falling back to CPU (float32).",
            props.name, free_gb, total_gb)
        return 'cpu', torch.float32

    if props.major >= 8:
        return 'cuda', torch.bfloat16

    logger.info("GPU '%s' is pre-Ampere (SM %d.%d) - using float16 instead of bfloat16.",
                props.name, props.major, props.minor)
    return 'cuda', torch.float16


class UnlimitedOCRProcessor:
    """
    Processes documents using Baidu's Unlimited-OCR for requirement extraction.

    Emits the same requirement dicts the Gemini branch produced, so the
    database, embeddings, and matching code work unchanged.
    """

    def __init__(self,
                 model_name: str = DEFAULT_MODEL_NAME,
                 device: Optional[str] = None,
                 dtype: Optional[Any] = None,
                 dpi: int = 300,
                 image_mode: str = 'base',
                 max_length: int = 32768,
                 cache_dir: Optional[str] = None,
                 lazy: bool = True):
        """
        Args:
            model_name: HuggingFace model id or a local path to the weights.
            device: 'cuda' / 'cpu'. Auto-detected when None.
            dtype: torch dtype. Auto-detected when None.
            dpi: rasterisation DPI for PDF pages.
            image_mode: 'base' (image_size=1024, no crop) or 'gundam'
                (base_size=1024, image_size=640, crop_mode=True). Multi-page
                parsing upstream only supports 'base'.
            max_length: generation cap, matches the upstream context length.
            cache_dir: HuggingFace cache directory override.
            lazy: when True the weights are only loaded on first use.
        """
        self.model_name = model_name
        self.dpi = dpi
        self.image_mode = image_mode
        self.max_length = max_length
        self.cache_dir = cache_dir

        self._device = device
        self._dtype = dtype
        self.model = None
        self.tokenizer = None

        logger.info("Configured Unlimited-OCR processor with model: %s", model_name)
        if not lazy:
            self.load()

    # ------------------------------------------------------------------ #
    # Model loading
    # ------------------------------------------------------------------ #
    def load(self):
        """Load tokenizer + weights. Idempotent."""
        if self.model is not None:
            return

        from transformers import AutoModel, AutoTokenizer

        if self._device is None or self._dtype is None:
            device, dtype = _select_device_and_dtype()
            self._device = self._device or device
            self._dtype = self._dtype or dtype

        logger.info("Loading %s onto %s (%s)...", self.model_name, self._device, self._dtype)

        kwargs = {"trust_remote_code": True}
        if self.cache_dir:
            kwargs["cache_dir"] = self.cache_dir

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, **kwargs)
        self.model = AutoModel.from_pretrained(
            self.model_name,
            use_safetensors=True,
            torch_dtype=self._dtype,
            **kwargs,
        )
        self.model = self.model.eval()
        if self._device == 'cuda':
            self.model = self.model.cuda()

        logger.info("Unlimited-OCR weights loaded")

    @property
    def device(self) -> Optional[str]:
        return self._device

    # ------------------------------------------------------------------ #
    # Stage 1: document parsing
    # ------------------------------------------------------------------ #
    def _read_saved_results(self, output_path: str) -> str:
        """Model writes markdown per page to output_path; concatenate in order."""
        files = sorted(
            glob.glob(os.path.join(output_path, '**', '*.mmd'), recursive=True) +
            glob.glob(os.path.join(output_path, '**', '*.md'), recursive=True) +
            glob.glob(os.path.join(output_path, '**', '*.txt'), recursive=True)
        )
        parts = []
        for path in files:
            try:
                with open(path, 'r', encoding='utf-8', errors='replace') as f:
                    text = f.read().strip()
                if text:
                    parts.append(text)
            except OSError as e:
                logger.warning("Could not read OCR output %s: %s", path, e)
        return '\n\n'.join(parts)

    def _normalise(self, raw: Any, output_path: str) -> str:
        """Turn whatever infer() returned into clean markdown."""
        if isinstance(raw, (list, tuple)):
            raw = '\n\n'.join(str(x) for x in raw if x)
        text = raw.strip() if isinstance(raw, str) else ''

        if not text:
            text = self._read_saved_results(output_path)

        text = SPECIAL_TOKEN_RE.sub('', remove_det(text))
        return re.sub(r'\n{3,}', '\n\n', text).strip()

    def parse_images(self, image_files: List[str], output_path: Optional[str] = None) -> str:
        """Parse one or more page images into markdown."""
        if not image_files:
            return ""

        self.load()
        tmp_out = output_path or tempfile.mkdtemp(prefix='uocr_out_')
        os.makedirs(tmp_out, exist_ok=True)

        try:
            if len(image_files) == 1 and self.image_mode == 'gundam':
                logger.info("Parsing 1 page (gundam mode)...")
                raw = self.model.infer(
                    self.tokenizer,
                    prompt=SINGLE_IMAGE_PROMPT,
                    image_file=image_files[0],
                    output_path=tmp_out,
                    base_size=1024, image_size=640, crop_mode=True,
                    max_length=self.max_length,
                    no_repeat_ngram_size=35, ngram_window=128,
                    save_results=True,
                )
            elif len(image_files) == 1:
                logger.info("Parsing 1 page (base mode)...")
                raw = self.model.infer(
                    self.tokenizer,
                    prompt=SINGLE_IMAGE_PROMPT,
                    image_file=image_files[0],
                    output_path=tmp_out,
                    base_size=1024, image_size=1024, crop_mode=False,
                    max_length=self.max_length,
                    no_repeat_ngram_size=35, ngram_window=128,
                    save_results=True,
                )
            else:
                # Multi-page parsing upstream only supports base / image_size=1024.
                logger.info("Parsing %d pages in one shot (base mode)...", len(image_files))
                raw = self.model.infer_multi(
                    self.tokenizer,
                    prompt=MULTI_PAGE_PROMPT,
                    image_files=image_files,
                    output_path=tmp_out,
                    image_size=1024,
                    max_length=self.max_length,
                    no_repeat_ngram_size=35, ngram_window=1024,
                    save_results=True,
                )

            text = self._normalise(raw, tmp_out)
            logger.info("Parsed %s characters of markdown", format(len(text), ','))
            return text
        finally:
            if output_path is None:
                shutil.rmtree(tmp_out, ignore_errors=True)

    def parse_pdf(self, pdf_path: str, output_path: Optional[str] = None) -> str:
        """Rasterise a PDF and parse every page with Unlimited-OCR."""
        tmp_img_dir = tempfile.mkdtemp(prefix='uocr_pages_')
        try:
            pages = pdf_to_images(pdf_path, dpi=self.dpi, out_dir=tmp_img_dir)
            logger.info("Rendered %d page(s) from %s at %d DPI",
                        len(pages), os.path.basename(pdf_path), self.dpi)
            return self.parse_images(pages, output_path=output_path)
        finally:
            shutil.rmtree(tmp_img_dir, ignore_errors=True)

    def parse_document(self, file_path: str, output_path: Optional[str] = None) -> str:
        """
        Parse any supported document into markdown.

        PDFs and images go through Unlimited-OCR. DOCX has a native text layer,
        so it is read directly rather than rasterised - OCR would only lose
        fidelity there.
        """
        ext = os.path.splitext(file_path)[1].lower()

        if ext == '.pdf':
            return self.parse_pdf(file_path, output_path=output_path)
        if ext in IMAGE_EXTS:
            return self.parse_images([file_path], output_path=output_path)
        if ext == '.docx':
            logger.info("DOCX has a native text layer - reading it directly instead of running OCR")
            from utils.extractors import extract_text_from_docx
            return extract_text_from_docx(file_path)

        if ext == '.doc':
            # Legacy OLE2 Word: convert to .docx first, then read the text layer.
            from utils.extractors import extract_text_from_docx
            converted = convert_doc_to_docx(file_path)
            return extract_text_from_docx(converted)

        raise ValueError("Unsupported file type for Unlimited-OCR parsing: " + ext)

    # ------------------------------------------------------------------ #
    # Stage 2: requirement structuring (Gemini-compatible output)
    # ------------------------------------------------------------------ #
    def extract_requirements_holistically(self,
                                          full_document_text: str,
                                          document_name: str = "Technical Document") -> List[Dict[str, Any]]:
        """
        Extract requirements from parsed document text.

        Return shape matches what the Gemini branch produced, so downstream
        storage and matching are unchanged.
        """
        if not full_document_text or len(full_document_text.strip()) < 50:
            logger.warning("Document text too short for analysis")
            return []

        logger.info("Structuring requirements from %d characters of parsed text",
                    len(full_document_text))
        requirements = RequirementStructurer().structure(full_document_text, document_name)
        logger.info("Extracted %d requirements from %s", len(requirements), document_name)
        return requirements

    def extract_comments_and_responses(self,
                                       full_document_text: str,
                                       document_name: str = "Document",
                                       file_bytes: Optional[bytes] = None) -> List[Dict[str, Any]]:
        """
        Extract commented requirements and their comments/responses.

        Comments come straight from the DOCX comment parts - that is exact
        structural data, so no model is involved and nothing is guessed.
        """
        if not file_bytes or not document_name.lower().endswith(('.docx', '.doc')):
            logger.warning("Comment extraction needs DOCX file bytes; got %s - returning no comments",
                           document_name)
            return []

        structured = _load_docx_comments(file_bytes)
        if not structured:
            logger.info("No comments found in %s", document_name)
            return []

        results = []
        for i, (commented_text, comments) in enumerate(structured.items()):
            validated = []
            for c in comments:
                text = (c.get('text') or '').strip()
                if text:
                    validated.append({
                        'comment_text': text,
                        'author': (c.get('author') or 'Unknown').strip(),
                        'comment_type': 'docx_structured',
                    })
            if not validated or not commented_text.strip():
                continue
            results.append({
                'id': 'UOCR_COMMENT_REQ_{}'.format(i + 1),
                'requirement_text': commented_text.strip(),
                'comments': validated,
                'page_reference': 'Unknown',
                'extracted_by': 'unlimited_ocr_docx_comments',
                'document_name': document_name,
            })

        logger.info("Extracted %d commented requirements from %s", len(results), document_name)
        return results

    def extract_requirements_with_comments_holistically(self,
                                                        full_document_text: str,
                                                        document_name: str = "Document",
                                                        file_bytes: Optional[bytes] = None) -> Dict[str, Any]:
        """
        Extract requirements and pair them with their DOCX comments.

        Returns {'requirements', 'requirement_comment_pairs', 'total_comments'}.
        """
        requirements = self.extract_requirements_holistically(full_document_text, document_name)

        pairs = [{'requirement': req, 'comments': []} for req in requirements]

        structured = {}
        if file_bytes and document_name.lower().endswith(('.docx', '.doc')):
            structured = _load_docx_comments(file_bytes)
            logger.info("Found %d commented text segments in DOCX", len(structured))

        total_comments = 0
        if structured:
            total_comments = _pair_comments_to_requirements(pairs, structured)

        return {
            'requirements': requirements,
            'requirement_comment_pairs': pairs,
            'total_comments': total_comments,
            'comments': [],
            'comment_mappings': {},
        }

    def extract_requirements_from_file(self,
                                       file_path: str,
                                       document_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Convenience: parse a document with OCR, then structure its requirements."""
        text = self.parse_document(file_path)
        return self.extract_requirements_holistically(
            text, document_name or os.path.basename(file_path))


# ---------------------------------------------------------------------- #
# DOCX comment handling (structural - no model involved)
# ---------------------------------------------------------------------- #
COMMENT_MATCH_THRESHOLD = 0.75


def _load_docx_comments(file_bytes: bytes) -> Dict[str, List[Dict[str, Any]]]:
    """Read the DOCX comment parts, mapping commented text -> comment records."""
    try:
        from utils.extractors import get_docx_comments_with_text_mapping
        return get_docx_comments_with_text_mapping(file_bytes) or {}
    except Exception as e:
        logger.warning("Could not extract structured comments: %s", e)
        return {}


def _pair_comments_to_requirements(pairs: List[Dict[str, Any]],
                                   structured: Dict[str, List[Dict[str, Any]]]) -> int:
    """
    Attach DOCX comments to the requirement each one belongs to.

    Same strategy as the Gemini path: exact substring match first, then semantic
    similarity above COMMENT_MATCH_THRESHOLD, falling back to word overlap when
    the embedding matcher is unavailable. Returns the number of comments attached.
    """
    try:
        from utils.extractors import get_semantic_matcher
        matcher = get_semantic_matcher()
    except Exception:
        matcher = None
        logger.warning("Semantic matcher not available, using word-overlap matching")

    attached = 0
    for pair in pairs:
        req_text = (pair['requirement'].get('text') or '').strip()
        if not req_text:
            continue

        req_lower = req_text.lower()
        best = None
        best_similarity = 0.0

        for docx_text, docx_comments in structured.items():
            docx_lower = docx_text.lower().strip()
            if not docx_lower:
                continue

            if docx_lower in req_lower or req_lower in docx_lower:
                similarity = 1.0
            elif matcher:
                similarity = matcher.calculate_semantic_similarity(req_text, docx_text)
            else:
                req_words = set(req_lower.split())
                docx_words = set(docx_lower.split())
                similarity = len(req_words & docx_words) / max(len(req_words | docx_words), 1)

            if similarity > best_similarity and similarity > COMMENT_MATCH_THRESHOLD:
                best_similarity = similarity
                best = (docx_text, docx_comments, similarity)

        if not best:
            continue

        matched_text, matched_comments, score = best
        logger.info("Matched requirement '%s...' to DOCX text (similarity %.2f)",
                    req_text[:60], score)

        existing = {c.get('comment_text', '') for c in pair['comments']}
        for docx_comment in matched_comments:
            comment_text = (docx_comment.get('text') or '').strip()
            if not comment_text or comment_text in existing:
                continue
            pair['comments'].append({
                'comment_text': comment_text,
                'author': (docx_comment.get('author') or 'Unknown').strip(),
                'comment_type': 'docx_structured',
                'confidence': score,
                'source': 'docx_structured',
                'matched_to': matched_text[:100],
            })
            existing.add(comment_text)
            attached += 1

    return attached


# ---------------------------------------------------------------------- #
# Rule-based requirement structuring
# ---------------------------------------------------------------------- #
class RequirementStructurer:
    """
    Turns parsed document markdown into requirement records.

    Replaces the semantic judgement Gemini used to supply with explicit,
    inspectable rules over document structure and requirement language.
    """

    # Modal verbs, strongest first. Drives both detection and priority.
    MODALS = [
        (re.compile(r'\bshall\b', re.I), 'critical', 0.95),
        (re.compile(r'\bmust\b', re.I), 'critical', 0.95),
        (re.compile(r'\b(?:is|are) required to\b', re.I), 'critical', 0.93),
        (re.compile(r'\brequired\b|\bmandatory\b', re.I), 'high', 0.88),
        (re.compile(r'\bshould\b', re.I), 'high', 0.82),
        (re.compile(r'\b(?:to be|will be) provided\b', re.I), 'high', 0.85),
        (re.compile(r'\bwill\b', re.I), 'medium', 0.75),
        (re.compile(r'\bmay\b|\bcan be\b|\boptional\b|\bpreferred\b|\bdesirable\b', re.I), 'low', 0.65),
    ]

    # Non-modal signals: spec sheets and tables rarely use "shall".
    SPEC_PATTERNS = [
        re.compile(r'\b(?:min|max|minimum|maximum|not less than|not more than|at least|up to)\b', re.I),
        re.compile(r'\b(?:capacity|accuracy|tolerance|range|rating|speed|pressure|temperature|'
                   r'flow|volume|voltage|power|dimension|weight|material)\b\s*[:\-]', re.I),
        re.compile(r'[\d.]+\s*(?:mm|cm|kg|g|ml|bar|psi|kw|kva|hz|rpm|ppm|lpm|cfm|'
                   r'°c|deg\s*c|%|µm|um|nm|mbar)\b', re.I),
        re.compile(r'\bcompl(?:y|iant|iance)\b|\bin accordance with\b|\bas per\b|\bconform', re.I),
    ]

    CATEGORIES = [
        ('safety', ['safety', 'hazard', 'interlock', 'emergency stop', 'e-stop', 'guard', 'protective',
                    'explosion', 'atex', 'risk', 'lockout', 'alarm', 'fail-safe', 'failsafe']),
        ('compliance', ['gmp', 'cgmp', 'cfr', 'part 11', 'gamp', 'iso ', 'iec ', 'astm',
                        'asme', 'usp', 'fda', 'regulatory', 'audit trail', 'validation', 'qualification',
                        'compliance', 'directive', 'standard']),
        ('performance', ['capacity', 'throughput', 'efficiency', 'output', 'yield', 'accuracy', 'precision',
                         'speed', 'cycle time', 'uptime', 'availability', 'tolerance', 'repeatability']),
        ('material', ['material', 'stainless', 'ss316', 'ss 316', 'aisi', 'contact part', 'gasket', 'seal',
                      'elastomer', 'ptfe', 'silicone', 'surface finish', 'corrosion']),
        ('electrical', ['electrical', 'voltage', 'power supply', 'phase', 'earthing', 'grounding', 'motor',
                        'vfd', 'panel', 'wiring', 'cable', 'ip55', 'ip65', 'kw', 'kva']),
        ('control', ['plc', 'hmi', 'scada', 'software', 'recipe', 'automation', 'control system', 'sensor',
                     'interface', 'set point', 'setpoint', 'report', 'audit', 'user access',
                     'password', 'login', 'batch record']),
        ('environmental', ['environment', 'humidity', 'ambient', 'noise', 'dust',
                           'emission', 'effluent', 'ventilation', 'hvac', 'clean room', 'cleanroom']),
        ('maintenance', ['maintenance', 'spare', 'service', 'lubricat', 'calibrat', 'cleaning', 'cip',
                         'sip', 'washdown', 'accessib', 'replace', 'wear part']),
        ('documentation', ['document', 'manual', 'drawing', 'certificate', 'datasheet', 'sop',
                           'p&id', 'traceability', 'as-built']),
        ('operational', ['operator', 'operation', 'start-up', 'startup', 'shutdown', 'changeover',
                         'loading', 'unloading', 'handling', 'procedure', 'training']),
        ('design', ['design', 'construction', 'dimension', 'layout', 'footprint', 'mounting', 'assembly',
                    'configuration', 'geometry', 'structure']),
    ]

    STANDARD_RE = re.compile(
        r'\b(?:ISO|IEC|EN|DIN|ASTM|ASME|ANSI|BS|NFPA|UL|USP|IEEE|API)\s?[-–]?\s?\d[\w.\-]*'
        r'|\b21\s?CFR\s?(?:Part\s?)?\d+'
        r'|\bGAMP\s?5?\b|\bcGMP\b|\bGMP\b|\bATEX\b|\bCE\s?mark(?:ing|ed)?\b|\bEU\s?GMP\b',
        re.I)

    PARAM_RE = re.compile(
        r'(?P<value>[<>≤≥]?\s?\d+(?:[.,]\d+)?(?:\s?[-–±]\s?\d+(?:[.,]\d+)?)?)\s*'
        r'(?P<unit>mm|cm|kg|mg|ml|mbar|bar|psi|kW|kVA|Hz|rpm|°C|degC|%|ppm|µm|um|nm|'
        r'lpm|cfm|m3/h|m³/h|min|sec|hrs?|inch|[LVAWgs])\b')

    HEADING_RE = re.compile(r'^\s*(?:#{1,6}\s+(?P<h>.+)|(?P<num>\d+(?:\.\d+)*)[.)]?\s+(?P<t>[A-Z][^.]{2,80}))\s*$')
    BULLET_RE = re.compile(r'^\s*(?:[-*+•·]|\(?[a-z0-9]{1,3}[.)])\s+')
    TABLE_ROW_RE = re.compile(r'^\s*\|(?P<body>.+)\|\s*$')
    NOISE_RE = re.compile(r'^\s*(?:page\s+\d+|\d+\s*/\s*\d+|rev\.?\s*\d+|confidential|table of contents)\s*$', re.I)

    MIN_LEN = 25
    MAX_LEN = 1200

    def structure(self, text: str, document_name: str) -> List[Dict[str, Any]]:
        candidates = self._collect_candidates(text)

        requirements: List[Dict[str, Any]] = []
        seen = set()

        for statement, section in candidates:
            priority, confidence = self._score(statement)
            if priority is None:
                continue

            key = re.sub(r'\W+', '', statement.lower())[:160]
            if key in seen:
                continue
            seen.add(key)

            category, subcategory = self._categorise(statement, section)
            requirements.append({
                'id': 'UOCR_REQ_{:03d}'.format(len(requirements) + 1),
                'text': statement,
                'category': category,
                'subcategory': subcategory,
                'confidence': confidence,
                'source_context': section or document_name,
                'priority': priority,
                'technical_parameters': self._parameters(statement),
                'dependencies': [],
                'compliance_standards': self._standards(statement),
                'notes': '',
                'extraction_method': 'unlimited_ocr_holistic',
            })

        self._link_dependencies(requirements)
        return requirements

    # -- candidate collection ------------------------------------------ #
    def _collect_candidates(self, text: str) -> List[Tuple[str, str]]:
        """Walk the markdown, tracking the current section heading."""
        candidates: List[Tuple[str, str]] = []
        state = {'section': ''}
        buffer: List[str] = []

        def flush():
            if not buffer:
                return
            para = ' '.join(buffer).strip()
            del buffer[:]
            for sentence in self._split_sentences(para):
                candidates.append((sentence, state['section']))

        for line in text.splitlines():
            stripped = line.strip()

            if not stripped or self.NOISE_RE.match(stripped):
                flush()
                continue

            heading = self.HEADING_RE.match(stripped)
            if heading:
                flush()
                if heading.group('h'):
                    state['section'] = heading.group('h').strip()
                else:
                    state['section'] = (heading.group('num') + ' ' + heading.group('t')).strip()
                continue

            row = self.TABLE_ROW_RE.match(stripped)
            if row:
                flush()
                cells = [c.strip() for c in row.group('body').split('|')]
                cells = [c for c in cells if c and not set(c) <= set('-: ')]
                if len(cells) >= 2:
                    candidates.append((' - '.join(cells), state['section']))
                elif cells:
                    candidates.append((cells[0], state['section']))
                continue

            if self.BULLET_RE.match(stripped):
                flush()
                candidates.append((self.BULLET_RE.sub('', stripped).strip(), state['section']))
                continue

            buffer.append(stripped)

        flush()

        cleaned = []
        for raw, section in candidates:
            statement = self._clean(raw)
            if statement:
                cleaned.append((statement, section))
        return cleaned

    @staticmethod
    def _split_sentences(paragraph: str) -> List[str]:
        parts = re.split(r'(?<=[.;:])\s+(?=[A-Z(\d])', paragraph)
        return [p.strip() for p in parts if p.strip()]

    def _clean(self, s: str) -> str:
        s = re.sub(r'\*\*|__|`', '', s).strip()
        s = re.sub(r'\s{2,}', ' ', s)
        s = s.strip(' .;:-')
        if len(s) < self.MIN_LEN or len(s) > self.MAX_LEN:
            return ''
        if not re.search(r'[A-Za-z]{3}', s):
            return ''
        return s

    # -- scoring -------------------------------------------------------- #
    def _score(self, statement: str) -> Tuple[Optional[str], float]:
        for pattern, priority, confidence in self.MODALS:
            if pattern.search(statement):
                if priority == 'high' and re.search(r'\bcritical\b|\bsafety\b|\bgmp\b|\bmandator',
                                                    statement, re.I):
                    priority = 'critical'
                    confidence = min(0.97, confidence + 0.05)
                return priority, round(confidence, 2)

        hits = sum(1 for p in self.SPEC_PATTERNS if p.search(statement))
        if hits >= 2:
            return 'medium', 0.75
        if hits == 1 and len(statement) >= 40:
            return 'medium', 0.65
        return None, 0.0

    def _categorise(self, statement: str, section: str) -> Tuple[str, str]:
        haystack = (section + ' ' + statement).lower()
        scores: Dict[str, int] = {}
        for name, keywords in self.CATEGORIES:
            hit = sum(1 for kw in keywords if kw in haystack)
            if hit:
                scores[name] = hit
        if not scores:
            return 'functional', ''

        ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        category = ordered[0][0]
        subcategory = ordered[1][0] if len(ordered) > 1 else (section[:60] if section else '')
        return category, subcategory

    def _parameters(self, statement: str) -> Dict[str, str]:
        params: Dict[str, str] = {}
        for i, m in enumerate(self.PARAM_RE.finditer(statement), start=1):
            value = re.sub(r'\s+', '', m.group('value'))
            params['param_{}'.format(i)] = value + ' ' + m.group('unit')
        return params

    def _standards(self, statement: str) -> List[str]:
        found = [re.sub(r'\s+', ' ', m.group(0)).strip() for m in self.STANDARD_RE.finditer(statement)]
        return sorted(set(found))

    def _link_dependencies(self, requirements: List[Dict[str, Any]]) -> None:
        """Link requirements that share a section heading."""
        by_section: Dict[str, List[str]] = {}
        for req in requirements:
            by_section.setdefault(req['source_context'], []).append(req['id'])

        for req in requirements:
            siblings = [rid for rid in by_section.get(req['source_context'], []) if rid != req['id']]
            req['dependencies'] = siblings[:5]


def build_processor_from_env() -> UnlimitedOCRProcessor:
    """Construct a processor using the values in config.py / the environment."""
    try:
        from config import (UNLIMITED_OCR_MODEL, UNLIMITED_OCR_DPI,
                            UNLIMITED_OCR_IMAGE_MODE, UNLIMITED_OCR_MAX_LENGTH,
                            UNLIMITED_OCR_DEVICE, UNLIMITED_OCR_CACHE_DIR)
    except ImportError:
        UNLIMITED_OCR_MODEL = os.getenv('UNLIMITED_OCR_MODEL', DEFAULT_MODEL_NAME)
        UNLIMITED_OCR_DPI = int(os.getenv('UNLIMITED_OCR_DPI', '300'))
        UNLIMITED_OCR_IMAGE_MODE = os.getenv('UNLIMITED_OCR_IMAGE_MODE', 'base')
        UNLIMITED_OCR_MAX_LENGTH = int(os.getenv('UNLIMITED_OCR_MAX_LENGTH', '32768'))
        UNLIMITED_OCR_DEVICE = os.getenv('UNLIMITED_OCR_DEVICE') or None
        UNLIMITED_OCR_CACHE_DIR = os.getenv('UNLIMITED_OCR_CACHE_DIR') or None

    return UnlimitedOCRProcessor(
        model_name=UNLIMITED_OCR_MODEL,
        device=UNLIMITED_OCR_DEVICE,
        dpi=UNLIMITED_OCR_DPI,
        image_mode=UNLIMITED_OCR_IMAGE_MODE,
        max_length=UNLIMITED_OCR_MAX_LENGTH,
        cache_dir=UNLIMITED_OCR_CACHE_DIR,
    )
