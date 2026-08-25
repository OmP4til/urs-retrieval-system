"""
Backend selector for holistic requirement extraction.

Both processors expose `extract_requirements_holistically(full_document_text,
document_name)` returning the same requirement dicts, so callers can stay
backend-agnostic:

    from utils.processor_factory import get_processor
    processor = get_processor()
    reqs = processor.extract_requirements_holistically(text, name)

Switch backends with the EXTRACTION_BACKEND setting in config.py, or the
EXTRACTION_BACKEND environment variable ("unlimited_ocr" or "gemini").
"""

import os
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def get_processor(backend: Optional[str] = None) -> Any:
    """
    Build the configured extraction processor.

    Args:
        backend: "unlimited_ocr" or "gemini". Falls back to config/env when None.

    Raises:
        ValueError: unknown backend, or Gemini selected without an API key.
    """
    if backend is None:
        try:
            from config import EXTRACTION_BACKEND
            backend = EXTRACTION_BACKEND
        except ImportError:
            backend = os.getenv("EXTRACTION_BACKEND", "unlimited_ocr")

    backend = (backend or "").strip().lower()

    if backend in ("unlimited_ocr", "unlimited-ocr", "ocr", "local"):
        from utils.unlimited_ocr_processor import build_processor_from_env
        logger.info("Using Unlimited-OCR extraction backend (local model)")
        return build_processor_from_env()

    if backend == "gemini":
        from utils.gemini_processor import GeminiProcessor
        try:
            from config import GEMINI_API_KEY, GEMINI_MODEL
        except ImportError:
            GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
            GEMINI_MODEL = "models/gemini-3.6-flash"

        if not GEMINI_API_KEY:
            raise ValueError("EXTRACTION_BACKEND=gemini but GEMINI_API_KEY is not set")

        logger.info("Using Gemini extraction backend (API)")
        return GeminiProcessor(api_key=GEMINI_API_KEY, model_name=GEMINI_MODEL)

    raise ValueError(
        "Unknown EXTRACTION_BACKEND: {!r} (expected 'unlimited_ocr' or 'gemini')".format(backend)
    )
