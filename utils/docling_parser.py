"""
Docling-based document parsing.

Docling (https://github.com/docling-project/docling) reads DOCX, PDF, PPTX,
XLSX and HTML into a structured document model. This module converts that model
into the same "one line per paragraph, cells joined with ' | '" text layout the
rest of the pipeline already expects, so RequirementStructurer needs no changes.

Why the structured model rather than export_to_markdown(): the markdown export
pads table columns to a fixed width and packs several source rows into one very
wide line (median ~1,200 characters on a real URS, up to 10,000), which destroys
per-requirement granularity. Iterating tables cell by cell keeps one line per
source row.

Comments are NOT read here. Docling does not expose DOCX comment parts, so
comment extraction and pairing continue to come from
extractors.get_docx_comments_with_text_mapping(), independent of the parser.
"""

import io
import os
import logging
import tempfile
from typing import List, Optional

logger = logging.getLogger(__name__)

_CONVERTER = None


def _get_converter():
    """Build the DocumentConverter once - construction is not free."""
    global _CONVERTER
    if _CONVERTER is None:
        from docling.document_converter import DocumentConverter
        _CONVERTER = DocumentConverter()
        logger.info("Docling converter initialised")
    return _CONVERTER


def is_available() -> bool:
    """True when docling is installed and importable."""
    try:
        import docling.document_converter  # noqa: F401
        return True
    except Exception:
        return False


def _rows_from_table(table) -> List[str]:
    """
    Turn one Docling table into "cell | cell | cell" lines, one per source row.

    Cells are ordered by column offset. A merged cell repeats across the columns
    it spans, so consecutive duplicates are collapsed - the same guard the
    python-docx reader needs.
    """
    rows = {}
    for cell in table.data.table_cells:
        text = (cell.text or '').strip()
        if not text:
            continue
        rows.setdefault(cell.start_row_offset_idx, []).append(
            (cell.start_col_offset_idx, text))

    lines = []
    for row_idx in sorted(rows):
        ordered = [text for _, text in sorted(rows[row_idx])]
        deduped = []
        for text in ordered:
            if not deduped or deduped[-1] != text:
                deduped.append(text)
        if deduped:
            lines.append(' | '.join(deduped))
    return lines


def parse_document(file_path: str) -> str:
    """
    Parse a document with Docling and return plain text.

    Raises whatever Docling raises; callers decide whether to fall back.
    """
    doc = _get_converter().convert(source=file_path).document

    lines = []
    for item in doc.texts:
        text = (getattr(item, 'text', '') or '').strip()
        if text:
            lines.append(text)

    for table in doc.tables:
        lines.extend(_rows_from_table(table))

    result = '\n'.join(lines)
    logger.info("Docling parsed %s -> %d characters", os.path.basename(file_path), len(result))
    return result


def parse_upload(uploaded_file, filename: str) -> str:
    """
    Parse a Streamlit upload or file-like object.

    Docling works from a path, so the bytes are written to a temporary file with
    the original suffix (it dispatches on extension).
    """
    if hasattr(uploaded_file, 'read'):
        uploaded_file.seek(0)
        data = uploaded_file.read()
        uploaded_file.seek(0)
    else:
        with open(uploaded_file, 'rb') as fh:
            data = fh.read()

    suffix = os.path.splitext(filename)[1] or '.docx'
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        return parse_document(tmp_path)
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
