"""
Configuration for the URS Retrieval System (Unlimited-OCR branch).

Settings are read from the environment, with sensible defaults, so nothing
machine-specific or secret lives in the repository. Copy .env.example to .env
and fill it in.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
# Anchored to this file so the app behaves the same whatever the working
# directory is - previously the master database was looked up relative to the
# process, which broke as soon as it was run from anywhere but the root.
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("URS_DATA_DIR", PROJECT_ROOT / "data"))
SAMPLES_DIR = DATA_DIR / "samples"

# Authoritative requirement/response workbook (Master_DB sheet).
MASTER_DB_PATH = Path(os.getenv("MASTER_DB_PATH", DATA_DIR / "master_database.xlsm"))

# --------------------------------------------------------------------------- #
# Word document parser
#
# "docling"     - Docling's structured document model. Reads noticeably more
#                 from nested tables: 365 vs 268 requirements on
#                 Novugen_URS IGL, 331 vs 238 on the GLATT document.
# "python-docx" - the original reader. No extra dependency, ~20x faster
#                 (0.1s vs 2.8s), and the fallback if Docling errors.
#
# Comments are unaffected either way - they are read from the DOCX comment
# parts directly, which Docling does not expose.
# --------------------------------------------------------------------------- #
DOC_PARSER = os.getenv("DOC_PARSER", "docling").strip().lower()

# --------------------------------------------------------------------------- #
# Unlimited-OCR (local model, no API key needed)
# Weights: https://huggingface.co/baidu/Unlimited-OCR (~3.4 GB, BF16)
# --------------------------------------------------------------------------- #
UNLIMITED_OCR_MODEL = os.getenv("UNLIMITED_OCR_MODEL", "baidu/Unlimited-OCR")
UNLIMITED_OCR_DEVICE = os.getenv("UNLIMITED_OCR_DEVICE") or None  # "cuda" / "cpu"; auto-detect when None
UNLIMITED_OCR_DPI = int(os.getenv("UNLIMITED_OCR_DPI", "300"))    # PDF rasterisation DPI
UNLIMITED_OCR_IMAGE_MODE = os.getenv("UNLIMITED_OCR_IMAGE_MODE", "base")  # "base" or "gundam"
UNLIMITED_OCR_MAX_LENGTH = int(os.getenv("UNLIMITED_OCR_MAX_LENGTH", "32768"))
UNLIMITED_OCR_CACHE_DIR = os.getenv("UNLIMITED_OCR_CACHE_DIR") or None

# --------------------------------------------------------------------------- #
# Matching thresholds
#
# All four are now on ONE calibrated scale (see utils/similarity.py), where an
# unrelated pair of requirements scores about 0 and an exact match scores 1.
# e5-large-v2's raw cosine puts unrelated text at ~0.75 and matches at ~0.95;
# calibration rescales that back to the spread all-mpnet-base-v2 produced, so
# these numbers mean what they look like again.
#
#   calibrated   raw e5 cosine   what it is
#      0.00          <= 0.75     unrelated
#      0.15           0.79       nearest unrelated pairs
#      0.30           0.83       recall floor
#      0.70           0.93       genuine match
#      1.00           1.00       identical text
# --------------------------------------------------------------------------- #
MASTER_DB_THRESHOLD = float(os.getenv("MASTER_DB_THRESHOLD", "0.70"))
COMMENT_MATCH_THRESHOLD = float(os.getenv("COMMENT_MATCH_THRESHOLD", "0.70"))
POSTGRES_SEARCH_THRESHOLD = float(os.getenv("POSTGRES_SEARCH_THRESHOLD", "0.30"))
HISTORICAL_MATCH_THRESHOLD = float(os.getenv("HISTORICAL_MATCH_THRESHOLD", "0.70"))

# Embedding cache entries held in memory (~4 KB each).
EMBEDDING_CACHE_SIZE = int(os.getenv("EMBEDDING_CACHE_SIZE", "20000"))
