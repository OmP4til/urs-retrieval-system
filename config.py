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
# TWO DIFFERENT SCALES - do not compare these numbers directly.
#
#   Master DB and comments use raw cosine similarity from e5-large-v2.
#   PostgreSQL uses pgvector's cosine distance rescaled as (1 + cosine) / 2,
#   so 0.94 there means the same as 0.88 here. The old defaults of 0.75 and
#   0.70 looked comparable but meant cosine 0.75 versus cosine 0.40.
#
# e5-large-v2 has a high similarity floor: two unrelated URS requirements score
# around 0.80, and outright nonsense still scores 0.69. Measured against the
# 233-row master database, a 0.75 threshold matched 40 of 40 requirements -
# everything landed in the Deviation List and nothing reached Historical.
#
# Tune MASTER_DB_THRESHOLD against documents where you know the right answer.
# Reference points from that measurement: 0.85 matches 13/40, 0.88 matches
# 2/40, 0.90 matches none.
# --------------------------------------------------------------------------- #
# Raw cosine scale
MASTER_DB_THRESHOLD = float(os.getenv("MASTER_DB_THRESHOLD", "0.88"))
COMMENT_MATCH_THRESHOLD = float(os.getenv("COMMENT_MATCH_THRESHOLD", "0.75"))

# pgvector (1 + cosine) / 2 scale
POSTGRES_SEARCH_THRESHOLD = float(os.getenv("POSTGRES_SEARCH_THRESHOLD", "0.85"))   # cosine 0.70
HISTORICAL_MATCH_THRESHOLD = float(os.getenv("HISTORICAL_MATCH_THRESHOLD", "0.94"))  # cosine 0.88
