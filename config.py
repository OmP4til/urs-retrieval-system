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
# Documented in docs/UNLIMITED_OCR_MATCHING.md. Kept here for reference; the
# values are currently applied at their call sites.
# --------------------------------------------------------------------------- #
MASTER_DB_THRESHOLD = 0.75        # Excel master database - strict, authoritative
POSTGRES_SEARCH_THRESHOLD = 0.30  # pgvector recall floor
HISTORICAL_MATCH_THRESHOLD = 0.70 # promotes a match into the Historical table
COMMENT_MATCH_THRESHOLD = 0.75    # DOCX comment to requirement pairing
