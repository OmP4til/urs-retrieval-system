# Configuration file for URS Retrieval System

# Load environment variables
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Unlimited-OCR Configuration (local model, no API key needed)
# Weights: https://huggingface.co/baidu/Unlimited-OCR (~3.4 GB, BF16)
UNLIMITED_OCR_MODEL = os.getenv("UNLIMITED_OCR_MODEL", "baidu/Unlimited-OCR")
UNLIMITED_OCR_DEVICE = os.getenv("UNLIMITED_OCR_DEVICE") or None  # "cuda" / "cpu"; auto-detect when None
UNLIMITED_OCR_DPI = int(os.getenv("UNLIMITED_OCR_DPI", "300"))    # PDF rasterisation DPI
UNLIMITED_OCR_IMAGE_MODE = os.getenv("UNLIMITED_OCR_IMAGE_MODE", "base")  # "base" or "gundam"
UNLIMITED_OCR_MAX_LENGTH = int(os.getenv("UNLIMITED_OCR_MAX_LENGTH", "32768"))
UNLIMITED_OCR_CACHE_DIR = os.getenv("UNLIMITED_OCR_CACHE_DIR") or None

# Database Configuration
POSTGRES_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "urs_db",
    "user": "postgres",
    "password": "password"
}

# Extraction Configuration
EXTRACTION_MODE = "ocr_enhanced"  # This branch runs Unlimited-OCR only
CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence for requirements