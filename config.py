# Configuration file for URS Retrieval System

# Load environment variables
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Gemini Pro API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # Set this as an environment variable
GEMINI_MODEL = "models/gemini-3.6-flash" #Experimental model

# Unlimited-OCR Configuration (local model, no API key needed)
# Weights: https://huggingface.co/baidu/Unlimited-OCR (~3.4 GB, BF16)
UNLIMITED_OCR_MODEL = os.getenv("UNLIMITED_OCR_MODEL", "baidu/Unlimited-OCR")
UNLIMITED_OCR_DEVICE = os.getenv("UNLIMITED_OCR_DEVICE") or None  # "cuda" / "cpu"; auto-detect when None
UNLIMITED_OCR_DPI = int(os.getenv("UNLIMITED_OCR_DPI", "300"))    # PDF rasterisation DPI
UNLIMITED_OCR_IMAGE_MODE = os.getenv("UNLIMITED_OCR_IMAGE_MODE", "base")  # "base" or "gundam"
UNLIMITED_OCR_MAX_LENGTH = int(os.getenv("UNLIMITED_OCR_MAX_LENGTH", "32768"))
UNLIMITED_OCR_CACHE_DIR = os.getenv("UNLIMITED_OCR_CACHE_DIR") or None

# Which backend performs extraction: "unlimited_ocr" (local) or "gemini" (API)
EXTRACTION_BACKEND = os.getenv("EXTRACTION_BACKEND", "unlimited_ocr")

# Processing Configuration
USE_GEMINI_PREPROCESSING = EXTRACTION_BACKEND == "gemini"
GEMINI_BATCH_SIZE = 10  # Number of requirements to process in each batch
GEMINI_MAX_TEXT_LENGTH = 4000  # Maximum text length to send to Gemini

# Database Configuration
POSTGRES_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "urs_db",
    "user": "postgres",
    "password": "password"
}

# Extraction Configuration
EXTRACTION_MODE = "ocr_enhanced"  # Options: "basic", "enhanced", "gemini_enhanced", "ocr_enhanced"
CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence for requirements