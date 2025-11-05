# Configuration file for URS Retrieval System

# Gemini Pro API Configuration
GEMINI_API_KEY = "your_gemini_api_key_here"  # Replace with your actual API key
GEMINI_MODEL = "gemini-pro"

# Processing Configuration
USE_GEMINI_PREPROCESSING = True  # Set to True to enable Gemini preprocessing
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
EXTRACTION_MODE = "gemini_enhanced"  # Options: "basic", "enhanced", "gemini_enhanced"
CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence for requirements