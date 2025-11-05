# Configuration file for URS Retrieval System

# Load environment variables
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Gemini Pro API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # Set this as an environment variable
GEMINI_MODEL = "models/gemini-2.5-flash"  # Updated to use the correct model name

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