# Gemini Branch - Restructured for Gemini-Only Extraction

## Overview

The `gemini` branch has been completely restructured to use **only** Gemini 2.5 Flash for holistic requirement extraction, with a dedicated PostgreSQL database (`urs_gemini`) separate from the main branch.

## Key Changes

### 🗄️ Database Structure
- **New Database**: `urs_gemini` (separate from main branch)
- **Enhanced Schema**: Additional columns for `comments` and `matched_document_name`
- **Location**: PostgreSQL on localhost:5433
- **Table Structure**:
  ```sql
  CREATE TABLE requirements (
      id SERIAL PRIMARY KEY,
      requirement TEXT NOT NULL,
      embedding vector(384),
      metadata JSONB,
      document_name TEXT,
      comments TEXT,
      matched_document_name TEXT,
      extraction_type TEXT DEFAULT 'holistic',
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
  );
  ```

### 🧠 Extraction Method
- **Only Gemini 2.5 Flash**: No traditional or enhanced rule-based extraction
- **Holistic Analysis**: Complete document understanding without restrictions
- **Consistent Approach**: Same logic as `standalone_holistic_extraction.py`

### 📁 File Structure

#### New Files Created:
1. **`utils/postgres_vectorstore_gemini.py`** - Dedicated vector store for urs_gemini database
2. **`app/main_gemini.py`** - Streamlit app using only Gemini extraction
3. **`standalone_holistic_extraction_gemini.py`** - Command-line tool for batch processing

#### Key Features:
- **PostgresVectorStoreGemini**: Connects to urs_gemini database with enhanced schema
- **Gemini-only Interface**: Streamlit app with simplified, focused UI
- **Text-based Search**: Reliable search functionality (vector search can be improved later)

## Usage

### 1. Standalone Processing
```bash
# Basic processing
python standalone_holistic_extraction_gemini.py "document.docx"

# With comments
python standalone_holistic_extraction_gemini.py "document.docx" --comments "Initial analysis"

# With matched document reference
python standalone_holistic_extraction_gemini.py "document.docx" --matched "reference.pdf" --comments "Comparison analysis"

# Clear database first (use with caution)
python standalone_holistic_extraction_gemini.py "document.docx" --clear-first
```

### 2. Streamlit Application
```bash
# Start the Gemini-only app
streamlit run app/main_gemini.py --server.port 8502
```

**Features:**
- ✅ File upload with holistic extraction
- ✅ Real-time processing with Gemini 2.5 Flash
- ✅ Analysis summary (categories, priorities)
- ✅ Database storage with comments
- ✅ Search functionality
- ✅ Database management tools

## Database Operations

### Connection Details
```python
connection_params = {
    'host': 'localhost',
    'port': 5433,
    'database': 'urs_gemini',
    'user': 'postgres',
    'password': 'Patil1234'
}
```

### API Methods
```python
from utils.postgres_vectorstore_gemini import PostgresVectorStoreGemini

vs = PostgresVectorStoreGemini()

# Add requirements
vs.add_requirements(
    requirements=["requirement text"],
    document_name="document.pdf",
    comments="Optional comments",
    matched_document_name="reference.pdf"
)

# Search requirements
results = vs.search_similar_requirements("query", top_k=5)

# Get all requirements
all_reqs = vs.get_all_requirements()

# Database statistics
stats = vs.get_stats()

# Clear database
vs.clear_database()
```

## Testing Results

### ✅ Standalone Script Test
- **Document**: `test_urs_for_comments.docx`
- **Results**: 9 requirements extracted
- **Categories**: Functional (4), Performance (1), System Capability (2), Operational (1), Usability (1)
- **Priorities**: Critical (2), High (6), Medium (1)
- **Database**: Successfully stored in urs_gemini

### ✅ Search Functionality Test
- **Query**: "temperature"
- **Result**: Found 1 matching requirement
- **Response**: "The system shall maintain temperature between 18-24°C at all times."

### ✅ Streamlit App
- **Status**: Running on http://localhost:8502
- **Features**: All core functionality working
- **Database**: Connected and operational

## Branch Isolation

### Gemini Branch (`gemini-preprocessing`)
- Uses `urs_gemini` database
- Only Gemini 2.5 Flash extraction
- Enhanced schema with comments and matched documents
- Simplified, focused interface

### Main Branch
- Uses original database structure
- Multiple extraction methods available
- Existing functionality preserved
- No interference with Gemini branch

## Environment Requirements

```bash
# Required environment variables
GEMINI_API_KEY=your_gemini_api_key_here

# Database connection (handled automatically)
# PostgreSQL 16 on localhost:5433
```

## Next Steps

1. **Vector Search**: Improve vector similarity search (currently using text search)
2. **Batch Processing**: Add support for multiple file processing
3. **Export Features**: Add requirement export functionality
4. **Advanced Analytics**: Enhanced requirement analysis and reporting
5. **Document Comparison**: Leverage matched_document_name for comparison features

## Summary

The Gemini branch is now fully operational with:
- ✅ Dedicated `urs_gemini` PostgreSQL database
- ✅ Gemini-only extraction workflow
- ✅ Enhanced schema with comments and matched documents
- ✅ Working standalone script
- ✅ Functional Streamlit application
- ✅ Complete isolation from main branch
- ✅ Successful testing with real documents

The restructuring achieves the goal of using only Gemini extraction while maintaining a clean separation from the main branch functionality.