# Gemini Pro Integration

This branch adds Google Gemini Pro integration for intelligent requirement extraction and preprocessing.

## Features

🤖 **AI-Powered Extraction**: Uses Gemini Pro's advanced language understanding to identify requirements with better accuracy than rule-based methods.

📊 **Smart Categorization**: Automatically categorizes requirements into:
- Safety requirements
- Technical specifications  
- Process requirements
- Equipment specifications
- Installation requirements
- Compliance requirements

🎯 **Confidence Scoring**: Each extracted requirement includes a confidence score for quality assessment.

🔧 **Multiple Extraction Modes**:
- **Basic**: Traditional rule-based extraction
- **Enhanced**: Improved pattern matching
- **Gemini Enhanced**: AI-powered extraction with fallback to traditional methods

## Setup

1. **Install Dependencies**:
   ```bash
   pip install google-generativeai
   ```

2. **Get Gemini API Key**:
   - Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
   - Create a new API key
   - Copy the key

3. **Configure API Key**:
   - Open `config.py`
   - Replace `"your_gemini_api_key_here"` with your actual API key
   - Set `USE_GEMINI_PREPROCESSING = True`

4. **Test Integration**:
   ```bash
   python test_gemini.py
   ```

## Usage

### In Streamlit App

1. **Enable Gemini**: Check "Enable Gemini Pro Enhancement" in the sidebar
2. **Enter API Key**: Input your Gemini API key 
3. **Select Mode**: Choose "gemini_enhanced" for AI-powered extraction
4. **Set Confidence**: Adjust the confidence threshold (0.1-1.0)
5. **Upload Documents**: Upload DOCX files for processing

### Programmatic Usage

```python
from utils.gemini_processor import GeminiProcessor

# Initialize processor
processor = GeminiProcessor(api_key="your_api_key")

# Extract requirements from text
requirements = processor.extract_requirements_from_text(
    text="6.1 CIP unit shall follow cleaning phases...",
    document_type="technical_specification"
)

# Each requirement includes:
for req in requirements:
    print(f"Text: {req['text']}")
    print(f"Category: {req['category']}")
    print(f"Priority: {req['priority']}")
    print(f"Confidence: {req['confidence']}")
```

### Enhanced Extraction Pipeline

```python
from utils.extractors import extract_with_gemini_enhancement

# Process document with Gemini
enhanced_pages = extract_with_gemini_enhancement(
    uploaded_file, 
    api_key="your_api_key"
)

# Results include both traditional and Gemini-extracted requirements
for page in enhanced_pages:
    if page.get("gemini_processed"):
        print(f"Gemini found: {page['content']}")
        print(f"Category: {page['category']}")
        print(f"Confidence: {page['confidence']}")
```

## How It Works

1. **Document Processing**: Documents are first processed with traditional extraction
2. **Gemini Analysis**: Each page/section is analyzed by Gemini Pro for requirement identification
3. **Smart Categorization**: Gemini categorizes requirements and assigns confidence scores
4. **Quality Filtering**: Only requirements above the confidence threshold are included
5. **Vector Storage**: Enhanced requirements are stored in PostgreSQL with metadata

## Benefits

✅ **Higher Accuracy**: Captures complex requirements that rule-based methods miss  
✅ **Better Context**: Understands technical specifications, safety requirements, and process steps  
✅ **Smart Categorization**: Automatically organizes requirements by type and priority  
✅ **Confidence Scoring**: Enables quality-based filtering  
✅ **Fallback Support**: Falls back to traditional methods if Gemini is unavailable  

## Examples

### Input Document
```
6.1 CIP unit shall follow the 4 cleaning phases steps as follows:
- Pre-rinse phase
- Caustic wash phase  
- Intermediate rinse phase
- Sanitization phase

6.2 All electrical wiring shall be concealed and with proper earthing arrangement.
```

### Gemini Output
```json
[
  {
    "text": "6.1 CIP unit shall follow the 4 cleaning phases steps",
    "category": "process",
    "priority": "critical",
    "confidence": 0.95
  },
  {
    "text": "All electrical wiring shall be concealed and with proper earthing arrangement",
    "category": "safety", 
    "priority": "critical",
    "confidence": 0.92
  }
]
```

## Configuration Options

```python
# config.py
GEMINI_API_KEY = "your_api_key_here"
USE_GEMINI_PREPROCESSING = True
EXTRACTION_MODE = "gemini_enhanced"  # basic, enhanced, gemini_enhanced
CONFIDENCE_THRESHOLD = 0.7  # 0.1 to 1.0
```

## Cost Considerations

- Gemini Pro API has usage limits and costs
- Text is limited to 4000 characters per request for efficiency
- Batching is used to minimize API calls
- Fallback to traditional extraction if API fails or is unavailable

## Troubleshooting

**API Key Issues**:
- Verify key is correct in config.py
- Check API quota and billing in Google Cloud Console

**Import Errors**:
- Ensure `google-generativeai` is installed
- Run `pip install -r requirements.txt`

**Extraction Quality**:
- Adjust confidence threshold
- Try different extraction modes
- Check document formatting and structure

**Performance**:
- Large documents may take longer due to API calls
- Consider adjusting `GEMINI_MAX_TEXT_LENGTH` for speed vs accuracy