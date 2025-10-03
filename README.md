# URS Intelligence System (Ready)

## Quick Start

1. Create & activate virtual env:
   python -m venv venv
   venv\Scripts\activate   # Windows
   source venv/bin/activate  # Linux/mac

2. Install Python deps:
   pip install -r requirements.txt

3. Install system deps:
   - Tesseract OCR (add to PATH)
   - Poppler (for pdf2image)

4. (Optional) Run Ollama locally for LLM extractions:
   ollama run llama2

5. Start app:
   streamlit run app/main.py

Place historical files via sidebar uploader to index them. New URS via main uploader to analyze.
