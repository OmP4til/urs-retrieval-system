import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import re
import requests
import pandas as pd
import streamlit as st

from utils.extractors import extract_structured_content
from utils.preprocess import rule_based_requirements
from utils.vectorstore import SimpleVectorStore
from utils.ollama_client import call_ollama_generate
from utils.json_helper import extract_and_parse_json

st.set_page_config(page_title="URS Intelligence", layout="wide")
st.title("🔍 URS Intelligence — Page-wise Matching")

# ---------------- Vector Store ----------------
EMBED_MODEL = os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2")
VECTOR_INDEX_PATH = "data/vectorstore/faiss_index.bin"
VECTOR_METADATA_PATH = "data/vectorstore/metadata.jsonl"

@st.cache_resource
def init_store():
    return SimpleVectorStore(
        model_name=EMBED_MODEL,
        index_path=VECTOR_INDEX_PATH,
        metadata_path=VECTOR_METADATA_PATH
    )

vs = init_store()

# ---------------- Sidebar Controls ----------------
st.sidebar.header("Settings")

# Defaults from env vars
default_ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
default_ollama_key = os.environ.get("OLLAMA_API_KEY", "")

# Try secrets (safe)
try:
    ollama_secrets = st.secrets.get("ollama", None)
    if ollama_secrets:
        default_ollama_url = ollama_secrets.get("url", default_ollama_url)
        default_ollama_key = ollama_secrets.get("api_key", default_ollama_key)
except Exception:
    pass  # no secrets available, stick to env/default

ollama_base_override = st.sidebar.text_input("Ollama Base URL", value=default_ollama_url)
ollama_key_override = st.sidebar.text_input("Ollama API Key (optional)", type="password", value=default_ollama_key)
ollama_model = st.sidebar.text_input("Ollama Model", value="llama3")  # default to llama3
use_ollama = st.sidebar.checkbox("Use Ollama (LLM extraction)", value=True)

# Health check
def ollama_health(base_url: str, timeout: float = 3.0) -> bool:
    try:
        url = base_url.rstrip("/") + "/api/tags"
        resp = requests.get(url, timeout=timeout)
        return resp.status_code == 200
    except Exception:
        return False

if use_ollama:
    if ollama_health(ollama_base_override):
        st.sidebar.success("Ollama reachable ✅")
    else:
        st.sidebar.error("Ollama not reachable ❌")

# ---------------- Upload + Index Historical ----------------
st.sidebar.header("Index Historical URS")
uploaded_train = st.sidebar.file_uploader(
    "Upload historical URS", accept_multiple_files=True,
    type=['pdf', 'docx', 'xlsx', 'txt']
)
index_timeout = st.sidebar.number_input("Indexing LLM timeout (sec)", 10, 600, 120)
batch_size = st.sidebar.number_input("Batch size for processing", 1, 10, 3)
min_requirement_confidence = st.sidebar.slider("Rule-based confidence", 0.0, 1.0, 0.7)

if st.sidebar.button("Index Files"):
    if not uploaded_train:
        st.sidebar.error("Upload at least one file.")
    else:
        progress_bar = st.sidebar.progress(0)
        total_files = len(uploaded_train)
        
        for file_idx, f in enumerate(uploaded_train):
            f.seek(0)
            st.sidebar.info(f"Processing {f.name}...")
            pages = extract_structured_content(f)
            
            if not pages:
                st.sidebar.warning(f"No pages in {f.name}")
                continue

            # First try rule-based extraction for all pages
            all_requirements = []
            uncertain_pages = []
            
            for page_idx, page in enumerate(pages):
                page_text = page.get("content", "")
                if not page_text.strip():
                    continue
                
                # Try rule-based first
                reqs = rule_based_requirements(page_text)
                
                # If we find good requirements, use them
                if len(reqs) > 0:
                    all_requirements.extend([{
                        "text": r,
                        "page": page_idx + 1,
                        "source": "rule-based"
                    } for r in reqs])
                else:
                    # If no requirements found, mark for LLM processing
                    uncertain_pages.append((page_idx + 1, page_text))
            
            # Only use LLM for pages where rule-based extraction found nothing
            if uncertain_pages and use_ollama:
                for batch_start in range(0, len(uncertain_pages), batch_size):
                    batch = uncertain_pages[batch_start:batch_start + batch_size]
                    batch_prompt = """Analyze each page and extract requirements. Return a JSON array of page results.
For each page, use this format:
[
    {
        "requirements": [
            {"text": "actual requirement text", "page": page_number}
        ]
    },
    // next page...
]

Remember:
1. Return ONLY the JSON array
2. Include page numbers in each requirement
3. Only extract actual requirements (statements of what the system must/should do)
4. Keep the original text of requirements

Pages to analyze:

"""
                    
                    for page_num, page_text in batch:
                        batch_prompt += f"PAGE {page_num}:\n{page_text}\n---\n"
                    
                    try:
                        raw = call_ollama_generate(
                            batch_prompt,
                            model=ollama_model,
                            max_tokens=1024,
                            timeout=index_timeout,
                            base_url=ollama_base_override.strip() or None,
                            api_key=ollama_key_override.strip() or None
                        )
                        extracted = extract_and_parse_json(raw, is_batch=True)
                        
                        if extracted and "requirements" in extracted:
                            for req in extracted["requirements"]:
                                if isinstance(req, dict):  # Ensure req is a dictionary
                                    req_text = req.get("text", "")
                                    if req_text:  # Only add if we have text
                                        all_requirements.append({
                                            "text": req_text,
                                            "page": req.get("page", batch[0][0]),  # Default to first page in batch if not specified
                                            "source": "llm"
                                        })
                    except Exception as e:
                        st.sidebar.warning(f"LLM processing failed for batch in {f.name}: {str(e)}")
                # For the first file, clear everything
                if file_idx == 0:
                    try:
                        vs.clear(keep_previous=False)
                    except Exception as e:
                        st.sidebar.warning(f"Could not clear index: {str(e)}")
                
                # Add all requirements to the vector store
                for req in all_requirements:
                    try:
                        vs.add_document(
                            req["text"],
                            {
                                "source_file": f.name,
                                "page_number": req["page"],
                                "requirement_id": f"R{req['page']}_{all_requirements.index(req)+1}",
                                "source_type": req["source"],
                                "comments": [],
                                "responses": []
                            }
                        )
                    except Exception as e:
                        st.sidebar.warning(f"Error adding requirement: {str(e)}")
                
            # Update progress
            progress = (file_idx + 1) / total_files
            progress_bar.progress(progress)
        vs.save()
        st.sidebar.success("Indexing complete ✅")

# ---------------- Analyze New URS ----------------
st.header("Analyze NEW URS")
uploaded_new = st.file_uploader("Upload new URS", type=['pdf', 'docx', 'xlsx', 'txt'])

similarity_threshold = st.slider("Similarity threshold", 0.2, 0.95, 0.7, 0.01)
max_matches = st.slider("Max matches per requirement", 1, 10, 3)
analysis_timeout = st.number_input("Analysis LLM timeout (sec)", 10, 600, 120)

def _parse_json_like(raw_text: str):
    try:
        return json.loads(raw_text.strip())
    except Exception:
        m = re.search(r"\{[\s\S]*\}", raw_text)
        return json.loads(m.group(0)) if m else None

if uploaded_new is not None:
    uploaded_new.seek(0)
    pages = extract_structured_content(uploaded_new)
    if not pages:
        st.error("Could not extract pages.")
    else:
        all_rows = []
        for page in pages:
            page_num = page.get("page_number")
            page_text = page.get("content", "")
            st.subheader(f"Page {page_num} preview")
            with st.expander(f"Page {page_num} content"):
                st.text_area(f"Page {page_num}", page_text[:2000], height=200)

            if not page_text.strip():
                continue

            prompt = f"""
Extract requirements from this PAGE. Return JSON only:
{{ "requirements": [{{"id":"R1","text":"...","short":"...","type":"functional|nonfunctional|other"}}] }}
PAGE:\n{page_text}
"""
            extracted = None
            if use_ollama:
                try:
                    raw = call_ollama_generate(
                        prompt,
                        model=ollama_model,
                        max_tokens=512,
                        timeout=analysis_timeout,
                        base_url=ollama_base_override.strip() or None,
                        api_key=ollama_key_override.strip() or None
                    )
                    extracted = _parse_json_like(raw)
                except Exception as e:
                    st.warning(f"Ollama failed page {page_num}: {e}")

            if not extracted:
                reqs = rule_based_requirements(page_text)
                extracted = {
                    "requirements": [
                        {"id": f"R{page_num}_{i+1}", "text": r, "short": r[:100], "type": "other"}
                        for i, r in enumerate(reqs)
                    ]
                }

            for req in extracted.get("requirements", []):
                q_text = req.get("text", "")
                if not q_text.strip():
                    continue

                matches = vs.search(q_text, top_k=max_matches, min_score=similarity_threshold)

                if not matches:
                    # No matches above threshold
                    row = {
                        "page_number": page_num,
                        "new_requirement_id": req.get("id", ""),
                        "new_requirement_text": q_text,
                        "found_before": False,
                        "best_match_score": 0,
                        "best_match_source_file": None,
                        "best_match_source_page": None,
                        "best_match_comments": [],
                        "best_match_responses": [],
                        "match_source": None
                    }
                    all_rows.append(row)
                else:
                    # For each match above threshold
                    for match in matches:
                        score = float(match["score"])
                        metadata = match["metadata"]
                        is_from_current = match["is_from_current"]
                        source_file = match["source_file"]
                        
                        row = {
                            "page_number": page_num,
                            "new_requirement_id": req.get("id", ""),
                            "new_requirement_text": q_text,
                            "found_before": True,
                            "best_match_score": score,
                            "best_match_source_file": source_file,
                            "best_match_source_page": metadata.get("page_number"),
                            "best_match_comments": metadata.get("comments", []),
                            "best_match_responses": metadata.get("responses", []),
                            "match_source": "Current Document" if is_from_current else "Previous Document"
                        }
                    all_rows.append(row)

        if all_rows and any(r["found_before"] for r in all_rows):
            df = pd.DataFrame(all_rows)
            st.dataframe(df)
            st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"), "urs_matches.csv")
            st.download_button("Download JSON", df.to_json(orient="records").encode("utf-8"), "urs_matches.json")
        else:
            st.info("No similar requirements found in historical data.")
