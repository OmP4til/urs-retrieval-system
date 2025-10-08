
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import re
import requests
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import time

from utils.extractors import extract_structured_content
from utils.preprocess import rule_based_requirements
from utils.postgres_vectorstore import PostgresVectorStore
from utils.ollama_client import call_ollama_generate
from utils.json_helper import extract_and_parse_json

# Load environment variables
load_dotenv()

st.set_page_config(page_title="URS Intelligence", layout="wide")
st.title("🔍 URS Intelligence — PostgreSQL Edition")

# ---------------- PostgreSQL Vector Store ----------------
EMBED_MODEL = os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2")
DATABASE_URL = os.environ.get("DATABASE_URL")

if not DATABASE_URL:
    st.error("❌ DATABASE_URL not found in environment variables. Please check your .env file.")
    st.stop()

@st.cache_resource
def init_store():
    try:
        return PostgresVectorStore(
            model_name=EMBED_MODEL,
            db_url=DATABASE_URL
        )
    except Exception as e:
        st.error(f"❌ Failed to initialize PostgreSQL connection: {e}")
        st.stop()

vs = init_store()

# Display database status
with st.sidebar:
    st.success("✅ PostgreSQL Connected")
    try:
        docs = vs.get_all_documents()
        st.info(f"📚 Documents in DB: {len(docs)}")
        if docs:
            with st.expander("View indexed documents"):
                for doc in docs:
                    st.write(f"- {doc['filename']} ({doc['requirement_count']} requirements)")
    except Exception as e:
        st.warning(f"Could not fetch documents: {e}")

# ---------------- Sidebar Controls ----------------
st.sidebar.header("Settings")

# Ollama settings
default_ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
default_ollama_key = os.environ.get("OLLAMA_API_KEY", "")

try:
    ollama_secrets = st.secrets.get("ollama", None)
    if ollama_secrets:
        default_ollama_url = ollama_secrets.get("url", default_ollama_url)
        default_ollama_key = ollama_secrets.get("api_key", default_ollama_key)
except Exception:
    pass

ollama_base_override = st.sidebar.text_input("Ollama Base URL", value=default_ollama_url)
ollama_key_override = st.sidebar.text_input("Ollama API Key (optional)", type="password", value=default_ollama_key)
ollama_model = st.sidebar.text_input("Ollama Model", value="llama3")
use_ollama = st.sidebar.checkbox("Use Ollama (LLM extraction)", value=False)  # Default to False

# Health check with better feedback
def ollama_health(base_url: str, timeout: float = 3.0) -> tuple[bool, str]:
    try:
        url = base_url.rstrip("/") + "/api/tags"
        resp = requests.get(url, timeout=timeout)
        if resp.status_code == 200:
            models = resp.json().get('models', [])
            return True, f"Found {len(models)} models"
        return False, f"Status code: {resp.status_code}"
    except requests.exceptions.ConnectionError:
        return False, "Cannot connect - Is Ollama running?"
    except requests.exceptions.Timeout:
        return False, "Timeout - Ollama not responding"
    except Exception as e:
        return False, str(e)

if use_ollama:
    healthy, message = ollama_health(ollama_base_override)
    if healthy:
        st.sidebar.success(f"Ollama reachable ✅\n{message}")
    else:
        st.sidebar.error(f"Ollama not reachable ❌\n{message}")
        st.sidebar.warning("⚠️ LLM extraction will fail. Use rule-based only or fix Ollama.")

# ---------------- Upload + Index Historical ----------------
st.sidebar.header("Index Historical URS")
uploaded_train = st.sidebar.file_uploader(
    "Upload historical URS", accept_multiple_files=True,
    type=['pdf', 'docx', 'xlsx', 'txt']
)
index_timeout = st.sidebar.number_input("Indexing LLM timeout (sec)", 10, 600, 120)
batch_size = st.sidebar.number_input("Batch size for processing", 1, 5, 1)  # Reduced default

# Add option to clear database
if st.sidebar.button("🗑️ Clear All Data", type="secondary"):
    try:
        vs.clear(keep_previous=False)
        st.sidebar.success("✅ Database cleared!")
        st.rerun()
    except Exception as e:
        st.sidebar.error(f"❌ Error clearing database: {e}")

if st.sidebar.button("📥 Index Files"):
    if not uploaded_train:
        st.sidebar.error("Upload at least one file.")
    else:
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()
        total_files = len(uploaded_train)
        
        total_requirements_added = 0
        
        for file_idx, f in enumerate(uploaded_train):
            f.seek(0)
            status_text.text(f"📄 Processing {f.name}...")
            
            pages = extract_structured_content(f)
            
            if not pages:
                st.sidebar.warning(f"No pages in {f.name}")
                continue

            # First try rule-based extraction for all pages
            all_requirements = []
            uncertain_pages = []
            
            status_text.text(f"📄 {f.name}: Rule-based extraction...")
            for page_idx, page in enumerate(pages):
                page_text = page.get("content", "")
                page_comments = page.get("comments", []) or []
                if not page_text.strip():
                    continue
                
                # Try rule-based first
                reqs = rule_based_requirements(page_text)
                
                if len(reqs) > 0:
                    all_requirements.extend([{
                        "text": r,
                        "page": page_idx + 1,
                        "source": "rule-based",
                        "comments": page_comments
                    } for r in reqs])
                else:
                    # Only add to uncertain if page has substantial content
                    if len(page_text.strip()) > 100:
                        uncertain_pages.append((page_idx + 1, page_text))
            
            status_text.text(f"📄 {f.name}: Found {len(all_requirements)} via rules, {len(uncertain_pages)} pages for LLM")
            
            # Only use LLM for pages where rule-based extraction found nothing
            if uncertain_pages and use_ollama:
                status_text.text(f"🤖 {f.name}: Processing {len(uncertain_pages)} pages with LLM...")
                
                for batch_start in range(0, len(uncertain_pages), batch_size):
                    batch = uncertain_pages[batch_start:batch_start + batch_size]
                    batch_num = (batch_start // batch_size) + 1
                    total_batches = (len(uncertain_pages) + batch_size - 1) // batch_size
                    
                    status_text.text(f"🤖 {f.name}: LLM batch {batch_num}/{total_batches}...")
                    
                    # Shorter, more focused prompt
                    batch_prompt = f"""Extract requirements from the following {len(batch)} page(s). 
Return JSON array format: [{{"requirements": [{{"text": "requirement", "page": 1}}]}}]

Only extract actual requirements (must/shall/should statements).

"""
                    
                    # Limit text length per page
                    for page_num, page_text in batch:
                        # Truncate very long pages
                        text_preview = page_text[:2000] if len(page_text) > 2000 else page_text
                        batch_prompt += f"PAGE {page_num}:\n{text_preview}\n---\n"
                    
                    try:
                        start_time = time.time()
                        raw = call_ollama_generate(
                            batch_prompt,
                            model=ollama_model,
                            max_tokens=1024,
                            timeout=index_timeout,
                            base_url=ollama_base_override.strip() or None,
                            api_key=ollama_key_override.strip() or None
                        )
                        elapsed = time.time() - start_time
                        
                        status_text.text(f"🤖 {f.name}: Batch {batch_num} completed in {elapsed:.1f}s")
                        
                        extracted = extract_and_parse_json(raw, is_batch=True)
                        
                        if extracted and "requirements" in extracted:
                            for req in extracted["requirements"]:
                                if isinstance(req, dict):
                                    req_text = req.get("text", "")
                                    if req_text:
                                        all_requirements.append({
                                            "text": req_text,
                                            "page": req.get("page", batch[0][0]),
                                            "source": "llm",
                                            "comments": page_comments
                                        })
                    except Exception as e:
                        st.sidebar.warning(f"⚠️ LLM batch {batch_num} failed: {str(e)[:100]}")
                        # Continue with next batch
            
            # Add all requirements to PostgreSQL
            if all_requirements:
                status_text.text(f"💾 {f.name}: Saving {len(all_requirements)} requirements...")
                for req_idx, req in enumerate(all_requirements):
                    try:
                        vs.add_document(
                            req["text"],
                            {
                                "source_file": f.name,
                                "page_number": req["page"],
                                "requirement_id": f"R{req['page']}_{req_idx+1}",
                                "source_type": req["source"],
                                "comments": req.get("comments", []),
                                "responses": []
                            }
                        )
                        total_requirements_added += 1
                    except Exception as e:
                        st.sidebar.warning(f"Error adding requirement: {str(e)[:100]}")
                
                vs.save()
            
            # Update progress
            progress = (file_idx + 1) / total_files
            progress_bar.progress(progress)
        
        status_text.text(f"✅ Indexed {total_files} files, {total_requirements_added} requirements!")
        st.sidebar.success(f"✅ Complete! Added {total_requirements_added} requirements")
        time.sleep(2)
        st.rerun()
        
# ---------------- Analyze New URS ----------------
st.header("Analyze NEW URS")
uploaded_new = st.file_uploader("Upload new URS", type=['pdf', 'docx', 'xlsx', 'txt'])

similarity_threshold = st.slider("Similarity threshold", 0.1, 0.95, 0.4, 0.01)
max_matches = st.slider("Max matches per requirement", 1, 10, 3)
analysis_timeout = st.number_input("Analysis LLM timeout (sec)", 10, 600, 60)  # Reduced default

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
        
        # Add overall progress
        progress_placeholder = st.empty()
        
        for page_idx, page in enumerate(pages):
            page_num = page.get("page_number")
            page_text = page.get("content", "")
            
            progress_placeholder.text(f"Processing page {page_idx + 1}/{len(pages)}...")
            
            st.subheader(f"Page {page_num} preview")
            with st.expander(f"Page {page_num} content"):
                st.text_area(f"Page {page_num}", page_text[:2000], height=200, key=f"page_{page_num}")

            if not page_text.strip():
                continue

            # Shorter prompt
            prompt = f"""Extract requirements from this page. Return JSON only:
{{"requirements": [{{"id":"R1","text":"requirement text","type":"functional"}}]}}

PAGE:\n{page_text[:3000]}"""
            
            extracted = None
            if use_ollama:
                try:
                    with st.spinner(f"🤖 Analyzing page {page_num} with LLM..."):
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
                    st.warning(f"⚠️ Ollama failed page {page_num}: {str(e)[:100]}")

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
                    row = {
                        "page_number": page_num,
                        "new_requirement_id": req.get("id", ""),
                        "new_requirement_text": q_text,
                        "found_before": False,
                        "best_match_score": 0,
                        "best_match_source_file": None,
                        "best_match_source_page": None,
                        "best_match_comments": "No comments",
                        "best_match_responses": [],
                        "match_source": None
                    }
                    all_rows.append(row)
                else:
                    for match in matches:
                        score = float(match["score"])
                        metadata = match["metadata"]
                        is_from_current = match.get("is_from_current", False)
                        source_file = match["source_file"]
                        
                        # Format comments for display
                        comments = metadata.get("comments", [])
                        if comments:
                            # Create a readable string from comments
                            comment_strings = []
                            for comment in comments[:3]:  # Show max 3 comments
                                if isinstance(comment, dict):
                                    author = comment.get("author", "Unknown")
                                    text = comment.get("text", "")[:100]  # Truncate long comments
                                    comment_strings.append(f"{author}: {text}...")
                                else:
                                    comment_strings.append(str(comment)[:100])
                            
                            formatted_comments = "; ".join(comment_strings)
                            if len(comments) > 3:
                                formatted_comments += f" (+{len(comments)-3} more)"
                        else:
                            formatted_comments = "No comments"
                        
                        row = {
                            "page_number": page_num,
                            "new_requirement_id": req.get("id", ""),
                            "new_requirement_text": q_text,
                            "found_before": True,
                            "best_match_score": score,
                            "best_match_source_file": source_file,
                            "best_match_source_page": metadata.get("page_number"),
                            "best_match_comments": formatted_comments,
                            "best_match_responses": metadata.get("responses", []),
                            "match_source": "Current Document" if is_from_current else "Previous Document",
                            "matched_text": match.get("text", "")[:200]
                        }
                        all_rows.append(row)
        
        progress_placeholder.empty()

        if all_rows:
            df = pd.DataFrame(all_rows)
            
            # Display summary
            total_reqs = len(df)
            found_reqs = df['found_before'].sum()
            st.success(f"📊 Found {found_reqs} out of {total_reqs} requirements in historical data")
            
            # Display dataframe
            st.dataframe(df, width='stretch')
            
            # Download buttons
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "📥 Download CSV", 
                    df.to_csv(index=False).encode("utf-8"), 
                    "urs_matches.csv",
                    mime="text/csv"
                )
            with col2:
                st.download_button(
                    "📥 Download JSON", 
                    df.to_json(orient="records", indent=2).encode("utf-8"), 
                    "urs_matches.json",
                    mime="application/json"
                )
        else:
            st.info("No requirements found to analyze.")