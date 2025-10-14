
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
st.title("🔍 URS Retrival")

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
        
        # Show auto-indexing info
        st.divider()
        st.markdown("**🔄 Auto-Indexing**")
        st.caption("New analysis files are automatically indexed if not already present")
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
                page_section = page.get("section", "Unknown Section")
                page_req_id = page.get("requirement_id", f"R{page_idx+1}")
                
                if not page_text.strip():
                    continue
                
                # Try rule-based first
                reqs = rule_based_requirements(page_text)
                
                if len(reqs) > 0:
                    all_requirements.extend([{
                        "text": r,
                        "page": page_idx + 1,
                        "source": "rule-based",
                        "comments": page_comments,
                        "section": page_section,
                        "requirement_id": page_req_id
                    } for r in reqs])
                else:
                    # Only add to uncertain if page has substantial content
                    if len(page_text.strip()) > 100:
                        uncertain_pages.append((page_idx + 1, page_text, page_comments, page_section, page_req_id))
            
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
                    for page_data in batch:
                        page_num, page_text = page_data[0], page_data[1]
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
                                        # Find the corresponding page data for section info
                                        req_page = req.get("page", batch[0][0])
                                        page_data = next((pd for pd in batch if pd[0] == req_page), batch[0])
                                        page_comments = page_data[2] if len(page_data) > 2 else []
                                        page_section = page_data[3] if len(page_data) > 3 else "Unknown Section"
                                        page_req_id = page_data[4] if len(page_data) > 4 else f"R{req_page}"
                                        
                                        all_requirements.append({
                                            "text": req_text,
                                            "page": req_page,
                                            "source": "llm",
                                            "comments": page_comments,
                                            "section": page_section,
                                            "requirement_id": page_req_id
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
                                "requirement_id": req.get("requirement_id", f"R{req['page']}_{req_idx+1}"),
                                "source_type": req["source"],
                                "section": req.get("section", "Unknown Section"),
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

# Initialize force_reindex variable
force_reindex = False

# Auto-indexing options
if uploaded_new is not None:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.info("🔄 Auto-indexing enabled: New files are automatically stored in database")
    with col2:
        force_reindex = st.checkbox("Force re-index", help="Re-index even if file already exists")

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
    
    # Check if file already exists in database
    filename = uploaded_new.name
    existing_docs = vs.get_all_documents()
    file_exists = any(doc['filename'] == filename for doc in existing_docs)
    
    # Determine if we should index
    should_index = not file_exists or force_reindex
    
    if should_index:
        if file_exists and force_reindex:
            st.warning(f"🔄 Force re-indexing '{filename}' (will replace existing entries)...")
            # Remove existing entries for this file
            try:
                # Note: You might need to implement a method to remove documents by filename
                # For now, we'll just add new entries which will supplement existing ones
                st.info("Adding new version alongside existing entries...")
            except Exception as e:
                st.warning(f"Could not remove existing entries: {e}")
        else:
            st.info(f"🔄 Auto-indexing '{filename}' as it's not in the database yet...")
        
        # Auto-index the new file using the same logic as historical indexing
        with st.spinner(f"📥 Indexing {filename}..."):
            uploaded_new.seek(0)  # Reset file position
            pages_for_indexing = extract_structured_content(uploaded_new)
            
            if pages_for_indexing:
                # Extract requirements using the same approach
                all_requirements = []
                uncertain_pages = []
                
                # Rule-based extraction first
                for page_idx, page in enumerate(pages_for_indexing):
                    page_text = page.get("content", "")
                    page_comments = page.get("comments", []) or []
                    page_section = page.get("section", "Unknown Section")
                    page_req_id = page.get("requirement_id", f"R{page_idx+1}")
                    
                    if not page_text.strip():
                        continue
                    
                    # Try rule-based first
                    reqs = rule_based_requirements(page_text)
                    
                    if len(reqs) > 0:
                        all_requirements.extend([{
                            "text": r,
                            "page": page_idx + 1,
                            "source": "rule-based",
                            "comments": page_comments,
                            "section": page_section,
                            "requirement_id": page_req_id
                        } for r in reqs])
                    else:
                        # Only add to uncertain if page has substantial content
                        if len(page_text.strip()) > 100:
                            uncertain_pages.append((page_idx + 1, page_text, page_comments, page_section, page_req_id))
                
                # Use LLM for uncertain pages if enabled
                if uncertain_pages and use_ollama:
                    for batch_start in range(0, len(uncertain_pages), batch_size):
                        batch = uncertain_pages[batch_start:batch_start + batch_size]
                        
                        batch_prompt = f"""Extract requirements from the following {len(batch)} page(s). 
Return JSON array format: [{{"requirements": [{{"text": "requirement", "page": 1}}]}}]

Only extract actual requirements (must/shall/should statements).

"""
                        
                        for page_data in batch:
                            page_num, page_text = page_data[0], page_data[1]
                            text_preview = page_text[:2000] if len(page_text) > 2000 else page_text
                            batch_prompt += f"PAGE {page_num}:\n{text_preview}\n---\n"
                        
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
                                    if isinstance(req, dict):
                                        req_text = req.get("text", "")
                                        if req_text:
                                            req_page = req.get("page", batch[0][0])
                                            page_data = next((pd for pd in batch if pd[0] == req_page), batch[0])
                                            page_comments = page_data[2] if len(page_data) > 2 else []
                                            page_section = page_data[3] if len(page_data) > 3 else "Unknown Section"
                                            page_req_id = page_data[4] if len(page_data) > 4 else f"R{req_page}"
                                            
                                            all_requirements.append({
                                                "text": req_text,
                                                "page": req_page,
                                                "source": "llm",
                                                "comments": page_comments,
                                                "section": page_section,
                                                "requirement_id": page_req_id
                                            })
                        except Exception as e:
                            st.warning(f"⚠️ LLM extraction failed for batch: {str(e)[:100]}")
                
                # Add all requirements to database
                if all_requirements:
                    added_count = 0
                    for req_idx, req in enumerate(all_requirements):
                        try:
                            vs.add_document(
                                req["text"],
                                {
                                    "source_file": filename,
                                    "page_number": req["page"],
                                    "requirement_id": req.get("requirement_id", f"R{req['page']}_{req_idx+1}"),
                                    "source_type": req["source"],
                                    "section": req.get("section", "Unknown Section"),
                                    "comments": req.get("comments", []),
                                    "responses": []
                                }
                            )
                            added_count += 1
                        except Exception as e:
                            st.warning(f"Error adding requirement: {str(e)[:100]}")
                    
                    vs.save()
                    if force_reindex and file_exists:
                        st.success(f"✅ Re-indexed {filename} with {added_count} requirements!")
                    else:
                        st.success(f"✅ Auto-indexed {filename} with {added_count} requirements!")
                else:
                    st.warning(f"⚠️ No requirements found in {filename} for indexing")
            else:
                st.warning(f"⚠️ Could not extract content from {filename}")
    else:
        st.info(f"📋 '{filename}' is already indexed in the database (use 'Force re-index' to update)")
    
    # Reset file position for analysis
    uploaded_new.seek(0)
    pages = extract_structured_content(uploaded_new)
    if not pages:
        st.error("Could not extract pages.")
    else:
        # Group pages by section for organized display
        sections = {}
        for page in pages:
            section = page.get("section", "Unknown Section")
            if section not in sections:
                sections[section] = []
            sections[section].append(page)
        
        st.info(f"📋 Document organized into {len(sections)} sections with {len(pages)} total requirements")
        
        # Display section summary
        with st.expander("📑 Document Structure Overview"):
            for section_name, section_pages in sections.items():
                comment_count = sum(len(page.get("comments", [])) for page in section_pages)
                st.write(f"**{section_name}**: {len(section_pages)} requirements ({comment_count} comments)")
        
        all_rows = []
        
        # Add overall progress
        progress_placeholder = st.empty()
        
        # Process pages by section for better organization
        for section_name, section_pages in sections.items():
            st.header(f"📂 {section_name}")
            
            for page_idx, page in enumerate(section_pages):
                page_num = page.get("page_number")
                page_text = page.get("content", "")
                page_comments = page.get("comments", [])
                req_id = page.get("requirement_id", f"REQ-{page_num}")
                
                progress_placeholder.text(f"Processing {section_name}: {page_idx + 1}/{len(section_pages)}...")
                
                # Enhanced page display with section context
                with st.expander(f"🔍 {req_id} - {section_name}"):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.text_area(f"Requirement Content", page_text[:2000], height=150, key=f"page_{page_num}")
                    
                    with col2:
                        st.write(f"**Section**: {section_name}")
                        st.write(f"**Requirement ID**: {req_id}")
                        st.write(f"**Page**: {page_num}")
                        
                        # Display comments if any
                        if page_comments:
                            st.write(f"**Comments** ({len(page_comments)}):")
                            for i, comment in enumerate(page_comments[:3]):  # Show first 3 comments
                                if isinstance(comment, dict):
                                    author = comment.get("author", "Unknown")
                                    comment_text = comment.get("text", "")[:100]
                                    st.write(f"• {author}: {comment_text}...")
                            if len(page_comments) > 3:
                                st.write(f"... and {len(page_comments) - 3} more comments")
                        else:
                            st.write("**Comments**: None")

                if not page_text.strip():
                    continue

                # Shorter prompt
                prompt = f"""Extract requirements from this page. Return JSON only:
{{"requirements": [{{"id":"R1","text":"requirement text","type":"functional"}}]}}

PAGE:\n{page_text[:3000]}"""
                
                extracted = None
                if use_ollama:
                    try:
                        with st.spinner(f"🤖 Analyzing {req_id} with LLM..."):
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
                        st.warning(f"⚠️ Ollama failed {req_id}: {str(e)[:100]}")

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
                            "section": section_name,
                            "page_number": page_num,
                            "requirement_id": req_id,
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
                            
                            # Format comments for display and extract authors
                            comments = metadata.get("comments", [])
                            formatted_comments = ""
                            comment_authors = ""
                            
                            if comments:
                                # Create a readable string from comments and collect authors
                                comment_strings = []
                                author_list = []
                                
                                for comment in comments[:3]:  # Show max 3 comments
                                    if isinstance(comment, dict):
                                        author = comment.get("author", "Unknown")
                                        text = comment.get("text", "")[:100]  # Truncate long comments
                                        comment_strings.append(f"{text}...")
                                        if author not in author_list:
                                            author_list.append(author)
                                    else:
                                        comment_strings.append(str(comment)[:100])
                                        author_list.append("Unknown")
                                
                                formatted_comments = "; ".join(comment_strings)
                                comment_authors = "; ".join(author_list)
                                
                                if len(comments) > 3:
                                    formatted_comments += f" (+{len(comments)-3} more)"
                            else:
                                formatted_comments = "No comments"
                                comment_authors = "N/A"
                            
                            row = {
                                "section": section_name,
                                "page_number": page_num,
                                "requirement_id": req_id,
                                "new_requirement_id": req.get("id", ""),
                                "new_requirement_text": q_text,
                                "found_before": True,
                                "best_match_score": score,
                                "best_match_source_file": source_file,
                                "best_match_source_page": metadata.get("page_number"),
                                "best_match_comments": formatted_comments,
                                "comment_authors": comment_authors,
                                "best_match_responses": metadata.get("responses", []),
                                "match_source": "Current Document" if is_from_current else "Previous Document",
                                "matched_text": match.get("text", "")[:200]
                            }
                            all_rows.append(row)
        
        progress_placeholder.empty()

        if all_rows:
            df = pd.DataFrame(all_rows)
            
            # Display enhanced summary with section breakdown
            total_reqs = len(df)
            found_reqs = df['found_before'].sum()
            
            st.success(f"📊 Analysis Complete: {found_reqs} out of {total_reqs} requirements found in historical data")
            
            # Section-based summary
            section_summary = df.groupby('section').agg({
                'found_before': ['count', 'sum'],
                'best_match_score': 'mean'
            }).round(3)
            
            st.subheader("📈 Section-wise Analysis Summary")
            
            # Create a cleaner summary dataframe
            summary_data = []
            for section in section_summary.index:
                total_in_section = section_summary.loc[section, ('found_before', 'count')]
                found_in_section = section_summary.loc[section, ('found_before', 'sum')]
                avg_score = section_summary.loc[section, ('best_match_score', 'mean')]
                
                summary_data.append({
                    'Section': section,
                    'Total Requirements': total_in_section,
                    'Found in Historical': found_in_section,
                    'Match Rate': f"{(found_in_section/total_in_section*100):.1f}%" if total_in_section > 0 else "0%",
                    'Avg Similarity Score': f"{avg_score:.3f}" if found_in_section > 0 else "N/A"
                })
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)
            
            # Section filter for detailed view
            st.subheader("🔍 Detailed Requirements Analysis")
            
            # Add section filter
            selected_sections = st.multiselect(
                "Filter by sections:",
                options=list(df['section'].unique()),
                default=list(df['section'].unique()),
                help="Select which sections to display in the detailed analysis"
            )
            
            # Filter dataframe by selected sections
            if selected_sections:
                filtered_df = df[df['section'].isin(selected_sections)]
                
                # Display filtered results
                st.write(f"Showing {len(filtered_df)} requirements from {len(selected_sections)} sections")
                
                # Reorganize columns for better display
                display_columns = [
                    'section', 'requirement_id', 'new_requirement_text', 'found_before',
                    'best_match_score', 'matched_text', 'best_match_source_file', 
                    'comment_authors', 'best_match_comments'
                ]
                
                # Only show existing columns
                available_columns = [col for col in display_columns if col in filtered_df.columns]
                
                # Rename columns for better display
                column_renames = {
                    'section': 'Section',
                    'requirement_id': 'Requirement ID',
                    'new_requirement_text': 'Requirement Text',
                    'found_before': 'Found Before',
                    'best_match_score': 'Match Score',
                    'matched_text': 'Matched Text',
                    'best_match_source_file': 'Source File',
                    'comment_authors': 'Comment Authors',
                    'best_match_comments': 'Comments'
                }
                
                display_df = filtered_df[available_columns].copy()
                display_df = display_df.rename(columns=column_renames)
                
                st.dataframe(display_df, use_container_width=True)
                
                # Download buttons for filtered data
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        "📥 Download Filtered CSV", 
                        filtered_df.to_csv(index=False).encode("utf-8"), 
                        f"urs_matches_filtered.csv",
                        mime="text/csv"
                    )
                with col2:
                    st.download_button(
                        "📥 Download Filtered JSON", 
                        filtered_df.to_json(orient="records", indent=2).encode("utf-8"), 
                        f"urs_matches_filtered.json",
                        mime="application/json"
                    )
            else:
                st.warning("Please select at least one section to display results.")
                
            # Full dataset download (always available)
            with st.expander("📁 Download Complete Dataset"):
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        "📥 Download Complete CSV", 
                        df.to_csv(index=False).encode("utf-8"), 
                        "urs_matches_complete.csv",
                        mime="text/csv"
                    )
                with col2:
                    st.download_button(
                        "📥 Download Complete JSON", 
                        df.to_json(orient="records", indent=2).encode("utf-8"), 
                        "urs_matches_complete.json",
                        mime="application/json"
                    )
        else:
            st.info("No requirements found to analyze.")