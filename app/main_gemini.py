"""
Streamlit App for Gemini Branch - Only Gemini Holistic Extraction
Uses dedicated urs_gemini PostgreSQL database
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from dotenv import load_dotenv
import time
from typing import List, Dict, Any

# Import Gemini-specific components
from utils.postgres_vectorstore_gemini import PostgresVectorStoreGemini
from utils.extractors import extract_text_from_docx, extract_text_from_file

# Import the extraction backend selected by config.EXTRACTION_BACKEND
# ("unlimited_ocr" = local Baidu Unlimited-OCR model, "gemini" = Gemini API)
try:
    from utils.processor_factory import get_processor
    from config import EXTRACTION_BACKEND
    GEMINI_PROCESSOR_AVAILABLE = True
except ImportError:
    GEMINI_PROCESSOR_AVAILABLE = False
    EXTRACTION_BACKEND = "gemini"
    st.error("❌ Extraction processor not available. Please ensure utils/processor_factory.py exists.")

USING_GEMINI = EXTRACTION_BACKEND == "gemini"
BACKEND_LABEL = "Gemini" if USING_GEMINI else "Unlimited-OCR"

# Import Master Database Layer
try:
    from utils.master_database import MasterDatabase
    MASTER_DB_AVAILABLE = True
except ImportError:
    MASTER_DB_AVAILABLE = False
    st.warning("⚠️ Master Database module not available. Will skip master database check.")

# Load environment variables from the parent directory
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

def extract_clean_comments(comments_data) -> List[Dict[str, str]]:
    """
    Extract clean, readable comments from various comment data formats.
    Returns a list of comment dictionaries with 'text' and 'author' keys.
    """
    clean_comments = []
    
    if not comments_data:
        return clean_comments
        
    try:
        import json
        import re
        
        # Handle string JSON format
        if isinstance(comments_data, str):
            try:
                parsed_comments = json.loads(comments_data)
            except json.JSONDecodeError:
                # Try to extract comment_text using regex for malformed JSON
                comment_matches = re.findall(r'"comment_text"\s*:\s*"([^"]*)"', comments_data)
                author_matches = re.findall(r'"author"\s*:\s*"([^"]*)"', comments_data)
                
                for i, comment_text in enumerate(comment_matches):
                    author = author_matches[i] if i < len(author_matches) else "Unknown"
                    clean_comments.append({
                        'text': comment_text,
                        'author': author
                    })
                return clean_comments
        else:
            parsed_comments = comments_data
            
        # Handle structured dictionary format
        if isinstance(parsed_comments, dict):
            if 'comments' in parsed_comments:
                # New structured format: {"count": N, "comments": [...]}
                for comment in parsed_comments.get('comments', []):
                    if isinstance(comment, dict):
                        clean_comments.append({
                            'text': comment.get('comment_text', comment.get('text', 'No comment text')),
                            'author': comment.get('author', 'Unknown'),
                            'type': comment.get('comment_type', 'response')
                        })
            else:
                # Single comment dictionary
                clean_comments.append({
                    'text': parsed_comments.get('comment_text', parsed_comments.get('text', 'No comment text')),
                    'author': parsed_comments.get('author', 'Unknown')
                })
        
        # Handle list format
        elif isinstance(parsed_comments, list):
            for comment in parsed_comments:
                if isinstance(comment, dict):
                    clean_comments.append({
                        'text': comment.get('comment_text', comment.get('text', 'No comment text')),
                        'author': comment.get('author', 'Unknown'),
                        'type': comment.get('comment_type', 'response')
                    })
                elif isinstance(comment, str):
                    clean_comments.append({
                        'text': comment,
                        'author': 'Unknown'
                    })
                    
    except Exception as e:
        # Last resort: try to extract any readable text
        text_content = str(comments_data)
        if len(text_content) > 20:  # Only if there's substantial content
            clean_comments.append({
                'text': text_content[:200] + "..." if len(text_content) > 200 else text_content,
                'author': 'Unknown'
            })
    
    return clean_comments

st.set_page_config(page_title="URS Gemini Intelligence", layout="wide")
st.title("🧠 URS Extraction")

# Semantic Matching Information
with st.expander("ℹ️ About Semantic Matching (Meaning-Based)", expanded=False):
    st.markdown("""
    ### 🎯 Intelligent Meaning-Based Matching
    
    This system uses **semantic matching** to understand the **meaning** of requirements, not just word overlap.
    
    **Key Features:**
    - ✅ **Finds true matches**: Different words, same meaning → MATCH
      - Example: "Provide certificates" ↔ "Supply documentation" ✓
    
    - ❌ **Avoids false matches**: Same words, different meaning → NO MATCH
      - Example: "Metallic materials" ↔ "Non-metallic materials" ✗
      - Example: "System validation" ↔ "Validation system" ✗
    
    **How It Works:**
    1. **Semantic Embeddings**: Converts text to meaning representations
    2. **Similarity Calculation**: Measures how similar meanings are
    3. **Semantic Validation**: Detects false positives (opposite meanings, context differences)
    
    **Matching Thresholds:**
    - 🎯 Master Database: **0.75** (strict - authoritative source)
    - 📄 DOCX Comments: **0.50** (moderate - same document context)
    - 🗄️ Historical DB: **0.40** (lenient - cross-document matching)
    
    **Match Quality:**
    - `0.90-1.00`: Excellent (nearly identical meaning)
    - `0.75-0.89`: Good (same concept, different wording)
    - `0.60-0.74`: Moderate (related concepts)
    - `0.40-0.59`: Weak (some similarity)
    
    📖 See `SEMANTIC_MATCHING_GUIDE.md` for detailed examples and technical info.
    """)

# ---------------- Initialize Vector Store (same as standalone script) ----------------
@st.cache_resource
def init_vectorstore():
    try:
        # Use Gemini-specific PostgreSQL database (urs_gemini)
        return PostgresVectorStoreGemini()
    except Exception as e:
        st.error(f"❌ Failed to initialize PostgreSQL Gemini connection: {e}")
        st.stop()

@st.cache_resource
def init_master_database():
    """Initialize the master database layer."""
    try:
        if MASTER_DB_AVAILABLE:
            # Use absolute path relative to project root
            import os
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            excel_path = os.path.join(project_root, "URS Response Automation Master Database (1).xlsm")
            master_db = MasterDatabase(excel_path)
            return master_db
        return None
    except Exception as e:
        st.warning(f"⚠️ Could not initialize master database: {e}")
        return None

vectorstore = init_vectorstore()
master_db = init_master_database()

# Display database status
with st.sidebar:
    st.header("📊 Database Status")
    
    # Master Database Status
    if master_db:
        with st.expander("📋 Master Database (Excel)", expanded=True):
            stats = master_db.get_statistics()
            st.metric("Total Requirements", stats['total_requirements'])
            st.metric("With Responses", stats['requirements_with_responses'])
            st.metric("Without Responses", stats['requirements_without_responses'])
            st.info("✅ Master DB will be checked first before historical DB")
    else:
        st.warning("⚠️ Master Database not available")
    
    # Historical PostgreSQL Database Status
    with st.expander("🗄️ Historical Database (PostgreSQL)", expanded=False):
        try:
            docs = vectorstore.get_all_documents()
            if docs:
                st.success(f"✅ Connected to PostgreSQL")
                st.info(f"📚 {len(docs)} documents")
                
                total_reqs = sum(doc['requirement_count'] for doc in docs)
                st.info(f"📝 {total_reqs} total requirements")
                
                # Show document list
                for doc in docs:
                    st.write(f"• **{doc['filename']}**: {doc['requirement_count']} reqs")
            else:
                st.warning("⚠️ Database is empty")
        except Exception as e:
            st.error(f"❌ Database error: {str(e)}")

# Check for Gemini API key (only the Gemini backend needs one)
gemini_api_key = os.environ.get("GEMINI_API_KEY")
if USING_GEMINI and not gemini_api_key:
    st.error("❌ GEMINI_API_KEY not found in environment variables")
    st.stop()

if not GEMINI_PROCESSOR_AVAILABLE:
    st.error("❌ Extraction processor not available")
    st.stop()

st.caption(f"⚙️ Extraction backend: **{BACKEND_LABEL}** (set EXTRACTION_BACKEND to change)")

# ---------------- Document Processing Section ----------------
st.header("📄 Document Processing")

# Processing mode selection
processing_mode = st.radio(
    "Select Processing Mode:",
    ["🔍 Extract + Match Requirements", "💾 Extract + Store Requirements", "💬 Extract Comments & Responses"],
    help="Choose whether to match against existing data, store new requirements, or extract comments/responses"
)

# User comments
user_comments = st.text_area(
    "📝 Comments (Optional)",
    placeholder="Add any comments about this document analysis...",
    help="Optional comments that will be stored with the requirements"
)

# Force reprocess option
force_reprocess = st.checkbox(
    "🔄 Force Reprocess",
    help="Reprocess document even if it already exists in database"
)

# File upload - now supports multiple types
uploaded_file = st.file_uploader(
    "Choose a document file",
    type=['docx', 'doc', 'pdf', 'xlsx', 'xls'],
    help="Upload a DOCX, PDF, or Excel file for requirement extraction or comment analysis"
)

if uploaded_file is not None:
    # Auto-process when file is uploaded
    filename = uploaded_file.name
    
    try:
        # Check if file already exists in database
        docs = vectorstore.get_all_documents()
        file_exists = any(doc['filename'] == filename for doc in docs)
        
        # Determine processing strategy
        should_process = force_reprocess or not file_exists
        
        if file_exists and not force_reprocess:
            st.info(f"ℹ️ '{filename}' already exists in database. Using existing data for matching analysis...")
            
            # Get existing requirements from database instead of re-extracting
            with st.spinner(f"📊 Loading existing requirements for {filename}..."):
                # Debug: Show what documents are in the database
                st.write("🔍 **Debug: Documents in database:**")
                for doc in docs:
                    st.write(f"- `{doc['filename']}` ({doc['requirement_count']} requirements)")
                
                # Use the proper method to get requirements by document name
                file_requirements = vectorstore.search_requirements_by_document(filename)
                
                # Debug: Show search result
                st.write(f"🔍 **Debug: Searching for:** `{filename}`")
                st.write(f"🔍 **Debug: Found {len(file_requirements)} requirements**")
                
                if not file_requirements:
                    st.error(f"❌ No requirements found in database for {filename}")
                    
                    # Try alternative search approaches
                    st.write("🔍 **Trying alternative searches:**")
                    
                    # Try partial name search
                    base_name = filename.replace('.docx', '').replace('.pdf', '').replace('.xlsx', '')
                    alt_requirements = vectorstore.search_requirements_by_document(base_name)
                    st.write(f"- Base name search (`{base_name}`): {len(alt_requirements)} results")
                    
                    # Try searching with just the core part of the name
                    if 'GLATT' in filename:
                        glatt_requirements = vectorstore.search_requirements_by_document('GLATT')
                        st.write(f"- GLATT search: {len(glatt_requirements)} results")
                        if glatt_requirements:
                            file_requirements = glatt_requirements
                    
                    if not file_requirements:
                        st.stop()
                
                # Convert to the format expected by rest of the code
                requirements = []
                for i, req in enumerate(file_requirements):
                    requirements.append({
                        'id': f"EXISTING_{i+1}",
                        'text': req['requirement'],  # Use 'requirement' field from database
                        'category': 'existing',  # Default category for existing requirements
                        'priority': 'medium',   # Default priority for existing requirements
                        'confidence': 0.95      # Default confidence for existing requirements
                    })
                
                st.success(f"✅ Loaded {len(requirements)} existing requirements from database")
        
        else:
            # Process the document (new file or forced reprocessing)
            with st.spinner(f"🧠 Processing {filename} with {BACKEND_LABEL}..."):
                # Step 1: Initialize the configured extraction processor
                st.info(f"🧠 Initializing {BACKEND_LABEL} processor...")
                gemini = get_processor()
                st.success(f"✅ {BACKEND_LABEL} processor initialized")

                # Step 2: Get text out of the document.
                # The OCR backend parses PDFs/images with the vision model; DOCX
                # has a native text layer, so it is read directly either way.
                st.info("📄 Extracting text from document...")
                is_docx = filename.lower().endswith(('.docx', '.doc'))

                if not USING_GEMINI and not is_docx:
                    import tempfile as _tempfile
                    uploaded_file.seek(0)
                    suffix = os.path.splitext(filename)[1] or '.pdf'
                    with _tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name
                    try:
                        st.info("🔍 Running Unlimited-OCR on the document pages...")
                        full_text = gemini.parse_document(tmp_path)
                    finally:
                        try:
                            os.unlink(tmp_path)
                        except OSError:
                            pass
                else:
                    uploaded_file.seek(0)
                    full_text = extract_text_from_file(uploaded_file, filename)

                if not full_text or len(full_text.strip()) < 100:
                    st.error("❌ Failed to extract meaningful text from document")
                    st.error(f"Extracted text length: {len(full_text) if full_text else 0}")
                    st.stop()

                st.success(f"✅ Extracted {len(full_text):,} characters from document")

                # Get file bytes for DOCX comment extraction
                uploaded_file.seek(0)
                file_bytes = uploaded_file.read() if is_docx else None
                
                # Step 3: Choose extraction type based on processing mode
                # Initialize variables for broader scope
                extraction_result = None
                requirement_comment_pairs = []
                
                if processing_mode == "💬 Extract Comments & Responses":
                    st.info("💬 Extracting comments and responses...")
                    st.info("⏳ Analyzing document for comments, replies, and author information...")
                    
                    comments_data = gemini.extract_comments_and_responses(
                        full_document_text=full_text,
                        document_name=filename,
                        file_bytes=file_bytes
                    )
                    
                    if not comments_data:
                        st.warning("⚠️ No comments or responses found in this document")
                        st.info("This might be a document without markup/comments, or the format may not be recognized")
                        st.stop()
                    
                    st.success(f"✅ Successfully extracted {len(comments_data)} requirements with comments!")
                    
                    # Display comments analysis
                    with st.expander("💬 Comments Analysis Summary", expanded=True):
                        total_comments = sum(len(item.get('comments', [])) for item in comments_data)
                        authors = set()
                        comment_types = set()
                        
                        for item in comments_data:
                            for comment in item.get('comments', []):
                                if isinstance(comment, dict):
                                    authors.add(comment.get('author', 'Unknown'))
                                    comment_types.add(comment.get('comment_type', 'response'))
                                else:
                                    authors.add('Unknown')
                                    comment_types.add('text')
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("📝 Requirements with Comments", len(comments_data))
                        with col2:
                            st.metric("💬 Total Comments", total_comments)
                        with col3:
                            st.metric("👥 Authors", len(authors))
                        
                        st.write("**👥 Authors found:**", ", ".join(sorted(authors)))
                        st.write("**📋 Comment types:**", ", ".join(sorted(comment_types)))
                    
                    # Display detailed comments
                    st.subheader("💬 Requirements with Comments")
                    for i, item in enumerate(comments_data):
                        with st.expander(f"📄 Requirement {i+1}: {item.get('requirement_text', '')[:100]}...", expanded=i < 3):
                            st.write("**Original Requirement:**")
                            st.write(item.get('requirement_text', ''))
                            
                            st.write("**Comments & Responses:**")
                            for j, comment in enumerate(item.get('comments', [])):
                                if isinstance(comment, dict):
                                    st.write(f"**{j+1}. [{comment.get('comment_type', 'response')}] {comment.get('author', 'Unknown')}:**")
                                    st.write(f"_{comment.get('comment_text', '')}_")
                                elif isinstance(comment, str):
                                    st.write(f"**{j+1}. [text] Unknown:**")
                                    st.write(f"_{comment}_")
                            
                            if item.get('page_reference'):
                                st.caption(f"Page: {item.get('page_reference')}")
                    
                    # Store in database with enhanced format
                    if st.button("💾 Save Comments to Database"):
                        with st.spinner("Saving comments to database..."):
                            # Extract requirements and comments for PostgresVectorStoreGemini
                            requirement_texts = [item.get('requirement_text', '') for item in comments_data]
                            comments_metadata = []
                            
                            for item in comments_data:
                                for comment in item.get('comments', []):
                                    if isinstance(comment, dict):
                                        comments_metadata.append({
                                            'text': comment.get('comment_text', ''),
                                            'author': comment.get('author', 'Unknown'),
                                            'type': comment.get('comment_type', 'response'),
                                        'associated_requirement': item.get('requirement_text', '')
                                    })
                            
                            try:
                                success = vectorstore.add_requirements(
                                    requirements=requirement_texts,
                                    document_name=filename,
                                    comments=str(comments_metadata) if comments_metadata else None
                                )
                                
                                if success:
                                    st.success(f"✅ Successfully saved {len(requirement_texts)} commented requirements to database!")
                                else:
                                    st.error("❌ Failed to save commented requirements")
                                    
                            except Exception as e:
                                st.error(f"❌ Error saving comments: {str(e)}")
                                st.warning(f"⚠️ Failed to save requirements: {str(e)[:100]}")
                    
                    st.stop()  # Exit early for comment mode
                
                else:
                    # Comprehensive requirement + comment extraction
                    st.info("🔍 Performing comprehensive requirement & comment extraction...")
                    st.info("⏳ Analyzing for both requirements and associated comments...")
                    
                    # Use the new comprehensive extraction method
                    extraction_result = gemini.extract_requirements_with_comments_holistically(
                        full_document_text=full_text,
                        document_name=filename,
                        file_bytes=file_bytes
                    )
                    
                    requirements = extraction_result.get('requirements', [])
                    requirement_comment_pairs = extraction_result.get('requirement_comment_pairs', [])
                    total_comments = extraction_result.get('total_comments', 0)
                    
                    if not requirements:
                        st.error("❌ No requirements extracted through comprehensive analysis")
                        st.stop()
                    
                    # Show extraction results
                    st.success(f"✅ Successfully extracted {len(requirements)} requirements!")
                    if total_comments > 0:
                        st.success(f"💬 Found {total_comments} associated comments with precise requirement pairing!")
                        
                        # Display comment pairing summary
                        with st.expander("💬 Requirement-Comment Pairing Summary", expanded=True):
                            if requirement_comment_pairs:
                                paired_reqs = [pair for pair in requirement_comment_pairs if pair.get('comments')]
                                unpaired_reqs = [pair for pair in requirement_comment_pairs if not pair.get('comments')]
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write(f"**📝 Requirements with Comments:** {len(paired_reqs)}")
                                    st.write(f"**📄 Requirements without Comments:** {len(unpaired_reqs)}")
                                
                                with col2:
                                    # Collect all authors and comment types from paired requirements
                                    authors = set()
                                    comment_types = set()
                                    for pair in paired_reqs:
                                        for comment in pair.get('comments', []):
                                            # Handle both dict and string comments
                                            if isinstance(comment, dict):
                                                authors.add(comment.get('author', 'Unknown'))
                                                comment_types.add(comment.get('comment_type', 'response'))
                                            elif isinstance(comment, str):
                                                authors.add('Unknown')
                                                comment_types.add('text')
                                    
                                    st.write("**👥 Comment Authors:**")
                                    for author in sorted(authors):
                                        st.write(f"• {author}")
                                    st.write("**� Comment Types:**")
                                    for ctype in sorted(comment_types):
                                        st.write(f"• {ctype}")
                                        
                                # Show sample pairings
                                if paired_reqs:
                                    st.write("**� Sample Requirement-Comment Pairings:**")
                                    for i, pair in enumerate(paired_reqs[:3]):  # Show first 3 pairs
                                        req_text = pair['requirement'].get('text', '')[:100] + "..."
                                        comment_count = len(pair.get('comments', []))
                                        st.write(f"• **Req {i+1}:** {req_text} → **{comment_count} comments**")
                    else:
                        st.info("ℹ️ No comments found associated with requirements")
        
        # Step 4: Display analysis summary for all cases
        with st.expander("📊 Extraction Summary", expanded=True):
            categories = {}
            priorities = {}
            
            for req in requirements:
                cat = req.get('category', 'unknown')
                pri = req.get('priority', 'unknown')
                categories[cat] = categories.get(cat, 0) + 1
                priorities[pri] = priorities.get(pri, 0) + 1
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**📂 By Category:**")
                for cat, count in sorted(categories.items()):
                    st.write(f"• {cat}: {count}")
            
            with col2:
                st.write("**⚡ By Priority:**")
                for pri, count in sorted(priorities.items()):
                    st.write(f"• {pri}: {count}")
        
        # Step 5: Save to database only if needed
        if should_process:
            st.info("💾 Saving to PostgreSQL Gemini database...")
            
            try:
                # requirement_comment_pairs is already available from broader scope
                # extraction_result is also available from broader scope
                
                # Debug: Show what we're about to save
                st.write(f"🔍 **Debug: Saving document as:** `{filename}`")
                st.write(f"🔍 **Debug: Saving {len(requirements)} requirements**")
                st.write(f"🔍 **Debug: Available requirement-comment pairs:** {len(requirement_comment_pairs)}")
                st.write(f"🔍 **Debug: Extraction result available:** {extraction_result is not None}")
                
                if requirement_comment_pairs and len(requirement_comment_pairs) > 0:
                    # Use new method with individual comment assignment
                    st.success("✅ Using NEW precise requirement-comment pairing method!")
                    st.info("💬 This will save each requirement with its specific comments...")
                    success = vectorstore.add_requirements_with_individual_comments(
                        requirement_comment_pairs=requirement_comment_pairs,
                        document_name=filename
                    )
                    
                    # Show pairing statistics
                    paired_count = sum(1 for pair in requirement_comment_pairs if pair.get('comments'))
                    st.write(f"� **Pairing Stats:** {paired_count}/{len(requirement_comment_pairs)} requirements have associated comments")
                    
                else:
                    # Fallback to old method for compatibility
                    st.warning("⚠️ Falling back to OLD standard requirement saving (all comments assigned to all requirements)...")
                    st.info("📝 Using legacy method - comments will not be individually paired...")
                    requirement_texts = [req['text'] for req in requirements]
                    
                    # Prepare comments metadata - extract from new structure if available
                    comments_metadata = []
                    if requirement_comment_pairs:
                        # Collect all comments from the pairs for fallback storage
                        all_comments = []
                        for pair in requirement_comment_pairs:
                            all_comments.extend(pair.get('comments', []))
                        if all_comments:
                            comments_metadata = all_comments
                    elif user_comments:
                        comments_metadata = [{'text': user_comments, 'author': 'User', 'type': 'general'}]
                    
                    success = vectorstore.add_requirements(
                        requirements=requirement_texts,
                        document_name=filename,
                        comments=str(comments_metadata) if comments_metadata else user_comments
                    )
                
                if success:
                    st.success(f"✅ Successfully saved {len(requirements)} requirements to database!")
                    
                    # Debug: Verify what was actually saved
                    st.write("🔍 **Debug: Verifying save by searching...**")
                    verify_reqs = vectorstore.search_requirements_by_document(filename)
                    st.write(f"- Found {len(verify_reqs)} requirements after save")
                else:
                    st.error("❌ Failed to save requirements to database")
                    
            except Exception as e:
                st.error(f"❌ Error saving to database: {str(e)}")
                st.warning(f"⚠️ Failed to save requirements: {str(e)[:100]}")
        else:
            st.info("💾 Skipping database save (file already exists)")
        
        # Step 6: SHOW MATCHING TABLE (ALWAYS FOR EXTRACT+MATCH MODE)
        if processing_mode == "🔍 Extract + Match Requirements":
            st.divider()
            st.subheader("📊 Requirement Matching with Historical Data")
            
            with st.spinner("🔍 Comparing new requirements with Master Database and Historical data..."):
                # LAYER 1: Check Master Database first (Excel)
                master_db_matches = {}
                if master_db:
                    st.info("🔍 **Layer 1:** Checking Master Database (Excel)...")
                    for req in requirements:
                        # Use threshold 0.75 for Master DB - stricter for authoritative source
                        # Ensures high-confidence semantic matches only
                        match = master_db.search_requirement(req['text'], threshold=0.75)
                        if match:
                            master_db_matches[req['text']] = match
                    
                    if master_db_matches:
                        st.success(f"✅ Found {len(master_db_matches)} matches in Master Database!")
                    else:
                        st.info("ℹ️ No matches found in Master Database")
                
                # LAYER 2: Check Historical PostgreSQL Database
                st.info("🔍 **Layer 2:** Checking Historical PostgreSQL Database...")
                
                # Get ALL requirements from database for comparison
                all_requirements = vectorstore.get_all_requirements()
                
                # Filter out the current file's requirements for historical comparison
                historical_requirements = [
                    req for req in all_requirements 
                    if req.get('document_name', '') != filename
                ]
                
                st.info(f"Found {len(historical_requirements)} historical requirements from other documents for comparison")
                
                # Create matching table data
                matching_data = []
                
                # Search each new requirement against both databases
                for req_idx, req in enumerate(requirements):
                    try:
                        req_text = req['text']
                        
                        # Check if we have a Master DB match first (priority)
                        master_match_found = False
                        if req_text in master_db_matches:
                            master_match = master_db_matches[req_text]
                            master_match_found = True
                            
                            matching_data.append({
                                'New Requirement': req_text[:200] + '...' if len(req_text) > 200 else req_text,
                                'Category': req.get('category', 'Unknown'),
                                'Priority': req.get('priority', 'Unknown'),
                                'Matched Requirement': master_match['requirement'][:200] + '...' if len(master_match['requirement']) > 200 else master_match['requirement'],
                                'Match Source': f"Master DB ({master_match['deviation_id']})",
                                'Historical Comments': master_match['response'][:150] + '...' if len(master_match['response']) > 150 else master_match['response'],
                                'Historical Responses': master_match['response'][:150] + '...' if len(master_match['response']) > 150 else master_match['response'],
                                'Similarity Score': f"{master_match['similarity']:.2f}",
                                'Has Match': 'Yes - Master DB',
                                'Match Type': master_match['match_type']
                            })
                        
                        # ALWAYS search PostgreSQL historical database (even if Master DB match found)
                        # Use correct search method from PostgresVectorStoreGemini
                        search_results = vectorstore.search_similar_requirements(
                            query=req_text,
                            top_k=3,  # Get top 3 matches
                            threshold=0.3  # Use 0.3 threshold for cross-document matching
                        )
                        
                        # DEBUG: Log search results
                        if search_results:
                            print(f"\n🔍 PostgreSQL search for: {req_text[:60]}...")
                            print(f"   Found {len(search_results)} results")
                            for i, res in enumerate(search_results, 1):
                                print(f"   {i}. Score: {res.get('similarity_score', 0):.3f}, Doc: {res.get('document_name', 'Unknown')}, Match: {res.get('requirement', '')[:50]}...")
                        
                        # Find the best match from a different document
                        best_match = None
                        best_score = 0
                        
                        if search_results:
                            print(f"   Current document name: '{filename}'")
                            for result in search_results:
                                # Check if this is from a different document
                                result_filename = result.get('document_name', '')
                                score = result.get('similarity_score', 0)
                                print(f"   Comparing: '{result_filename}' != '{filename}' ? {result_filename != filename}, Score: {score:.3f}")
                                if result_filename != filename and score > best_score:
                                    best_match = result
                                    best_score = score
                            
                            if best_match:
                                print(f"   ✅ Best match: {best_score:.3f} from '{best_match.get('document_name', '')}'")
                            else:
                                print(f"   ⚠️ No matches from different documents (all matches are from same document)")
                        else:
                            print(f"   ❌ No search results returned from PostgreSQL")
                        
                        if best_match and best_score >= 0.3:
                            # Check if the matched requirement has comments
                            matched_comments = ""
                            matched_responses = ""
                            
                            # Get the full requirement data to check for comments
                            full_matched_req = None
                            for hist_req in historical_requirements:
                                if hist_req['requirement'] == best_match['requirement']:
                                    full_matched_req = hist_req
                                    break
                            
                            if full_matched_req:
                                comments_data = full_matched_req.get('comments')
                                if comments_data:
                                    try:
                                        # Parse comments properly using the extract_clean_comments function
                                        clean_comments = extract_clean_comments(comments_data)
                                        
                                        if clean_comments:
                                            # Format comments nicely with full text (not truncated)
                                            comment_texts = []
                                            for comment in clean_comments[:3]:  # Show up to 3 comments
                                                text = comment.get('text', '')
                                                author = comment.get('author', 'Unknown')
                                                if text:
                                                    # Show full comment text, not truncated
                                                    comment_texts.append(f"[{author}] {text}")
                                            
                                            matched_comments = " | ".join(comment_texts)
                                            matched_responses = matched_comments  # Use same data for responses
                                            
                                            if len(clean_comments) > 3:
                                                matched_comments += f" ... (+{len(clean_comments) - 3} more)"
                                                matched_responses = matched_comments
                                        else:
                                            matched_comments = "No comments"
                                            matched_responses = "No responses"
                                    except Exception as e:
                                        matched_comments = f"Error parsing: {str(e)[:50]}"
                                        matched_responses = matched_comments
                            
                            matching_data.append({
                                'New Requirement': req_text[:200] + '...' if len(req_text) > 200 else req_text,
                                'Category': req.get('category', 'Unknown'),
                                'Priority': req.get('priority', 'Unknown'),
                                'Matched Requirement': best_match['requirement'][:200] + '...' if len(best_match['requirement']) > 200 else best_match['requirement'],
                                'Match Source': best_match.get('document_name', 'Unknown'),
                                'Historical Comments': matched_comments or 'No comments',
                                'Historical Responses': matched_responses or 'No responses',
                                'Similarity Score': f"{best_score:.2f}",
                                'Has Match': 'Yes - PostgreSQL',
                                'Match Type': 'semantic'
                            })  
                        else:
                            # No match found in either database
                            matching_data.append({
                                'New Requirement': req_text[:200] + '...' if len(req_text) > 200 else req_text,
                                'Category': req.get('category', 'Unknown'),
                                'Priority': req.get('priority', 'Unknown'),
                                'Matched Requirement': 'No historical match found',
                                'Match Source': '-',
                                'Historical Comments': '-',
                                'Historical Responses': '-',
                                'Similarity Score': '0.00',
                                'Has Match': 'No',
                                'Match Type': '-'
                            })
                        
                    except Exception as e:
                        # Add error entry
                        matching_data.append({
                            'New Requirement': req['text'],
                            'Category': req.get('category', 'Unknown'),
                            'Priority': req.get('priority', 'Unknown'),
                            'Matched Requirement': f'Search error: {str(e)[:50]}',
                            'Match Source': 'Error',
                            'Similarity Score': '0.00',
                            'Has Match': 'Error'
                        })
            
            # Display the matching table - Split into 3 tables
            if matching_data:
                import pandas as pd
                
                # Separate data into three categories
                deviation_list_data = [x for x in matching_data if x.get('Has Match', '').startswith('Yes - Master DB')]
                historical_matches_data = [x for x in matching_data if x.get('Has Match', '') == 'Yes - PostgreSQL' and float(x.get('Similarity Score', '0.0')) >= 0.7]
                no_match_data = [x for x in matching_data if x.get('Has Match', '') == 'No' or (x.get('Has Match', '') == 'Yes - PostgreSQL' and float(x.get('Similarity Score', '0.0')) < 0.7)]
                
                # Create enhanced summary statistics
                total_requirements = len(matching_data)
                
                # Display summary
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📝 Total New Requirements", total_requirements)
                with col2:
                    st.metric("📋 Deviation List Matches", len(deviation_list_data), help="Matched from Excel Master Database")
                with col3:
                    st.metric("🗄️ Historical Matches (≥70%)", len(historical_matches_data), help="PostgreSQL matches with similarity ≥ 0.7")
                with col4:
                    st.metric("❌ No Match / Low Similarity", len(no_match_data), help="No match or similarity < 0.7")
                
                st.subheader("📊 Multi-Layer Requirement Matching Results")
                st.caption("✅ Results split into 3 tables: Deviation List (Master DB), Historical Matches (≥70%), and No Match/Low Similarity (<70%)")
                
                # === TABLE 1: Deviation List (Master Database Matches) ===
                if deviation_list_data:
                    st.markdown("---")
                    st.subheader("📋 Table 1: Deviation List (Master Database Matches)")
                    st.caption(f"✅ {len(deviation_list_data)} requirements matched from Excel Master Database")
                    
                    deviation_df = pd.DataFrame(deviation_list_data)
                    deviation_df['Deviations'] = deviation_df['Historical Responses']
                    
                    column_order = ['New Requirement', 'Category', 'Priority', 'Deviations', 
                                  'Similarity Score', 'Matched Requirement', 'Match Source']
                    column_order = [col for col in column_order if col in deviation_df.columns]
                    deviation_df = deviation_df[column_order]
                    
                    edited_deviation_df = st.data_editor(
                        deviation_df,
                        width='stretch',
                        hide_index=True,
                        column_config={
                            "Deviations": st.column_config.TextColumn("Deviations / Response", max_chars=1000, width="large"),
                            "New Requirement": st.column_config.TextColumn("New Requirement", width="large"),
                            "Matched Requirement": st.column_config.TextColumn("Matched Requirement", width="large")
                        },
                        disabled=['New Requirement', 'Category', 'Priority', 'Similarity Score', 
                                'Matched Requirement', 'Match Source'],
                        key="deviation_table"
                    )
                    
                    # Export button for Deviation List
                    csv_deviation = edited_deviation_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Deviation List as CSV",
                        data=csv_deviation,
                        file_name=f"deviation_list_{filename.replace('.docx', '')}.csv",
                        mime="text/csv",
                        key="download_deviation"
                    )
                else:
                    st.info("ℹ️ No matches found in Master Database")
                
                # === TABLE 2: Historical Matches (PostgreSQL ≥70%) ===
                if historical_matches_data:
                    st.markdown("---")
                    st.subheader("🗄️ Table 2: Historical Matches (Similarity ≥ 70%)")
                    st.caption(f"✅ {len(historical_matches_data)} requirements matched from PostgreSQL with high similarity")
                    
                    historical_df = pd.DataFrame(historical_matches_data)
                    historical_df['Deviations'] = ''  # Empty for user input
                    
                    column_order = ['New Requirement', 'Category', 'Priority', 'Deviations', 
                                  'Similarity Score', 'Matched Requirement', 'Match Source', 
                                  'Historical Comments', 'Historical Responses']
                    column_order = [col for col in column_order if col in historical_df.columns]
                    historical_df = historical_df[column_order]
                    
                    edited_historical_df = st.data_editor(
                        historical_df,
                        width='stretch',
                        hide_index=True,
                        column_config={
                            "Deviations": st.column_config.TextColumn("Deviations / Response", max_chars=1000, width="large"),
                            "New Requirement": st.column_config.TextColumn("New Requirement", width="large"),
                            "Matched Requirement": st.column_config.TextColumn("Matched Requirement", width="large"),
                            "Historical Comments": st.column_config.TextColumn("Historical Comments", width="large"),
                            "Historical Responses": st.column_config.TextColumn("Historical Responses", width="large")
                        },
                        disabled=['New Requirement', 'Category', 'Priority', 'Similarity Score', 
                                'Matched Requirement', 'Match Source', 'Historical Comments', 'Historical Responses'],
                        key="historical_table"
                    )
                    
                    # Export button for Historical Matches
                    csv_historical = edited_historical_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Historical Matches as CSV",
                        data=csv_historical,
                        file_name=f"historical_matches_{filename.replace('.docx', '')}.csv",
                        mime="text/csv",
                        key="download_historical"
                    )
                else:
                    st.info("ℹ️ No high-similarity matches found in PostgreSQL")
                
                # === TABLE 3: No Match / Low Similarity (<70%) ===
                if no_match_data:
                    st.markdown("---")
                    st.subheader("❌ Table 3: No Match / Low Similarity (< 70%)")
                    st.caption(f"⚠️ {len(no_match_data)} requirements with no match or low similarity - Manual review needed")
                    
                    no_match_df = pd.DataFrame(no_match_data)
                    no_match_df['Deviations'] = ''  # Empty for user input
                    
                    column_order = ['New Requirement', 'Category', 'Priority', 'Deviations', 
                                  'Similarity Score', 'Matched Requirement', 'Match Source']
                    column_order = [col for col in column_order if col in no_match_df.columns]
                    no_match_df = no_match_df[column_order]
                    
                    edited_no_match_df = st.data_editor(
                        no_match_df,
                        width='stretch',
                        hide_index=True,
                        column_config={
                            "Deviations": st.column_config.TextColumn("Deviations / Response (Manual Input Required)", 
                                                                     max_chars=1000, width="large"),
                            "New Requirement": st.column_config.TextColumn("New Requirement", width="large"),
                            "Matched Requirement": st.column_config.TextColumn("Matched Requirement", width="medium")
                        },
                        disabled=['New Requirement', 'Category', 'Priority', 'Similarity Score', 
                                'Matched Requirement', 'Match Source'],
                        key="no_match_table"
                    )
                    
                    # Export button for No Match
                    csv_no_match = edited_no_match_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download No Match List as CSV",
                        data=csv_no_match,
                        file_name=f"no_match_list_{filename.replace('.docx', '')}.csv",
                        mime="text/csv",
                        key="download_no_match"
                    )
                else:
                    st.success("✅ All requirements have high-confidence matches!")
                
                # Export all combined
                st.markdown("---")
                st.subheader("📦 Export All Results")
                
                # Combine all for full export
                all_data = deviation_list_data + historical_matches_data + no_match_data
                all_df = pd.DataFrame(all_data)
                if 'Deviations' not in all_df.columns:
                    all_df['Deviations'] = all_df.apply(
                        lambda row: row.get('Historical Responses', '') if row.get('Has Match', '').startswith('Yes - Master DB') else '',
                        axis=1
                    )
                
                csv_all = all_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download All Results as CSV",
                    data=csv_all,
                    file_name=f"all_matching_results_{filename.replace('.docx', '')}.csv",
                    mime="text/csv",
                    help="Download all three tables combined"
                )
            
            st.success("🎯 Multi-layer historical matching analysis complete!")
        
        else:
            st.success("💾 Requirements extracted and stored successfully!")
        
        # Display sample requirements
        with st.expander("📝 Sample Extracted Requirements", expanded=False):
            for i, req in enumerate(requirements[:5]):  # Show first 5
                st.write(f"**{i+1}.** {req['text']}")
                st.caption(f"Category: {req.get('category', 'N/A')} | Priority: {req.get('priority', 'N/A')} | Confidence: {req.get('confidence', 'N/A')}")
            
            if len(requirements) > 5:
                st.caption(f"... and {len(requirements) - 5} more requirements")
        
        st.success("🎉 Document processing complete!")
        
        # Note: Removed automatic refresh to prevent endless loop
        
    except Exception as e:
        st.error(f"❌ Error during document processing: {str(e)}")
        st.exception(e)

# ---------------- Database Management ----------------
st.header("🗄️ Database Management")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📊 Show All Documents"):
        docs = vectorstore.get_all_documents()
        if docs:
            st.write("**Documents in database:**")
            for doc in docs:
                st.write(f"• {doc['filename']}: {doc['requirement_count']} requirements")
        else:
            st.info("No documents in database")

with col2:
    if st.button("🗑️ Clear Database"):
        if st.session_state.get('confirm_clear'):
            vectorstore.clear_database()
            st.success("Database cleared!")
            st.session_state.confirm_clear = False
            st.rerun()
        else:
            st.session_state.confirm_clear = True
            st.warning("⚠️ Click again to confirm database clearing")

with col3:
    if st.button("🔄 Refresh Stats"):
        st.rerun()

# ---------------- Requirements & Comments Viewer Section ----------------
st.header("🔍 Requirements & Comments Viewer")
st.markdown("View requirements from existing documents with their associated comments and authors")

# Document selection for viewing
docs = vectorstore.get_all_documents()
if docs:
    # Create document selection
    doc_options = ["Select a document..."] + [doc['filename'] for doc in docs]
    selected_doc = st.selectbox(
        "📄 Select Document to View:",
        doc_options,
        key="doc_viewer_select"
    )
    
    if selected_doc and selected_doc != "Select a document...":
        with st.spinner("Loading requirements and comments..."):
            requirements = vectorstore.search_requirements_by_document(selected_doc)
            
        if requirements:
            st.success(f"✅ Found {len(requirements)} requirements in {selected_doc}")
            
            # Filter and display options
            col1, col2, col3 = st.columns(3)
            
            with col1:
                show_comments_only = st.checkbox("💬 Show only requirements with comments", key="filter_comments")
            
            with col2:
                show_authors = st.checkbox("👥 Show comment authors", value=True, key="show_authors")
                
            with col3:
                max_display = st.selectbox("📊 Show", [10, 25, 50, 100, "All"], index=1, key="max_display")
            
            # Filter requirements based on options
            filtered_reqs = requirements
            if show_comments_only:
                filtered_reqs = [req for req in requirements if req.get('comments')]
            
            # Limit display count
            if max_display != "All":
                filtered_reqs = filtered_reqs[:max_display]
            
            st.info(f"📋 Displaying {len(filtered_reqs)} of {len(requirements)} requirements")
            
            # Display requirements with comments
            for i, req in enumerate(filtered_reqs, 1):
                with st.expander(f"📝 Requirement {i}: {req.get('requirement', '')[:100]}{'...' if len(req.get('requirement', '')) > 100 else ''}", expanded=False):
                    
                    # Display requirement text
                    st.markdown("**🎯 Requirement:**")
                    st.write(req.get('requirement', 'No requirement text'))
                    
                    # Display metadata if available
                    metadata = req.get('metadata', {})
                    if metadata:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            if metadata.get('category'):
                                st.write(f"**Category:** {metadata.get('category')}")
                        with col2:
                            if metadata.get('priority'):
                                st.write(f"**Priority:** {metadata.get('priority')}")
                        with col3:
                            if metadata.get('confidence'):
                                st.write(f"**Confidence:** {metadata.get('confidence')}")
                    
                    # Display comments if available
                    comments_data = req.get('comments')
                    if comments_data:
                        # Use the helper function to extract clean comments
                        clean_comments = extract_clean_comments(comments_data)
                        
                        if clean_comments:
                            st.markdown("**💬 Associated Comments:**")
                            st.write(f"📊 **{len(clean_comments)} comment(s)** found")
                            
                            for j, comment in enumerate(clean_comments, 1):
                                with st.container():
                                    # Comment details in columns
                                    comment_col1, comment_col2 = st.columns([4, 2])
                                    
                                    with comment_col1:
                                        st.markdown(f"**💭 Comment {j}:**")
                                        # Display the clean comment text
                                        comment_text = comment.get('text', 'No comment text')
                                        st.write(comment_text)
                                    
                                    with comment_col2:
                                        if show_authors:
                                            author = comment.get('author', 'Unknown')
                                            st.write(f"👤 **Author:** {author}")
                                        
                                        comment_type = comment.get('type')
                                        if comment_type:
                                            st.write(f"🏷️ **Type:** {comment_type}")
                                    
                                    if j < len(clean_comments):  # Don't add separator after last comment
                                        st.markdown("---")
                        else:
                            st.markdown("**💬 Comments:**")
                            st.write("Unable to parse comment data properly")
                            # Show raw data for debugging (limited)
                            st.caption(f"Raw data: {str(comments_data)[:100]}...")
                    else:
                        st.info("ℹ️ No comments associated with this requirement")
                    
                    # Show extraction metadata
                    extraction_type = req.get('extraction_type', 'unknown')
                    if extraction_type:
                        st.caption(f"🔬 Extraction method: {extraction_type}")
        else:
            st.warning(f"No requirements found for {selected_doc}")
else:
    st.info("No documents available in database. Upload and process documents first.")