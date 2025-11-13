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

# Import Gemini processor for holistic extraction
try:
    from utils.gemini_processor import GeminiProcessor
    GEMINI_PROCESSOR_AVAILABLE = True
except ImportError:
    GEMINI_PROCESSOR_AVAILABLE = False
    st.error("❌ Gemini processor not available. Please ensure utils/gemini_processor.py exists.")

# Load environment variables
load_dotenv()

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

# ---------------- Initialize Vector Store (same as standalone script) ----------------
@st.cache_resource
def init_vectorstore():
    try:
        # Use Gemini-specific PostgreSQL database (urs_gemini)
        return PostgresVectorStoreGemini()
    except Exception as e:
        st.error(f"❌ Failed to initialize PostgreSQL Gemini connection: {e}")
        st.stop()

vectorstore = init_vectorstore()

# Display database status
with st.sidebar:
    st.header("📊 Database Status")
    try:
        docs = vectorstore.get_all_documents()
        if docs:
            st.success(f"✅ Connected to PostgreSQL")
            st.info(f"📚 {len(docs)} documents in database")
            
            total_reqs = sum(doc['requirement_count'] for doc in docs)
            st.info(f"📝 {total_reqs} total requirements")
            
            # Show document list
            with st.expander("📄 Documents", expanded=False):
                for doc in docs:
                    st.write(f"• **{doc['filename']}**: {doc['requirement_count']} reqs")
        else:
            st.warning("⚠️ Database is empty")
    except Exception as e:
        st.error(f"❌ Database error: {str(e)}")

# Check for Gemini API key
gemini_api_key = os.environ.get("GEMINI_API_KEY")
if not gemini_api_key:
    st.error("❌ GEMINI_API_KEY not found in environment variables")
    st.stop()

if not GEMINI_PROCESSOR_AVAILABLE:
    st.error("❌ Gemini processor not available")
    st.stop()

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
            # Process the document with Gemini (new file or forced reprocessing)
            with st.spinner(f"🧠 Processing {filename} with Gemini..."):
                # Step 1: Extract text from document (universal extractor)
                st.info("📄 Extracting text from document...")
                uploaded_file.seek(0)
                full_text = extract_text_from_file(uploaded_file, filename)
                
                if not full_text or len(full_text.strip()) < 100:
                    st.error("❌ Failed to extract meaningful text from document")
                    st.error(f"Extracted text length: {len(full_text) if full_text else 0}")
                    st.stop()
                
                st.success(f"✅ Extracted {len(full_text):,} characters from document")
                
                # Get file bytes for DOCX comment extraction
                uploaded_file.seek(0)
                file_bytes = uploaded_file.read() if filename.lower().endswith('.docx') else None
                
                # Step 2: Initialize Gemini processor
                st.info("🧠 Initializing Gemini processor...")
                gemini = GeminiProcessor(gemini_api_key)
                st.success("✅ Gemini processor initialized")
                
                # Step 3: Choose extraction type based on processing mode
                # Initialize variables for broader scope
                extraction_result = None
                requirement_comment_pairs = []
                
                if processing_mode == "💬 Extract Comments & Responses":
                    st.info("💬 Extracting comments and responses...")
                    st.info("⏳ Analyzing document for comments, replies, and author information...")
                    
                    comments_data = gemini.extract_comments_and_responses(
                        full_document_text=full_text,
                        document_name=filename
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
                                authors.add(comment.get('author', 'Unknown'))
                                comment_types.add(comment.get('comment_type', 'response'))
                        
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
                                st.write(f"**{j+1}. [{comment.get('comment_type', 'response')}] {comment.get('author', 'Unknown')}:**")
                                st.write(f"_{comment.get('comment_text', '')}_")
                            
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
                                            authors.add(comment.get('author', 'Unknown'))
                                            comment_types.add(comment.get('comment_type', 'response'))
                                    
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
            
            with st.spinner("🔍 Comparing new requirements with historical database..."):
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
                
                # Search each new requirement against existing database
                for req_idx, req in enumerate(requirements):
                    try:
                        # Use correct search method from PostgresVectorStoreGemini
                        search_results = vectorstore.search_similar_requirements(
                            query=req['text'],
                            top_k=3  # Get top 3 matches
                        )
                        
                        # Find the best match from a different document
                        best_match = None
                        best_score = 0
                        
                        if search_results:
                            for result in search_results:
                                # Check if this is from a different document
                                result_filename = result.get('document_name', '')
                                if result_filename != filename and result.get('similarity_score', 0) > best_score:
                                    best_match = result
                                    best_score = result.get('similarity_score', 0)
                        
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
                                        # Handle both string and list formats
                                        if isinstance(comments_data, str):
                                            matched_comments = comments_data[:200]
                                        elif isinstance(comments_data, list):
                                            comment_texts = []
                                            for comment in comments_data[:2]:
                                                if isinstance(comment, dict):
                                                    text = comment.get('text', '') or comment.get('comment_text', '')
                                                    author = comment.get('author', 'Unknown')
                                                    if text:
                                                        comment_texts.append(f"{author}: {text[:50]}")
                                                elif isinstance(comment, str):
                                                    comment_texts.append(comment[:100])
                                            matched_comments = "; ".join(comment_texts)
                                            matched_responses = matched_comments  # Use same data for responses
                                    except Exception as e:
                                        matched_comments = "Error parsing comments"
                            
                            matching_data.append({
                                'New Requirement': req['text'][:200] + '...' if len(req['text']) > 200 else req['text'],
                                'Category': req.get('category', 'Unknown'),
                                'Priority': req.get('priority', 'Unknown'),
                                'Matched Requirement': best_match['requirement'][:200] + '...' if len(best_match['requirement']) > 200 else best_match['requirement'],
                                'Match Source': best_match.get('document_name', 'Unknown'),
                                'Historical Comments': matched_comments[:100] + '...' if len(matched_comments) > 100 else matched_comments or 'No comments',
                                'Historical Responses': matched_responses[:100] + '...' if len(matched_responses) > 100 else matched_responses or 'No responses',
                                'Similarity Score': f"{best_score:.2f}",
                                'Has Match': 'Yes'
                            })
                        else:
                            # No match found
                            matching_data.append({
                                'New Requirement': req['text'][:200] + '...' if len(req['text']) > 200 else req['text'],
                                'Category': req.get('category', 'Unknown'),
                                'Priority': req.get('priority', 'Unknown'),
                                'Matched Requirement': 'No historical match found',
                                'Match Source': '-',
                                'Historical Comments': '-',
                                'Historical Responses': '-',
                                'Similarity Score': '0.00',
                                'Has Match': 'No'
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
            
            # Display the matching table
            if matching_data:
                import pandas as pd
                
                # Create summary statistics
                total_requirements = len(matching_data)
                matched_requirements = len([x for x in matching_data if x['Has Match'] == 'Yes'])
                no_match_requirements = len([x for x in matching_data if x['Has Match'] == 'No'])
                
                # Display summary
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📝 Total New Requirements", total_requirements)
                with col2:
                    st.metric("✅ Found Historical Matches", matched_requirements)
                with col3:
                    st.metric("❌ No Historical Matches", no_match_requirements)
                
                st.subheader("📊 New vs Historical Requirements Comparison")
                
                # Create and display the table
                display_df = pd.DataFrame(matching_data)
                
                # Display the table
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True
                )
            
            st.success("🎯 Historical matching analysis complete!")
        
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