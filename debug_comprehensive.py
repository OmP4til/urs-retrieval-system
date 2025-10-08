#!/usr/bin/env python3
"""
Comprehensive debugging script to check comment extraction and storage
"""

import os
import json
from dotenv import load_dotenv
from utils.postgres_vectorstore import PostgresVectorStore
from utils.db_models import get_session, Requirement, Document
from utils.extractors import extract_structured_content

# Load environment variables
load_dotenv()

def debug_database_comments():
    """Check what's actually stored in the database"""
    print("=== DEBUGGING DATABASE COMMENTS ===\n")
    
    try:
        DATABASE_URL = os.getenv("DATABASE_URL")
        vs = PostgresVectorStore(model_name="all-MiniLM-L6-v2", db_url=DATABASE_URL)
        
        # Get all documents
        docs = vs.get_all_documents()
        print(f"📚 Total documents in database: {len(docs)}")
        
        for doc in docs:
            print(f"\n📄 Document: {doc['filename']}")
            print(f"   Requirements: {doc['requirement_count']}")
            print(f"   Upload date: {doc['upload_date']}")
        
        # Check requirements and their comments
        session = get_session(vs.engine)
        try:
            print(f"\n=== CHECKING REQUIREMENTS FOR COMMENTS ===")
            
            # Get all requirements
            all_reqs = session.query(Requirement).all()
            print(f"Total requirements in database: {len(all_reqs)}")
            
            # Count requirements with non-empty comments
            reqs_with_comments = 0
            reqs_with_empty_comments = 0
            
            print(f"\n📊 Sample of requirements and their comments:")
            for i, req in enumerate(all_reqs[:10]):  # Show first 10
                print(f"\n{i+1}. Requirement ID: {req.requirement_id}")
                print(f"   Text: {req.text[:80]}...")
                print(f"   Source file: {req.document.filename if req.document else 'Unknown'}")
                print(f"   Page: {req.page_number}")
                print(f"   Comments type: {type(req.comments)}")
                print(f"   Comments: {req.comments}")
                
                if req.comments and len(req.comments) > 0:
                    reqs_with_comments += 1
                    print(f"   ✅ HAS COMMENTS!")
                else:
                    reqs_with_empty_comments += 1
                    print(f"   ❌ No comments")
            
            print(f"\n📈 SUMMARY:")
            print(f"   Requirements with comments: {reqs_with_comments}")
            print(f"   Requirements with empty comments: {reqs_with_empty_comments}")
            
            # If we have requirements with comments, show them
            if reqs_with_comments > 0:
                print(f"\n🔍 ALL REQUIREMENTS WITH COMMENTS:")
                for req in all_reqs:
                    if req.comments and len(req.comments) > 0:
                        print(f"\n📝 Requirement: {req.text[:100]}...")
                        print(f"   File: {req.document.filename if req.document else 'Unknown'}")
                        print(f"   Comments: {json.dumps(req.comments, indent=2)}")
            
        finally:
            session.close()
            
    except Exception as e:
        print(f"❌ Error checking database: {e}")

def debug_file_extraction(filename):
    """Test comment extraction on a specific file"""
    print(f"\n=== DEBUGGING FILE EXTRACTION: {filename} ===\n")
    
    if not os.path.exists(filename):
        print(f"❌ File {filename} not found!")
        return
    
    # Create a mock uploaded file object
    class MockFile:
        def __init__(self, path):
            self.name = os.path.basename(path)
            with open(path, 'rb') as f:
                self.content = f.read()
            self.pos = 0
        
        def read(self):
            return self.content
        
        def seek(self, pos):
            self.pos = pos
    
    try:
        mock_file = MockFile(filename)
        print(f"📄 Processing {filename}...")
        
        pages = extract_structured_content(mock_file)
        print(f"✅ Extracted {len(pages)} pages/chunks")
        
        total_comments = 0
        for i, page in enumerate(pages):
            print(f"\n📄 Page/Chunk {i+1}:")
            print(f"   Page number: {page.get('page_number')}")
            print(f"   Content type: {page.get('content_type')}")
            print(f"   Content length: {len(page.get('content', ''))}")
            
            comments = page.get('comments', [])
            print(f"   Comments found: {len(comments)}")
            
            if comments:
                total_comments += len(comments)
                print(f"   📝 COMMENTS:")
                for j, comment in enumerate(comments):
                    if isinstance(comment, dict):
                        print(f"      Comment {j+1}:")
                        print(f"         ID: {comment.get('id', 'N/A')}")
                        print(f"         Text: {comment.get('text', 'No text')[:100]}...")
                        print(f"         Author: {comment.get('author', 'Unknown')}")
                        print(f"         Initials: {comment.get('initials', 'N/A')}")
                        print(f"         Date: {comment.get('date', 'N/A')}")
                    else:
                        print(f"      Comment {j+1}: {comment}")
            else:
                print(f"   ❌ No comments found")
        
        print(f"\n📊 TOTAL COMMENTS EXTRACTED: {total_comments}")
        
        if total_comments == 0:
            print(f"\n❓ No comments found. This could mean:")
            print(f"   1. The file doesn't have Word comments/annotations")
            print(f"   2. The comments were deleted or resolved")
            print(f"   3. The file format doesn't support comments")
            print(f"   4. There's an issue with comment extraction")
        
    except Exception as e:
        print(f"❌ Error processing file: {e}")

def test_search_with_comments():
    """Test if search returns comments properly"""
    print(f"\n=== TESTING SEARCH WITH COMMENTS ===\n")
    
    try:
        DATABASE_URL = os.getenv("DATABASE_URL")
        vs = PostgresVectorStore(model_name="all-MiniLM-L6-v2", db_url=DATABASE_URL)
        
        # Test search
        results = vs.search("system requirements", top_k=3, min_score=0.1)
        print(f"Search results: {len(results)}")
        
        for i, result in enumerate(results):
            print(f"\n🔍 Result {i+1}:")
            print(f"   Score: {result['score']:.3f}")
            print(f"   Text: {result['text'][:100]}...")
            print(f"   Source: {result['source_file']}")
            print(f"   Metadata: {result['metadata']}")
            
            comments = result['metadata'].get('comments', [])
            print(f"   Comments in metadata: {len(comments)}")
            if comments:
                print(f"   📝 Comments: {json.dumps(comments, indent=6)}")
            else:
                print(f"   ❌ No comments in search result")
        
    except Exception as e:
        print(f"❌ Error testing search: {e}")

if __name__ == "__main__":
    print("🔍 COMPREHENSIVE COMMENT DEBUGGING\n")
    
    # Step 1: Check database
    debug_database_comments()
    
    # Step 2: Test file extraction if files exist
    test_files = [
        "test_urs_for_comments.docx",
        "Novugen_URS IGL (1).docx",
        "URS Coating Machine Rev 1 - GLATT comments 03092025.docx"
    ]
    
    for test_file in test_files:
        if os.path.exists(test_file):
            debug_file_extraction(test_file)
    
    # Step 3: Test search
    test_search_with_comments()
    
    print(f"\n✅ Debugging complete!")
    print(f"\nℹ️  If no comments are found:")
    print(f"   1. Make sure your DOCX files have actual Word comments (Review → New Comment)")
    print(f"   2. Re-index files after clearing database")
    print(f"   3. Check if comments are in tables vs regular paragraphs")