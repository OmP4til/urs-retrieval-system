#!/usr/bin/env python3
"""
Test script to check if comments are being extracted from DOCX files
"""

import os
from dotenv import load_dotenv
from utils.extractors import extract_structured_content
from utils.postgres_vectorstore import PostgresVectorStore

# Load environment variables
load_dotenv()

def test_comment_extraction():
    print("Testing comment extraction from DOCX files...")
    
    # Test with a sample file (you can upload any DOCX with comments)
    test_files = [
        "test.docx",  # Replace with actual file name
        "sample.docx",
        "URS Coating Machine Rev 1 - GLATT comments 03092025.docx"
    ]
    
    for filename in test_files:
        if os.path.exists(filename):
            print(f"\n=== Testing {filename} ===")
            
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
                pages = extract_structured_content(mock_file)
                
                print(f"Extracted {len(pages)} pages/chunks")
                
                total_comments = 0
                for i, page in enumerate(pages):
                    comments = page.get('comments', [])
                    if comments:
                        print(f"\nPage {i+1} has {len(comments)} comments:")
                        for j, comment in enumerate(comments):
                            if isinstance(comment, dict):
                                print(f"  Comment {j+1}:")
                                print(f"    Text: {comment.get('text', 'No text')[:100]}...")
                                print(f"    Author: {comment.get('author', 'Unknown')}")
                                print(f"    Initials: {comment.get('initials', 'N/A')}")
                                print(f"    Date: {comment.get('date', 'N/A')}")
                            else:
                                print(f"  Comment {j+1}: {comment}")
                        total_comments += len(comments)
                
                print(f"\nTotal comments found: {total_comments}")
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
        else:
            print(f"File {filename} not found")

def test_database_comments():
    print("\n=== Testing database comment storage ===")
    
    try:
        DATABASE_URL = os.getenv("DATABASE_URL")
        vs = PostgresVectorStore(model_name="all-MiniLM-L6-v2", db_url=DATABASE_URL)
        
        # Get all documents to see their comments
        docs = vs.get_all_documents()
        print(f"Found {len(docs)} documents in database")
        
        # Check some requirements for comments
        from utils.db_models import get_session, Requirement
        session = get_session(vs.engine)
        try:
            # Check all requirements and see their comments
            all_reqs = session.query(Requirement).limit(10).all()
            
            print(f"Checking first 10 requirements for comments...")
            
            for req in all_reqs:
                print(f"\nRequirement: {req.text[:50]}...")
                comments = req.comments
                print(f"Comments type: {type(comments)}")
                print(f"Comments: {comments}")
                if comments and len(comments) > 0:
                    print("  ^ This requirement has comments!")
                
        finally:
            session.close()
            
    except Exception as e:
        print(f"Error checking database: {e}")

if __name__ == "__main__":
    test_comment_extraction()
    test_database_comments()