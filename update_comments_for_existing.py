#!/usr/bin/env python3
"""
Update comments for existing requirements in the database.
This script will extract comments from the original documents and update the database records.
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from utils.extractors import extract_text_from_file
from utils.gemini_processor import GeminiProcessor
from utils.postgres_vectorstore_gemini import PostgresVectorStoreGemini

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def process_document_for_comments(file_path: str, gemini_processor: GeminiProcessor, vectorstore: PostgresVectorStoreGemini):
    """
    Extract comments from a document and update existing requirements in the database.
    
    Args:
        file_path: Path to the document file
        gemini_processor: Initialized Gemini processor
        vectorstore: Database connection
    """
    filename = os.path.basename(file_path)
    print(f"\n🎯 Processing document: {filename}")
    
    # Check if this document exists in the database
    try:
        docs = vectorstore.get_all_documents()
        matching_doc = None
        for doc in docs:
            if doc['filename'] == filename or filename in doc['filename'] or doc['filename'] in filename:
                matching_doc = doc
                break
        
        if not matching_doc:
            print(f"❌ No matching document found in database for: {filename}")
            return
            
        print(f"✅ Found matching document in database: {matching_doc['filename']} (ID: {matching_doc['id']})")
        print(f"   Document has {matching_doc['requirement_count']} requirements")
        
    except Exception as e:
        print(f"❌ Error checking database: {e}")
        return
    
    # Extract text from the document
    try:
        print("📄 Extracting text from document...")
        
        # Create a mock uploaded file object for the extractor
        class MockUploadedFile:
            def __init__(self, file_path):
                self.name = os.path.basename(file_path)
                self._file_path = file_path
                
            def read(self):
                with open(self._file_path, 'rb') as f:
                    return f.read()
                    
            def seek(self, pos):
                pass  # Mock implementation
        
        mock_file = MockUploadedFile(file_path)
        document_text = extract_text_from_file(mock_file, filename)
        
        if not document_text or len(document_text.strip()) < 100:
            print(f"❌ Could not extract meaningful text from {filename}")
            return
            
        print(f"✅ Extracted {len(document_text)} characters from document")
        
    except Exception as e:
        print(f"❌ Error extracting text: {e}")
        return
    
    # Extract comments using Gemini
    try:
        print("🤖 Extracting comments using Gemini AI...")
        comments_data = gemini_processor.extract_comments_and_responses(
            document_text, 
            filename
        )
        
        if not comments_data:
            print("⚠️ No comments found in document")
            return
            
        print(f"✅ Found {len(comments_data)} requirements with comments")
        
        # Display what was found
        for i, item in enumerate(comments_data, 1):
            print(f"\n{i}. Requirement: {item['requirement_text'][:100]}...")
            print(f"   Comments: {len(item.get('comments', []))}")
            for j, comment in enumerate(item.get('comments', []), 1):
                print(f"     {j}. [{comment.get('author', 'Unknown')}] {comment.get('comment_text', '')[:80]}...")
        
    except Exception as e:
        print(f"❌ Error extracting comments: {e}")
        return
    
    # Update requirements in database with comments
    try:
        print(f"\n💾 Updating database records...")
        
        # Get existing requirements for this document
        existing_reqs = vectorstore.search_requirements_by_document(matching_doc['filename'])
        print(f"Found {len(existing_reqs)} existing requirements in database")
        
        updated_count = 0
        
        for comment_item in comments_data:
            # Try to match this comment requirement with existing requirements
            requirement_text = comment_item['requirement_text'].strip()
            
            # Find best matching requirement in database
            best_match = None
            best_score = 0
            
            for existing_req in existing_reqs:
                existing_text = existing_req['requirement'].strip()
                
                # Simple matching - check if significant portion of text matches
                if len(requirement_text) > 50 and len(existing_text) > 50:
                    # Calculate overlap
                    req_words = set(requirement_text.lower().split())
                    existing_words = set(existing_text.lower().split())
                    
                    if len(req_words) > 0 and len(existing_words) > 0:
                        overlap = len(req_words.intersection(existing_words))
                        score = overlap / max(len(req_words), len(existing_words))
                        
                        if score > best_score and score > 0.3:  # 30% overlap threshold
                            best_match = existing_req
                            best_score = score
            
            if best_match:
                # Update the requirement with comments
                comments_json = json.dumps(comment_item['comments'])
                
                try:
                    vectorstore.update_requirement_comments(
                        best_match['id'], 
                        comments_json,
                        comment_item.get('page_reference', 'Unknown')
                    )
                    updated_count += 1
                    print(f"✅ Updated requirement {best_match['id']} with {len(comment_item['comments'])} comments")
                    
                except Exception as e:
                    print(f"❌ Error updating requirement {best_match['id']}: {e}")
            else:
                print(f"⚠️ No matching requirement found for: {requirement_text[:80]}...")
        
        print(f"\n🎉 Successfully updated {updated_count} requirements with comments")
        
    except Exception as e:
        print(f"❌ Error updating database: {e}")


def main():
    """Main function to process documents for comment extraction."""
    
    # Initialize Gemini processor
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ GEMINI_API_KEY environment variable not set")
        return
    
    try:
        gemini_processor = GeminiProcessor(api_key)
        print("✅ Gemini processor initialized")
    except Exception as e:
        print(f"❌ Error initializing Gemini processor: {e}")
        return
    
    # Initialize database connection
    try:
        vectorstore = PostgresVectorStoreGemini()
        print("✅ Database connection established")
    except Exception as e:
        print(f"❌ Error connecting to database: {e}")
        return
    
    # List of document files to process
    document_files = [
        "URS Coating Machine Rev 1 - GLATT comments 03092025.docx",
        "Novugen_URS IGL (1).docx", 
        "test_urs_for_comments.docx"
    ]
    
    processed_count = 0
    
    for filename in document_files:
        file_path = os.path.join(os.getcwd(), filename)
        
        if os.path.exists(file_path):
            try:
                process_document_for_comments(file_path, gemini_processor, vectorstore)
                processed_count += 1
            except Exception as e:
                print(f"❌ Error processing {filename}: {e}")
        else:
            print(f"⚠️ File not found: {filename}")
    
    print(f"\n🏁 Processing complete! Processed {processed_count} documents")


if __name__ == "__main__":
    main()