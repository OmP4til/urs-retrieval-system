#!/usr/bin/env python3
"""
Comment Extraction Script

This script extracts comments and responses from marked-up documents
using Gemini AI and stores them in the database.

Usage:
    python extract_comments.py <file_path>
    
Example:
    python extract_comments.py "URS Coating Machine Rev 1 - GLATT comments 03092025.docx"
"""

import sys
import os
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from utils.gemini_processor import GeminiProcessor
from utils.extractors import extract_text_from_file
from utils.postgres_vectorstore import PostgresVectorStore
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

def extract_comments_from_document(file_path: str):
    """
    Extract comments and responses from a marked-up document
    
    Args:
        file_path: Path to the document to process for comments
    """
    print(f"🎯 Starting comment extraction for: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found: {file_path}")
        return False
    
    try:
        # Initialize components
        print("📊 Connecting to database...")
        vectorstore = PostgresVectorStore()
        print("✅ Connected to database successfully")
        
        print("📄 Extracting text from document...")
        filename = os.path.basename(file_path)
        
        # Read the file and extract text
        with open(file_path, 'rb') as f:
            full_text = extract_text_from_file(f, filename)
        
        if not full_text:
            print("❌ Error: Could not extract text from document")
            return False
            
        print(f"✅ Extracted {len(full_text)} characters from document")
        
        print("🧠 Initializing Gemini processor...")
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            print("❌ Error: GEMINI_API_KEY not found in environment variables")
            return False
        gemini = GeminiProcessor(api_key)
        print("✅ Gemini processor initialized")
        
        print("💬 Extracting comments and responses using Gemini...")
        comments_data = gemini.extract_comments_and_responses(full_text, file_path)
        
        if not comments_data:
            print("⚠️ No comments found in the document")
            return True
            
        print(f"✅ Extracted {len(comments_data)} commented requirements!")
        
        # Display extracted comments
        print("\n📝 Extracted Comments:")
        for i, item in enumerate(comments_data[:3], 1):  # Show first 3
            print(f"\n{i}. Requirement: {item.get('requirement_text', '')[:80]}...")
            comments = item.get('comments', [])
            print(f"   Comments ({len(comments)}):")
            for comment in comments[:2]:  # Show first 2 comments
                author = comment.get('author', 'Unknown')
                text = comment.get('comment_text', '')
                print(f"     - {author}: {text[:60]}...")
        
        if len(comments_data) > 3:
            print(f"\n... and {len(comments_data) - 3} more commented requirements")
        
        # Store in database
        print("\n💾 Storing comments in database...")
        filename = os.path.basename(file_path)
        
        saved_count = 0
        for item in comments_data:
            try:
                requirement_entry = {
                    'id': item.get('id', f"COMMENT_{saved_count+1}"),
                    'text': item.get('requirement_text', ''),
                    'page_number': item.get('page_reference', 'Unknown'),
                    'source_file': filename,
                    'confidence_score': 0.95,
                    'category': 'commented_requirement',
                    'priority': 'high',  # Requirements with comments are often important
                    'comments': [c.get('comment_text', '') for c in item.get('comments', [])],
                    'responses': item.get('comments', [])  # Store full comment structure
                }
                
                vectorstore.add_requirement(requirement_entry)
                saved_count += 1
                
            except Exception as e:
                print(f"⚠️ Failed to save requirement {saved_count + 1}: {str(e)[:100]}")
        
        vectorstore.save()
        print(f"✅ Successfully saved {saved_count} commented requirements to database!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during comment extraction: {str(e)}")
        return False

def main():
    """Main function to handle command line arguments"""
    parser = argparse.ArgumentParser(description="Extract comments from documents using Gemini AI")
    parser.add_argument("file_path", help="Path to the document file")
    
    args = parser.parse_args()
    
    success = extract_comments_from_document(args.file_path)
    
    if success:
        print("\n🎉 Comment extraction completed successfully!")
        print("You can now use the Streamlit app to view the extracted comments.")
    else:
        print("\n❌ Comment extraction failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()