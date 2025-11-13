#!/usr/bin/env python3
"""
Standalone Holistic Extraction Script for Gemini Branch

This script performs comprehensive holistic extraction using Gemini 2.5 Flash
and saves results directly to the dedicated urs_gemini PostgreSQL database.

Usage:
    python standalone_holistic_extraction_gemini.py <file_path> [options]
    
Example:
    python standalone_holistic_extraction_gemini.py "URS Coating Machine Rev 1.docx" --comments "Initial analysis"
"""

import sys
import os
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from utils.gemini_processor import GeminiProcessor
from utils.extractors import extract_text_from_docx
from utils.postgres_vectorstore_gemini import PostgresVectorStoreGemini
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def process_document_holistically(file_path: str, comments: str = None, matched_doc: str = None):
    """
    Process a document using holistic Gemini extraction and store in urs_gemini database
    
    Args:
        file_path: Path to the document to process
        comments: Optional comments to add to the requirements
        matched_doc: Optional matched document name
    """
    
    # Validate file exists
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found: {file_path}")
        return False
    
    filename = os.path.basename(file_path)
    print(f"🧠 Starting holistic extraction for: {filename}")
    
    try:
        # Step 1: Initialize Gemini database
        print("📊 Connecting to urs_gemini database...")
        vs_gemini = PostgresVectorStoreGemini()
        print("✅ Connected to urs_gemini database")
        
        # Step 2: Extract text from document
        print("📄 Extracting text from document...")
        full_text = extract_text_from_docx(file_path)
        
        if not full_text or len(full_text.strip()) < 100:
            print(f"❌ Failed to extract meaningful text from document")
            print(f"Extracted text length: {len(full_text) if full_text else 0}")
            return False
        
        print(f"✅ Extracted {len(full_text):,} characters from document")
        
        # Step 3: Initialize Gemini processor
        print("🧠 Initializing Gemini processor...")
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        
        if not gemini_api_key:
            print("❌ Error: GEMINI_API_KEY not found in environment variables")
            print("Please set your Gemini API key in the .env file")
            return False
        
        gemini = GeminiProcessor(gemini_api_key)
        print("✅ Gemini processor initialized")
        
        # Step 4: Perform holistic extraction
        print("🔍 Performing holistic requirement extraction...")
        holistic_requirements = gemini.extract_requirements_holistically(
            full_document_text=full_text,
            document_name=filename
        )
        
        if not holistic_requirements:
            print("❌ No requirements extracted through holistic analysis")
            return False
        
        print(f"✅ Extracted {len(holistic_requirements)} requirements!")
        
        # Step 5: Display analysis summary
        print("\n📊 Analysis Summary:")
        categories = {}
        priorities = {}
        
        for req in holistic_requirements:
            cat = req.get('category', 'unknown')
            pri = req.get('priority', 'unknown')
            categories[cat] = categories.get(cat, 0) + 1
            priorities[pri] = priorities.get(pri, 0) + 1
        
        print("\n📂 By Category:")
        for cat, count in sorted(categories.items()):
            print(f"  • {cat}: {count}")
        
        print("\n⚡ By Priority:")
        for pri, count in sorted(priorities.items()):
            print(f"  • {pri}: {count}")
        
        # Step 6: Store in database
        print(f"\n💾 Storing requirements in urs_gemini database...")
        
        # Extract just the requirement texts for database storage
        requirement_texts = [req['text'] for req in holistic_requirements]
        
        # Store in database
        success = vs_gemini.add_requirements(
            requirements=requirement_texts,
            document_name=filename,
            comments=comments,
            matched_document_name=matched_doc
        )
        
        if success:
            print(f"✅ Successfully stored {len(requirement_texts)} requirements in urs_gemini database!")
            
            # Display some sample requirements
            print(f"\n📝 Sample Requirements (first 3):")
            for i, req in enumerate(holistic_requirements[:3]):
                print(f"\n{i+1}. {req['text']}")
                print(f"   Category: {req.get('category', 'N/A')} | Priority: {req.get('priority', 'N/A')} | Confidence: {req.get('confidence', 'N/A')}")
            
            if len(holistic_requirements) > 3:
                print(f"\n... and {len(holistic_requirements) - 3} more requirements")
            
            # Show database stats
            stats = vs_gemini.get_stats()
            print(f"\n📊 Database Statistics:")
            print(f"  Total requirements: {stats.get('total_requirements', 0)}")
            print(f"  Documents processed: {len(stats.get('documents', {}))}")
            
            return True
        else:
            print("❌ Failed to store requirements in database")
            return False
            
    except Exception as e:
        print(f"❌ Error during holistic analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function to handle command line arguments and process document"""
    
    parser = argparse.ArgumentParser(
        description="Holistic requirement extraction using Gemini 2.5 Flash for urs_gemini database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python standalone_holistic_extraction_gemini.py "document.docx"
    python standalone_holistic_extraction_gemini.py "document.docx" --comments "Initial analysis"
    python standalone_holistic_extraction_gemini.py "document.docx" --matched "reference.pdf" --comments "Comparison analysis"
        """
    )
    
    parser.add_argument(
        "file_path",
        help="Path to the document to process"
    )
    
    parser.add_argument(
        "--comments",
        help="Optional comments to add to the extracted requirements",
        default=None
    )
    
    parser.add_argument(
        "--matched",
        help="Optional matched document name for reference",
        default=None
    )
    
    parser.add_argument(
        "--clear-first",
        action="store_true",
        help="Clear database before processing (use with caution)"
    )
    
    args = parser.parse_args()
    
    # Check if we should clear database first
    if args.clear_first:
        print("🗑️ Clearing urs_gemini database...")
        vs_gemini = PostgresVectorStoreGemini()
        if vs_gemini.clear_database():
            print("✅ Database cleared")
        else:
            print("❌ Failed to clear database")
            return
    
    # Process the document
    success = process_document_holistically(
        file_path=args.file_path,
        comments=args.comments,
        matched_doc=args.matched
    )
    
    if success:
        print(f"\n🎉 Successfully processed {os.path.basename(args.file_path)}!")
        print("You can now use the Streamlit app to search and analyze the extracted requirements.")
    else:
        print(f"\n❌ Failed to process {os.path.basename(args.file_path)}")
        sys.exit(1)

if __name__ == "__main__":
    main()