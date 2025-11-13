#!/usr/bin/env python3

"""
Quick test to check database schema and fix the search method
"""

from utils.postgres_vectorstore_gemini import PostgresVectorStoreGemini

def test_database():
    try:
        vs = PostgresVectorStoreGemini()
        
        # Test get_all_documents
        print("Testing get_all_documents...")
        docs = vs.get_all_documents()
        print(f"Found {len(docs)} documents:")
        for doc in docs:
            print(f"  - {doc.get('filename', 'No filename')}: {doc.get('requirement_count', 0)} requirements")
        
        # Test search_requirements_by_document
        if docs:
            test_doc = docs[0]['filename']
            print(f"\nTesting search_requirements_by_document for: {test_doc}")
            requirements = vs.search_requirements_by_document(test_doc)
            print(f"Found {len(requirements)} requirements for {test_doc}")
            
            if requirements:
                print("Sample requirement keys:", list(requirements[0].keys()))
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_database()