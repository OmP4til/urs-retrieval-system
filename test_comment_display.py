#!/usr/bin/env python3

"""
Quick test to check how comments are stored and displayed
"""

from utils.postgres_vectorstore_gemini import PostgresVectorStoreGemini
import json

def test_comment_display():
    try:
        vs = PostgresVectorStoreGemini()
        
        # Get all documents
        docs = vs.get_all_documents()
        print(f"Found {len(docs)} documents:")
        for doc in docs:
            print(f"  - {doc['filename']}: {doc['requirement_count']} requirements")
        
        # Test the first document with comments
        if docs:
            test_doc = docs[0]['filename']
            print(f"\nTesting comment display for: {test_doc}")
            requirements = vs.search_requirements_by_document(test_doc)
            
            # Find requirements with comments
            reqs_with_comments = [req for req in requirements if req.get('comments')]
            print(f"Found {len(reqs_with_comments)} requirements with comments")
            
            if reqs_with_comments:
                print("\nSample comment data structures:")
                for i, req in enumerate(reqs_with_comments[:3], 1):  # Show first 3
                    print(f"\n--- Requirement {i} ---")
                    print(f"Requirement text: {req.get('requirement', '')[:100]}...")
                    
                    comments_data = req.get('comments')
                    print(f"Comments data type: {type(comments_data)}")
                    print(f"Comments data (first 200 chars): {str(comments_data)[:200]}...")
                    
                    # Try to parse comments
                    if isinstance(comments_data, str):
                        try:
                            parsed = json.loads(comments_data)
                            print(f"Successfully parsed JSON: {type(parsed)}")
                            if isinstance(parsed, dict) and 'comments' in parsed:
                                print(f"Found {len(parsed['comments'])} structured comments")
                                for j, comment in enumerate(parsed['comments'][:2], 1):
                                    comment_text = comment.get('comment_text', 'No text')[:100]
                                    author = comment.get('author', 'Unknown')
                                    print(f"  Comment {j}: {comment_text}... (by {author})")
                        except json.JSONDecodeError:
                            print("Failed to parse as JSON")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_comment_display()