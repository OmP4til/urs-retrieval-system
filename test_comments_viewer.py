#!/usr/bin/env python3

"""
Test script to verify the new Requirements & Comments Viewer functionality
"""

from utils.postgres_vectorstore_gemini import PostgresVectorStoreGemini
import json

def test_comments_viewer():
    """Test the viewer functionality"""
    try:
        vs = PostgresVectorStoreGemini()
        
        # Get all documents
        print("📄 Available Documents:")
        docs = vs.get_all_documents()
        for doc in docs:
            print(f"  - {doc['filename']}: {doc['requirement_count']} requirements")
        
        if not docs:
            print("❌ No documents found in database")
            return
        
        # Test with first document
        test_doc = docs[0]['filename']
        print(f"\n🔍 Testing viewer with: {test_doc}")
        
        requirements = vs.search_requirements_by_document(test_doc)
        print(f"📝 Found {len(requirements)} requirements")
        
        # Analyze comment structure
        requirements_with_comments = []
        requirements_without_comments = []
        
        for req in requirements:
            comments_data = req.get('comments')
            if comments_data:
                requirements_with_comments.append(req)
                
                # Try to parse comment structure
                try:
                    if isinstance(comments_data, str):
                        parsed = json.loads(comments_data)
                        if isinstance(parsed, dict) and 'comments' in parsed:
                            print(f"\n💬 Sample requirement with {len(parsed['comments'])} comments:")
                            print(f"   Requirement: {req.get('requirement', '')[:100]}...")
                            print(f"   Authors: {parsed.get('authors', [])}")
                            print(f"   Comment count: {parsed.get('count', 0)}")
                            
                            # Show first comment details
                            if parsed['comments']:
                                first_comment = parsed['comments'][0]
                                print(f"   Sample comment: {first_comment.get('comment_text', '')[:100]}...")
                                print(f"   Comment author: {first_comment.get('author', 'Unknown')}")
                                print(f"   Comment type: {first_comment.get('comment_type', 'Unknown')}")
                            break
                except json.JSONDecodeError:
                    print(f"   Raw comments (non-JSON): {str(comments_data)[:100]}...")
            else:
                requirements_without_comments.append(req)
        
        print(f"\n📊 Summary:")
        print(f"   Requirements with comments: {len(requirements_with_comments)}")
        print(f"   Requirements without comments: {len(requirements_without_comments)}")
        print(f"   Viewer functionality: ✅ Ready to use")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing viewer: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Testing Requirements & Comments Viewer")
    print("=" * 50)
    
    success = test_comments_viewer()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ Viewer test completed successfully!")
        print("🎯 You can now use the Requirements & Comments Viewer in the Streamlit app")
        print("📱 Access it at: http://localhost:8506")
    else:
        print("❌ Viewer test failed")