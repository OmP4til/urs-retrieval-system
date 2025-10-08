#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'utils'))

from postgres_vectorstore import PostgresVectorStore

def test_comments_display():
    """Test that comments are properly formatted for display"""
    
    vs = PostgresVectorStore()
    
    # Test search to see if comments are properly returned
    results = vs.search("GLATT", top_k=3)
    
    print("Testing comment formatting:")
    print("="*50)
    
    for i, result in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"Content: {result['text'][:60]}...")
        print(f"Score: {result['score']}")
        
        comments = result['metadata'].get('comments', [])
        print(f"Comments count: {len(comments)}")
        
        if comments:
            # Test the formatting logic from main.py
            comment_strings = []
            for comment in comments[:3]:  # Show max 3 comments
                if isinstance(comment, dict):
                    author = comment.get("author", "Unknown")
                    text = comment.get("text", "")[:100]  # Truncate long comments
                    comment_strings.append(f"{author}: {text}...")
                else:
                    comment_strings.append(str(comment)[:100])
            
            formatted_comments = "; ".join(comment_strings)
            if len(comments) > 3:
                formatted_comments += f" (+{len(comments)-3} more)"
                
            print(f"Formatted comments: {formatted_comments[:200]}...")
            
            # Check for the 21 CFR comment specifically
            cfr_found = False
            for comment in comments:
                if isinstance(comment, dict) and "21 CFR" in comment.get("text", ""):
                    cfr_found = True
                    print(f"✅ Found 21 CFR comment: {comment['author']}: {comment['text'][:100]}...")
                    break
            
            if not cfr_found:
                print("❌ 21 CFR comment not found in this result")
        else:
            print("No comments found")

if __name__ == "__main__":
    test_comments_display()