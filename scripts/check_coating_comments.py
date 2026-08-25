"""
Check how comments are stored for the URS Coating Machine document
"""
import psycopg2
import json
from utils.postgres_vectorstore_gemini import get_db_password

conn = psycopg2.connect(
    host='localhost',
    port=5433,
    database='urs_gemini',
    user='postgres',
    password=get_db_password()
)
cur = conn.cursor()

# Get requirements from the Coating Machine document
cur.execute("""
    SELECT requirement, comments, metadata 
    FROM requirements 
    WHERE document_name LIKE '%Coating%GLATT%'
    LIMIT 10
""")

rows = cur.fetchall()

print(f"Found {len(rows)} requirements from Coating Machine document\n")
print("=" * 100)

for i, row in enumerate(rows, 1):
    req_text = row[0]
    comments_str = row[1]
    metadata = row[2]
    
    print(f"\n{i}. REQUIREMENT:")
    print(f"   {req_text[:150]}...")
    
    print(f"\n   METADATA:")
    if metadata:
        print(f"   Category: {metadata.get('category', 'N/A')}")
        print(f"   Source: {metadata.get('source', 'N/A')}")
    
    print(f"\n   COMMENTS:")
    if comments_str:
        try:
            # Try to parse as JSON first
            comments_data = json.loads(comments_str)
            if isinstance(comments_data, dict):
                comments_list = comments_data.get('comments', [])
                print(f"   Total: {len(comments_list)} comments")
                for j, comment in enumerate(comments_list[:3], 1):
                    print(f"\n   Comment {j}:")
                    print(f"     Author: {comment.get('author', 'Unknown')}")
                    print(f"     Text: {comment.get('text', comment.get('comment_text', 'N/A'))[:100]}...")
                    print(f"     Type: {comment.get('comment_type', 'N/A')}")
            else:
                print(f"   {comments_str[:200]}...")
        except:
            # Not JSON, just show raw
            print(f"   {comments_str[:200]}...")
    else:
        print("   No comments stored")
    
    print("\n" + "-" * 100)

conn.close()
