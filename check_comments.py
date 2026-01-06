import psycopg2
import json

# Connect to database
conn = psycopg2.connect(dbname='urs_gemini', user='postgres', password='Patil1234', host='localhost', port=5433)
cur = conn.cursor()

# Count total requirements
cur.execute("SELECT COUNT(*) FROM requirements")
total_count = cur.fetchone()[0]

# Count requirements with non-null comments
cur.execute("SELECT COUNT(*) FROM requirements WHERE comments IS NOT NULL AND comments != ''")
has_comments_count = cur.fetchone()[0]

# Count requirements with actual comment content (not just empty JSON)
cur.execute("""
    SELECT COUNT(*) 
    FROM requirements 
    WHERE comments IS NOT NULL 
    AND comments != '' 
    AND comments::jsonb->'comments' IS NOT NULL
    AND jsonb_array_length(comments::jsonb->'comments') > 0
""")
has_real_comments_count = cur.fetchone()[0]

print("=" * 80)
print("COMMENTS ANALYSIS")
print("=" * 80)
print(f"\nTotal requirements: {total_count}")
print(f"Requirements with comments field: {has_comments_count}")
print(f"Requirements with actual comment text: {has_real_comments_count}")
print(f"Percentage with responses: {(has_real_comments_count/total_count)*100:.1f}%")

# Show some examples with comments
print("\n" + "=" * 80)
print("REQUIREMENTS WITH COMMENTS (Sample)")
print("=" * 80)
cur.execute("""
    SELECT requirement, comments, document_name
    FROM requirements 
    WHERE comments IS NOT NULL 
    AND comments != '' 
    AND comments::jsonb->'comments' IS NOT NULL
    AND jsonb_array_length(comments::jsonb->'comments') > 0
    LIMIT 5
""")

results = cur.fetchall()
for i, row in enumerate(results, 1):
    req = row[0]
    comments_str = row[1]
    doc = row[2]
    
    try:
        comments_data = json.loads(comments_str) if isinstance(comments_str, str) else comments_str
        comment_list = comments_data.get('comments', [])
        
        print(f"\n{i}. Requirement: {req[:100]}...")
        print(f"   Document: {doc}")
        print(f"   Number of comments: {len(comment_list)}")
        for j, comment in enumerate(comment_list[:2], 1):
            author = comment.get('author', 'Unknown')
            text = comment.get('text', '')[:150]
            print(f"   Comment {j} [{author}]: {text}...")
    except Exception as e:
        print(f"   Error parsing: {e}")

cur.close()
conn.close()
