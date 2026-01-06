import psycopg2
import numpy as np
from sentence_transformers import SentenceTransformer

# Initialize the same model used in the app
model = SentenceTransformer('all-mpnet-base-v2')

# Connect to database
conn = psycopg2.connect(dbname='urs_gemini', user='postgres', password='Patil1234', host='localhost', port=5433)
cur = conn.cursor()

# Test with a real requirement
test_query = "On the HMI a recipe for the product is selected with validated parameters (e.g., pump speed, impeller speed, inlet air flow, and milling speed)."

print("=" * 80)
print("TESTING MATCHING QUALITY")
print("=" * 80)
print(f"\nQuery: {test_query}\n")

# Generate query embedding
query_embedding = model.encode([test_query])[0]
embedding_str = '[' + ','.join(map(str, query_embedding.tolist())) + ']'

# Search for matches
cur.execute("""
    SELECT 
        requirement, 
        document_name,
        comments,
        embedding <=> %s::vector as distance
    FROM requirements
    WHERE document_name IS NOT NULL
    ORDER BY embedding <=> %s::vector
    LIMIT 10
""", (embedding_str, embedding_str))

results = cur.fetchall()

print(f"Filtering results with threshold >= 0.75:\n")
count = 0
for i, row in enumerate(results, 1):
    req = row[0]
    doc = row[1]
    comments = row[2]
    distance = float(row[3])
    similarity = 1.0 - (distance / 2.0)
    
    # Apply threshold filter
    if similarity < 0.75:
        continue
    
    count += 1
    
    # Extract comment text if available
    comment_text = "No comments"
    if comments:
        try:
            import json
            comments_data = json.loads(comments) if isinstance(comments, str) else comments
            if isinstance(comments_data, dict) and 'comments' in comments_data:
                comment_list = comments_data['comments']
                if comment_list and len(comment_list) > 0:
                    comment_text = comment_list[0].get('text', 'No text')[:100]
        except:
            comment_text = str(comments)[:100]
    
    print(f"\n{count}. Similarity: {similarity:.3f}")
    print(f"   Document: {doc}")
    print(f"   Requirement: {req[:150]}...")
    print(f"   Response: {comment_text}...")

if count == 0:
    print("No matches found with similarity >= 0.75")

cur.close()
conn.close()
