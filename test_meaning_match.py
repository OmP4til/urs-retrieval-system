from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-mpnet-base-v2')

# Test pairs from user's examples
test_pairs = [
    (
        "The tablet coating machine shall meet all cGMP, safety, and user requirement specifications.",
        "The equipment shall be compatible with Oil Free, dust free, Pharma grade clean air of 0.3 micron or better filtered at 6 kg/cm2."
    ),
    (
        "The tablet loading system shall utilize a Closed IBC bucket.",
        "A stainless-steel chute shall be provided to transfer tablets from the IBC bin to the Coater Pan via the Auto Discharge station."
    ),
    (
        "The HMI shall display outlet temperature.",
        "The system shall allow for control and monitoring of Product temperature."
    ),
    (
        "In the event of a power failure, the system shall protect the product against damage.",
        "Supplier shall ensure that systems and data backups are continuously maintained during the system validation, for effective restoration if required."
    ),
]

print("=" * 80)
print("TESTING SEMANTIC SIMILARITY SCORES")
print("=" * 80)

for i, (req1, req2) in enumerate(test_pairs, 1):
    # Get embeddings
    emb1 = model.encode([req1])[0]
    emb2 = model.encode([req2])[0]
    
    # Calculate cosine distance (same as pgvector)
    dot_product = np.dot(emb1, emb2)
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    cosine_similarity = dot_product / (norm1 * norm2)
    
    # Convert to distance (pgvector uses: distance = 1 - cosine_similarity for cosine distance)
    cosine_distance = 1.0 - cosine_similarity
    
    # Apply the formula from your code
    similarity_score = 1.0 - (cosine_distance / 2.0)
    
    print(f"\n{'='*80}")
    print(f"Pair {i}:")
    print(f"Req 1: {req1[:80]}...")
    print(f"Req 2: {req2[:80]}...")
    print(f"Cosine Similarity (raw): {cosine_similarity:.4f}")
    print(f"Cosine Distance: {cosine_distance:.4f}")
    print(f"Similarity Score (your formula): {similarity_score:.4f}")
    print(f"Would match at 0.75 threshold: {'YES ❌' if similarity_score >= 0.75 else 'NO ✓'}")
    print(f"Would match at 0.85 threshold: {'YES ❌' if similarity_score >= 0.85 else 'NO ✓'}")
    print(f"Would match at 0.90 threshold: {'YES ❌' if similarity_score >= 0.90 else 'NO ✓'}")
