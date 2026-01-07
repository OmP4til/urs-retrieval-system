"""
Test E5 model search to diagnose matching issues
"""
import sys
sys.path.insert(0, 'c:\\vv')

from utils.postgres_vectorstore_gemini import PostgresVectorStoreGemini

# Initialize vectorstore
vectorstore = PostgresVectorStoreGemini()

# Test queries from your sample requirements
test_queries = [
    "The powder then flows by gravity into the mixer bowl.",
    "Inside the high shear mixer, the materials are blended and granulated with the addition of a binder solution.",
    "The Wet Mill shall control Speed.",
    "PQ will be performed and completed by Novugen Oncology."
]

print("=" * 80)
print("TESTING E5 MODEL SEARCH")
print("=" * 80)

for query in test_queries:
    print(f"\n🔍 Query: {query[:70]}...")
    print("-" * 80)
    
    # Search with different thresholds
    for threshold in [0.3, 0.5, 0.7, 0.85]:
        results = vectorstore.search_similar_requirements(
            query=query,
            top_k=3,
            threshold=threshold
        )
        
        print(f"\n  Threshold {threshold}:")
        if results:
            print(f"    ✅ Found {len(results)} matches")
            for i, result in enumerate(results, 1):
                print(f"       {i}. Score: {result['similarity_score']:.3f} - {result['requirement'][:60]}...")
        else:
            print(f"    ❌ No matches found")

print("\n" + "=" * 80)
