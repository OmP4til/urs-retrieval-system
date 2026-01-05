"""Test database save directly"""
import sys
sys.path.insert(0, 'c:\\vv')

from utils.postgres_vectorstore_gemini import PostgresVectorStoreGemini

# Initialize vectorstore
vectorstore = PostgresVectorStoreGemini()

# Test data
test_pairs = [
    {
        'requirement': {
            'text': 'Test requirement 1',
            'category': 'functional',
            'priority': 'high',
            'confidence': 0.9,
            'source_context': 'Test context'
        },
        'comments': [
            {
                'text': 'Test comment',
                'author': 'Test Author',
                'comment_type': 'docx_structured'
            }
        ]
    }
]

print("Testing database save...")
print(f"Number of test pairs: {len(test_pairs)}")

success = vectorstore.add_requirements_with_individual_comments(
    requirement_comment_pairs=test_pairs,
    document_name="TEST_DOCUMENT.docx"
)

print(f"Result: {success}")

if success:
    print("✅ Save successful!")
else:
    print("❌ Save failed!")
