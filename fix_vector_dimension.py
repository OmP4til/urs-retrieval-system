"""
Fix PostgreSQL table to support 768-dimensional embeddings (all-mpnet-base-v2)
"""

import psycopg2

def fix_vector_dimension():
    """Update the requirements table to support 768-dimensional vectors"""
    try:
        conn = psycopg2.connect(
            host='localhost',
            port=5433,
            database='urs_gemini',
            user='postgres',
            password='Patil1234'
        )
        cur = conn.cursor()
        
        print("🔧 Checking current table structure...")
        
        # Check if table exists and get column info
        cur.execute("""
            SELECT column_name, data_type, udt_name 
            FROM information_schema.columns 
            WHERE table_name = 'requirements' AND column_name = 'embedding'
        """)
        
        result = cur.fetchone()
        if result:
            print(f"Current embedding column: {result}")
        
        # Drop and recreate the table with correct dimensions
        print("\n🔄 Recreating table with 768-dimensional vectors...")
        
        cur.execute("DROP TABLE IF EXISTS requirements CASCADE")
        
        cur.execute("""
            CREATE TABLE requirements (
                id SERIAL PRIMARY KEY,
                requirement TEXT NOT NULL,
                embedding vector(768),
                metadata JSONB,
                document_name TEXT,
                comments TEXT,
                matched_document_name TEXT,
                extraction_type VARCHAR(50),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create index for vector similarity search
        cur.execute("""
            CREATE INDEX ON requirements 
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100)
        """)
        
        conn.commit()
        
        print("✅ Table recreated successfully with 768-dimensional vectors!")
        print("✅ Created IVFFlat index for fast similarity search")
        
        cur.close()
        conn.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("=" * 60)
    print("FIX VECTOR DIMENSION FOR all-mpnet-base-v2")
    print("=" * 60)
    print("\nThis will recreate the requirements table with:")
    print("  - 768-dimensional vectors (all-mpnet-base-v2)")
    print("  - IVFFlat index for fast similarity search")
    print("=" * 60)
    
    fix_vector_dimension()
