
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

def test_db_connection():
    """Test PostgreSQL connection and pgvector extension."""
    
    # Load environment variables
    load_dotenv()
    
    # Get database URL
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("❌ ERROR: DATABASE_URL not found in .env file")
        return False
    
    print(f"🔍 Testing connection to: {database_url.split('@')[1] if '@' in database_url else 'database'}")
    
    try:
        # Create engine
        engine = create_engine(database_url, pool_pre_ping=True)
        
        # Test basic connection
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version()")).scalar()
            print(f"✅ Successfully connected to PostgreSQL!")
            print(f"   Version: {result.split(',')[0]}")
            
            # Test vector extension
            result = conn.execute(
                text("SELECT extname, extversion FROM pg_extension WHERE extname = 'vector'")
            ).fetchone()
            
            if result:
                print(f"✅ pgvector extension is installed!")
                print(f"   Version: {result[1]}")
            else:
                print("❌ pgvector extension is NOT installed!")
                print("   Run this SQL command:")
                print("   CREATE EXTENSION vector;")
                return False
            
            # Check if tables exist
            tables_query = text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name IN ('documents', 'requirements')
            """)
            tables = conn.execute(tables_query).fetchall()
            
            if tables:
                print(f"✅ Found {len(tables)} table(s):")
                for table in tables:
                    print(f"   - {table[0]}")
                    
                # Count documents and requirements
                doc_count = conn.execute(text("SELECT COUNT(*) FROM documents")).scalar()
                req_count = conn.execute(text("SELECT COUNT(*) FROM requirements")).scalar()
                print(f"\n📊 Database Statistics:")
                print(f"   Documents: {doc_count}")
                print(f"   Requirements: {req_count}")
            else:
                print("ℹ️  No tables found. They will be created on first run.")
        
        print("\n✅ All tests passed! Database is ready to use.")
        return True
        
    except Exception as e:
        print(f"\n❌ Database connection failed!")
        print(f"   Error: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Check if PostgreSQL is running")
        print("2. Verify DATABASE_URL in .env file")
        print("3. Ensure the database exists")
        print("4. Check username and password")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("PostgreSQL Database Connection Test")
    print("=" * 60)
    test_db_connection()
    print("=" * 60)