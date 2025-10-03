import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

def test_db_connection():
    # Load environment variables
    load_dotenv()
    
    # Get database URL
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("ERROR: DATABASE_URL not found in .env file")
        return False
    
    try:
        # Create engine
        engine = create_engine(database_url)
        
        # Test connection
        with engine.connect() as conn:
            # Test basic query
            result = conn.execute(text("SELECT 1")).scalar()
            print("✅ Successfully connected to database!")
            
            # Test vector extension
            result = conn.execute(text("SELECT extname FROM pg_extension WHERE extname = 'vector'")).scalar()
            if result == 'vector':
                print("✅ Vector extension is installed!")
            else:
                print("❌ Vector extension is not installed!")
                print("Run this SQL command: CREATE EXTENSION vector;")
        
        return True
        
    except Exception as e:
        print(f"❌ Database connection failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_db_connection()