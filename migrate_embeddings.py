"""
Migration script to clear old embeddings after model upgrade
Run this once when switching from all-MiniLM-L6-v2 (384d) to all-mpnet-base-v2 (768d)
"""

import psycopg2

def clear_old_embeddings():
    """Clear old embeddings from PostgreSQL database"""
    try:
        conn = psycopg2.connect(
            host='localhost',
            port=5433,
            database='urs_gemini',
            user='postgres',
            password='Patil1234'
        )
        cur = conn.cursor()
        
        # Check current count
        cur.execute("SELECT COUNT(*) FROM requirements")
        count = cur.fetchone()[0]
        print(f"📊 Current database has {count} requirements with old embeddings (384d)")
        
        if count > 0:
            response = input(f"\n⚠️  This will DELETE all {count} requirements from the database.\nAre you sure? (yes/no): ")
            if response.lower() == 'yes':
                cur.execute("DELETE FROM requirements")
                conn.commit()
                print(f"✅ Cleared {count} old requirements from database")
                print("🔄 You can now upload documents and they will be stored with new embeddings (768d)")
            else:
                print("❌ Migration cancelled")
        else:
            print("✅ Database is already empty")
        
        cur.close()
        conn.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("EMBEDDING MODEL MIGRATION")
    print("=" * 60)
    print("\nThis script will clear old embeddings after upgrading from:")
    print("  OLD: all-MiniLM-L6-v2 (384 dimensions)")
    print("  NEW: all-mpnet-base-v2 (768 dimensions)")
    print("\nOld embeddings are incompatible with the new model.")
    print("=" * 60)
    
    clear_old_embeddings()
