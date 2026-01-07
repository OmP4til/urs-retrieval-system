"""
Clear old embeddings that were created with the wrong model.
These 332 records were created with all-mpnet-base-v2 (768d) but the new
model is intfloat/e5-large-v2 (1024d), so they're incompatible.
"""
import psycopg2

conn = psycopg2.connect(
    host='localhost',
    port=5433,
    database='urs_gemini',
    user='postgres',
    password='Patil1234'
)
cur = conn.cursor()

# Check current count
cur.execute('SELECT COUNT(*) FROM requirements')
count = cur.fetchone()[0]
print(f'📊 Current records: {count}')

if count > 0:
    # Show what we're deleting
    cur.execute('SELECT DISTINCT document_name FROM requirements')
    docs = cur.fetchall()
    print(f'\n📄 Documents to be cleared:')
    for doc in docs:
        print(f'   - {doc[0]}')
    
    # Delete all
    print(f'\n🗑️ Deleting {count} old records (encoded with wrong model)...')
    cur.execute('DELETE FROM requirements')
    conn.commit()
    print('✅ Database cleared!')
    print('\n⚠️ Next step: Re-upload your historical documents')
    print('   They will be encoded with the new E5 model for better matching')
else:
    print('Database is already empty')

conn.close()
