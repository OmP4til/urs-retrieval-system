import psycopg2

conn = psycopg2.connect(
    host='localhost',
    port=5433,
    database='urs_gemini',
    user='postgres',
    password='Patil1234'
)
cur = conn.cursor()

cur.execute('SELECT COUNT(*) FROM requirements')
count = cur.fetchone()[0]
print(f'📊 Total records in PostgreSQL: {count}')

if count > 0:
    cur.execute('SELECT requirement, document_name FROM requirements LIMIT 5')
    rows = cur.fetchall()
    print('\n📝 Sample records:')
    for i, row in enumerate(rows, 1):
        print(f'  {i}. {row[0][:70]}... (from: {row[1]})')
else:
    print('\n⚠️ Database is empty! You need to upload historical documents first.')

conn.close()
