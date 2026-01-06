import psycopg2
import numpy as np

conn = psycopg2.connect(dbname='urs_gemini', user='postgres', password='Patil1234', host='localhost', port=5433)
cur = conn.cursor()

# Check what's actually in the database
cur.execute('SELECT COUNT(*) FROM requirements')
print(f"Total requirements: {cur.fetchone()[0]}\n")

# Check table schema
cur.execute("""
    SELECT column_name, data_type 
    FROM information_schema.columns 
    WHERE table_name = 'requirements'
    ORDER BY ordinal_position
""")
print("=== TABLE SCHEMA ===")
for col in cur.fetchall():
    print(f"{col[0]}: {col[1]}")
print()

# Sample data
cur.execute('SELECT * FROM requirements LIMIT 3')
rows = cur.fetchall()
print("=== SAMPLE ROWS ===")
for i, row in enumerate(rows, 1):
    print(f"{i}. {row}")

cur.close()
conn.close()
