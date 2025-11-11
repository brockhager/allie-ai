import mysql.connector
import sys

try:
    connection = mysql.connector.connect(
        host='localhost',
        user='allie',
        password='StrongPassword123!',
        database='allie_memory'
    )
    cursor = connection.cursor()

    # Check learning_queue table
    cursor.execute('SELECT COUNT(*) FROM learning_queue WHERE status="pending"')
    pending_count = cursor.fetchone()[0]

    cursor.execute('SELECT COUNT(*) FROM learning_queue WHERE status="processed"')
    processed_count = cursor.fetchone()[0]

    cursor.execute('SELECT * FROM learning_queue LIMIT 5')
    rows = cursor.fetchall()

    print(f'Pending items: {pending_count}')
    print(f'Processed items: {processed_count}')
    print(f'Total items: {pending_count + processed_count}')
    print()
    print('Sample items:')
    for row in rows:
        print(f'ID: {row[0]}, Keyword: {row[1]}, Fact: {row[2][:50]}..., Status: {row[3]}, Created: {row[4]}')

    cursor.close()
    connection.close()

except Exception as e:
    print(f'Error: {e}')