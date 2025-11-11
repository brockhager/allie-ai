import mysql.connector
import json

# Connect to database
conn = mysql.connector.connect(
    host='localhost',
    user='root',
    password='NM#W%ud3jFo75JMBNM#W%ud',
    database='allie_memory'
)
cursor = conn.cursor(dictionary=True)

# Check learning_queue count
cursor.execute('SELECT COUNT(*) as count FROM learning_queue')
queue_count = cursor.fetchone()['count']

# Check facts table count
cursor.execute('SELECT COUNT(*) as count FROM facts')
facts_count = cursor.fetchone()['count']

# Check table structure
cursor.execute('DESCRIBE learning_queue')
queue_columns = cursor.fetchall()
print('=== LEARNING_QUEUE TABLE STRUCTURE ===')
for col in queue_columns:
    print(f'{col["Field"]}: {col["Type"]}')

print()
cursor.execute('DESCRIBE facts')
facts_columns = cursor.fetchall()
print('=== FACTS TABLE STRUCTURE ===')
for col in facts_columns:
    print(f'{col["Field"]}: {col["Type"]}')

print()
print('=== DATABASE STATE ===')
print(f'Facts table: {facts_count} items')
print(f'Learning queue: {queue_count} items')
print()

# Check recent facts
cursor.execute('SELECT id, keyword, fact, status, confidence, created_at FROM facts ORDER BY created_at DESC LIMIT 5')
recent_facts = cursor.fetchall()

print('=== RECENT FACTS ===')
for fact in recent_facts:
    print(f'ID: {fact["id"]}, Keyword: {fact["keyword"]}, Status: {fact["status"]}, Confidence: {fact["confidence"]}, Created: {fact["created_at"]}')
    print(f'  Fact: {fact["fact"][:100]}...')
print()

# Check learning_queue items (use correct column names)
cursor.execute('SELECT id, keyword, fact, source, status, created_at, processed_at FROM learning_queue ORDER BY created_at DESC LIMIT 10')
queue_items = cursor.fetchall()

print('=== LEARNING QUEUE ITEMS ===')
for item in queue_items:
    print(f'ID: {item["id"]}, Keyword: {item["keyword"]}, Source: {item["source"]}, Status: {item["status"]}, Created: {item["created_at"]}, Processed: {item["processed_at"]}')
    print(f'  Fact: {item["fact"][:100]}...')

# Check for processed items
cursor.execute('SELECT COUNT(*) as count FROM learning_queue WHERE status = "processed"')
processed_count = cursor.fetchone()['count']

# Check for pending items
cursor.execute('SELECT COUNT(*) as count FROM learning_queue WHERE status = "pending"')
pending_count = cursor.fetchone()['count']

print(f'\nProcessed items: {processed_count}')
print(f'Pending items: {pending_count}')

# Check for any reconciliation logs or telemetry (if tables exist)
try:
    cursor.execute('SHOW TABLES LIKE "learning_log"')
    if cursor.fetchone():
        cursor.execute('SELECT COUNT(*) as count FROM learning_log')
        log_count = cursor.fetchone()['count']
        print(f'Learning log entries: {log_count}')
        
        cursor.execute('SELECT action, created_at FROM learning_log ORDER BY created_at DESC LIMIT 3')
        recent_logs = cursor.fetchall()
        print('Recent learning log entries:')
        for log in recent_logs:
            print(f'  {log["created_at"]}: {log["action"]}')
except:
    print('No learning_log table found')

conn.close()