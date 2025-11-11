#!/usr/bin/env python3
"""Check learning queue status breakdown"""
import sys
import importlib.util

spec = importlib.util.spec_from_file_location('db', 'advanced-memory/db.py')
db_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(db_module)

db = db_module.AllieMemoryDB()
cursor = db.connection.cursor(dictionary=True)
cursor.execute("SELECT status, COUNT(*) as count FROM learning_queue GROUP BY status")
results = cursor.fetchall()
cursor.close()

print('Learning Queue Status Breakdown:')
for r in results:
    print(f'  {r["status"]}: {r["count"]}')

# Check recent processed items
cursor = db.connection.cursor(dictionary=True)
cursor.execute("SELECT * FROM learning_queue WHERE status != 'pending' ORDER BY processed_at DESC LIMIT 5")
processed = cursor.fetchall()
cursor.close()

print(f'\nRecently Processed Items ({len(processed)}):')
for item in processed:
    print(f'  ID {item["id"]}: {item["keyword"]} - Status: {item["status"]} - Processed: {item["processed_at"]}')