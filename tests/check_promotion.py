#!/usr/bin/env python3
"""Check processed queue items and their confidence scores"""
import sys
import importlib.util

spec = importlib.util.spec_from_file_location('db', 'advanced-memory/db.py')
db_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(db_module)

db = db_module.AllieMemoryDB()

# Check all processed items
cursor = db.connection.cursor(dictionary=True)
cursor.execute("SELECT id, keyword, fact, confidence, source, status FROM learning_queue WHERE status = 'processed' ORDER BY confidence DESC")
processed = cursor.fetchall()
cursor.close()

print(f'Processed Queue Items ({len(processed)}):')
high_conf = 0
for item in processed:
    conf = item['confidence']
    if conf >= 0.8:
        high_conf += 1
        print(f'  ✅ HIGH CONF (should be in KB): ID {item["id"]} - {item["keyword"]} - Conf: {conf}')
    else:
        print(f'  📋 LOW CONF: ID {item["id"]} - {item["keyword"]} - Conf: {conf}')

print(f'\nSummary: {high_conf} items had confidence >= 0.8 and should have been promoted to KB')

# Check if any of these keywords exist in KB
if high_conf > 0:
    print('\nChecking if high-confidence items made it to KB:')
    cursor = db.connection.cursor(dictionary=True)
    cursor.execute("SELECT keyword FROM learning_queue WHERE status = 'processed' AND confidence >= 0.8")
    high_conf_keywords = [row['keyword'] for row in cursor.fetchall()]
    cursor.close()

    for keyword in high_conf_keywords:
        kb_fact = db.get_kb_fact(keyword)
        if kb_fact:
            print(f'  ✅ {keyword}: Found in KB (ID: {kb_fact["id"]})')
        else:
            print(f'  ❌ {keyword}: NOT found in KB (should have been promoted!)')