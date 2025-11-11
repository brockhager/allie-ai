#!/usr/bin/env python3
"""Quick script to check KB contents"""
import sys
import importlib.util

spec = importlib.util.spec_from_file_location('db', 'advanced-memory/db.py')
db_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(db_module)

db = db_module.AllieMemoryDB()
facts = db.get_all_kb_facts(limit=50)

print(f'Total KB facts found: {len(facts)}')
print('-' * 80)

for f in facts[:10]:
    fact_preview = f['fact'][:80] if len(f['fact']) > 80 else f['fact']
    print(f"ID {f['id']}: {f['keyword']}")
    print(f"  Fact: {fact_preview}")
    print(f"  Status: {f['status']} | Confidence: {f['confidence_score']} | Source: {f['source']}")
    print(f"  Updated: {f['updated_at']}")
    print()
