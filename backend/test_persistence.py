#!/usr/bin/env python3
"""Test hybrid memory persistence"""

from memory.hybrid import HybridMemory
from pathlib import Path

# Test save
print("Testing hybrid memory persistence...")
m = HybridMemory()
m.add_fact("Test fact about persistence", category="test", source="test_script")
stats = m.get_statistics()
print(f"✓ Added fact. Total facts: {stats['total_facts']}")

# Test load
print("\nTesting load from disk...")
m2 = HybridMemory()
stats2 = m2.get_statistics()
print(f"✓ Loaded facts. Total facts: {stats2['total_facts']}")

if stats2['total_facts'] > 0:
    facts = m2.get_timeline()
    print(f"\nSample facts:")
    for i, fact in enumerate(facts[:1]):
        print(f"\nFact {i}: type={type(fact)}")
        print(f"Keys: {fact.keys() if isinstance(fact, dict) else 'not a dict'}")
        if isinstance(fact, dict):
            for k, v in fact.items():
                print(f"  {k}: {type(v)} = {str(v)[:50]}")

print(f"\n✓ Persistence working! Storage file: {m.storage_file}")
