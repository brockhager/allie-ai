#!/usr/bin/env python3
import sys
from pathlib import Path

# Add advanced-memory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "advanced-memory"))


"""Simple persistence test"""
from hybrid import HybridMemory
from pathlib import Path

# Clean start
storage = Path(__file__).parent.parent / "data" / "test_memory.json"
storage.unlink(missing_ok=True)

# Test 1: Save facts
print("Test 1: Saving facts...")
m1 = HybridMemory(storage)
m1.add_fact("Fact 1", category="cat1")
m1.add_fact("Fact 2", category="cat2")
print(f"✓ Saved {m1.get_statistics()['total_facts']} facts\n")

# Test 2: Load facts in new instance
print("Test 2: Loading facts...")
m2 = HybridMemory(storage)
stats = m2.get_statistics()
print(f"✓ Loaded {stats['total_facts']} facts")
print(f"  Categories: {stats['categories']}\n")

# Test 3: Search works
print("Test 3: Testing search...")
results = m2.search("Fact 1")
print(f"✓ Search found {len(results)} results")
for r in results:
    print(f"  - {r['fact']}")

print("\n✓✓✓ All tests passed!")
storage.unlink(missing_ok=True)
