"""
Test MySQL MemoryDB connector
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from memory.db import MemoryDB

def test_connection():
    """Test database connection and basic operations"""
    print("=" * 60)
    print("Testing MySQL MemoryDB Connector")
    print("=" * 60)
    
    # Initialize connector
    print("\n1. Connecting to MySQL...")
    db = MemoryDB()
    
    if not db.connection:
        print("✗ Failed to connect to MySQL")
        print("Make sure MySQL is running and 'allie_memory' database exists")
        return
    
    print("✓ Connected to MySQL successfully")
    
    # Test add_fact
    print("\n2. Testing add_fact...")
    result = db.add_fact(
        keyword="capital_of_france",
        fact="Paris is the capital of France",
        source="test",
        category="geography",
        confidence=1.0
    )
    print(f"   Status: {result['status']}")
    print(f"   Message: {result['message']}")
    
    # Test get_fact
    print("\n3. Testing get_fact...")
    fact = db.get_fact("capital_of_france")
    if fact:
        print(f"   ✓ Found fact: {fact['fact']}")
        print(f"   Source: {fact['source']}, Updated: {fact['updated_at']}")
    else:
        print("   ✗ Fact not found")
    
    # Test update_fact
    print("\n4. Testing update_fact...")
    result = db.update_fact(
        keyword="capital_of_france",
        new_fact="Paris is the capital and largest city of France",
        source="correction"
    )
    print(f"   Status: {result['status']}")
    if result['status'] == 'updated':
        print(f"   Old: {result['old_fact']}")
        print(f"   New: {result['new_fact']}")
    
    # Test search_facts
    print("\n5. Testing search_facts...")
    results = db.search_facts("france", limit=5)
    print(f"   Found {len(results)} matching fact(s)")
    for r in results:
        print(f"   - {r['keyword']}: {r['fact'][:50]}...")
    
    # Test timeline
    print("\n6. Testing timeline...")
    timeline = db.timeline(limit=10)
    print(f"   Found {len(timeline)} fact(s) in timeline")
    for i, t in enumerate(timeline[:3], 1):
        print(f"   {i}. [{t['updated_at']}] {t['keyword']}: {t['fact'][:40]}...")
    
    # Test statistics
    print("\n7. Testing get_statistics...")
    stats = db.get_statistics()
    print(f"   Total facts: {stats['total_facts']}")
    print(f"   Sources: {stats['sources']}")
    print(f"   Categories: {stats['categories']}")
    
    # Cleanup test fact
    print("\n8. Cleaning up test fact...")
    deleted = db.delete_fact("capital_of_france")
    print(f"   {'✓' if deleted else '✗'} Delete result: {deleted}")
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
    
    db.close()

if __name__ == "__main__":
    test_connection()
