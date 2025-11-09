"""
Test MySQL integration with HybridMemory
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from hybrid import HybridMemory
from db import MemoryDB

def test_mysql_integration():
    """Test complete MySQL integration with hybrid memory"""
    print("=" * 60)
    print("Testing MySQL Integration with HybridMemory")
    print("=" * 60)
    
    # Initialize hybrid memory (should auto-sync from MySQL)
    print("\n1. Initializing HybridMemory...")
    memory = HybridMemory()
    print(f"   ✓ Initialized: {memory}")
    
    # Check initial stats
    print("\n2. Checking initial statistics...")
    stats = memory.get_statistics()
    print(f"   Total facts: {stats.get('total_facts', 0)}")
    print(f"   Sources: {stats.get('sources', {})}")
    
    # Add a new fact
    print("\n3. Adding new fact...")
    result = memory.add_fact(
        fact="The Eiffel Tower is 330 meters tall",
        category="landmarks",
        confidence=1.0,
        source="test"
    )
    print(f"   Status: {result['status']}")
    print(f"   Message: {result['message']}")
    print(f"   Keyword: {result.get('keyword')}")
    
    # Search for the fact
    print("\n4. Searching for 'eiffel tower'...")
    search_results = memory.search("eiffel tower", limit=3)
    print(f"   Found {len(search_results)} result(s)")
    for i, r in enumerate(search_results, 1):
        print(f"   {i}. {r['fact'][:60]}...")
    
    # Add another fact
    print("\n5. Adding another fact...")
    result2 = memory.add_fact(
        fact="Tokyo is the capital of Japan",
        category="geography",
        confidence=1.0,
        source="test"
    )
    print(f"   Status: {result2['status']}")
    print(f"   Keyword: {result2.get('keyword')}")
    
    # Update existing fact
    print("\n6. Updating 'Tokyo' fact...")
    result3 = memory.add_fact(
        fact="Tokyo is the capital and largest city of Japan",
        category="geography",
        confidence=1.0,
        source="correction"
    )
    print(f"   Status: {result3['status']}")
    if result3.get('updated'):
        print(f"   Old: {result3.get('old_fact')}")
        print(f"   New: {result3['fact']}")
    
    # Search for geography facts
    print("\n7. Searching for 'japan'...")
    japan_results = memory.search("japan", limit=5)
    print(f"   Found {len(japan_results)} result(s)")
    for i, r in enumerate(japan_results, 1):
        print(f"   {i}. {r['fact']}")
    
    # Get updated statistics
    print("\n8. Final statistics...")
    final_stats = memory.get_statistics()
    print(f"   Total facts: {final_stats.get('total_facts', 0)}")
    print(f"   Categories: {final_stats.get('categories', {})}")
    print(f"   Sources: {final_stats.get('sources', {})}")
    
    # Verify MySQL persistence
    print("\n9. Verifying MySQL persistence...")
    db = MemoryDB()
    tokyo_fact = db.get_fact("tokyo")
    if tokyo_fact:
        print(f"   ✓ Found in MySQL: {tokyo_fact['fact']}")
        print(f"   Updated: {tokyo_fact['updated_at']}")
    else:
        print("   ✗ Not found in MySQL")
    
    eiffel_fact = db.get_fact("eiffel")
    if eiffel_fact:
        print(f"   ✓ Found in MySQL: {eiffel_fact['fact']}")
    else:
        print("   ✗ Not found in MySQL")
    
    # Test restart scenario
    print("\n10. Testing restart scenario...")
    print("    Creating new HybridMemory instance (simulating restart)...")
    memory2 = HybridMemory()
    
    restart_results = memory2.search("tokyo", limit=3)
    print(f"    Found {len(restart_results)} fact(s) after restart")
    for r in restart_results:
        print(f"    - {r['fact']}")
    
    if len(restart_results) > 0:
        print("    ✓ Facts persisted across restart!")
    else:
        print("    ✗ Facts not found after restart")
    
    # Cleanup test facts
    print("\n11. Cleaning up test facts...")
    db.delete_fact("eiffel")
    db.delete_fact("tokyo")
    print("    ✓ Test facts deleted")
    
    db.close()
    
    print("\n" + "=" * 60)
    print("MySQL Integration Test Completed!")
    print("=" * 60)

if __name__ == "__main__":
    try:
        test_mysql_integration()
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
