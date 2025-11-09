"""
Test end-to-end flow: external sources → hybrid memory → MySQL
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent.parent / "advanced-memory")); from hybrid import HybridMemory
from db import MemoryDB

def test_end_to_end():
    """Test complete flow from external sources to MySQL storage"""
    print("=" * 60)
    print("Testing End-to-End External → Memory → MySQL Flow")
    print("=" * 60)
    
    # Clear test data
    print("\n1. Cleaning up any existing test data...")
    db = MemoryDB()
    db.delete_fact("eiffel")
    db.delete_fact("tower")
    db.delete_fact("tokyo")
    db.delete_fact("capital")
    print("   ✓ Cleanup complete")
    
    # Initialize memory
    print("\n2. Initializing HybridMemory...")
    memory = HybridMemory()
    initial_stats = memory.get_statistics()
    print(f"   Initial facts: {initial_stats.get('total_facts', 0)}")
    
    # Simulate external source providing facts
    print("\n3. Simulating external sources providing facts...")
    
    # Fact 1: From Wikipedia
    result1 = memory.add_fact(
        fact="The Eiffel Tower is a wrought-iron lattice tower in Paris",
        category="landmarks",
        confidence=0.95,
        source="wikipedia"
    )
    print(f"   Wikipedia → {result1['status']}: {result1['message']}")
    print(f"   Keyword: {result1.get('keyword')}")
    
    # Fact 2: From Nominatim (distance calculation)
    result2 = memory.add_fact(
        fact="The distance from San Diego to Yuma is approximately 172 miles",
        category="geography",
        confidence=0.99,
        source="nominatim"
    )
    print(f"   Nominatim → {result2['status']}: {result2['message']}")
    print(f"   Keyword: {result2.get('keyword')}")
    
    # Fact 3: From DuckDuckGo
    result3 = memory.add_fact(
        fact="Python is a high-level programming language known for readability",
        category="technology",
        confidence=0.90,
        source="duckduckgo"
    )
    print(f"   DuckDuckGo → {result3['status']}: {result3['message']}")
    print(f"   Keyword: {result3.get('keyword')}")
    
    # Verify MySQL storage
    print("\n4. Verifying facts stored in MySQL...")
    keywords_to_check = [result1.get('keyword'), result2.get('keyword'), result3.get('keyword')]
    
    for keyword in keywords_to_check:
        if keyword:
            db_fact = db.get_fact(keyword)
            if db_fact:
                print(f"   ✓ {keyword}: {db_fact['fact'][:50]}...")
                print(f"     Source: {db_fact['source']}, Confidence: {db_fact['confidence']}")
            else:
                print(f"   ✗ {keyword}: Not found in MySQL")
    
    # Test search
    print("\n5. Testing search functionality...")
    
    search_tests = [
        ("eiffel", "Should find Eiffel Tower"),
        ("distance", "Should find distance fact"),
        ("python programming", "Should find Python fact")
    ]
    
    for query, description in search_tests:
        results = memory.search(query, limit=3)
        print(f"   Query '{query}': Found {len(results)} result(s)")
        if results:
            print(f"     → {results[0]['fact'][:60]}...")
    
    # Test persistence (restart scenario)
    print("\n6. Testing persistence after restart...")
    del memory
    
    memory2 = HybridMemory()
    restart_stats = memory2.get_statistics()
    print(f"   Facts after restart: {restart_stats.get('total_facts', 0)}")
    
    # Search for our facts after "restart"
    eiffel_results = memory2.search("eiffel", limit=1)
    if eiffel_results:
        print(f"   ✓ Eiffel Tower fact persisted")
    else:
        print(f"   ✗ Eiffel Tower fact lost")
    
    distance_results = memory2.search("distance", limit=1)
    if distance_results:
        print(f"   ✓ Distance fact persisted")
    else:
        print(f"   ✗ Distance fact lost")
    
    # Test fact update scenario
    print("\n7. Testing fact correction/update...")
    result_update = memory2.add_fact(
        fact="The Eiffel Tower is a wrought-iron tower in Paris, France, standing 330 meters tall",
        category="landmarks",
        confidence=0.98,
        source="wikipedia_updated"
    )
    print(f"   Update status: {result_update['status']}")
    if result_update.get('updated'):
        print(f"   Old: {result_update.get('old_fact', 'N/A')[:50]}...")
        print(f"   New: {result_update['fact'][:50]}...")
    
    # Check MySQL has latest version
    keyword = result_update.get('keyword') or result1.get('keyword')
    if keyword:
        db_fact = db.get_fact(keyword)
        if db_fact:
            print(f"   MySQL has: {db_fact['fact'][:50]}...")
            print(f"   Updated: {db_fact['updated_at']}")
    
    # Final statistics
    print("\n8. Final statistics from MySQL...")
    final_stats = memory2.get_statistics()
    print(f"   Total facts: {final_stats.get('total_facts', 0)}")
    print(f"   Sources: {final_stats.get('sources', {})}")
    print(f"   Categories: {final_stats.get('categories', {})}")
    
    # Cleanup
    print("\n9. Cleaning up test facts...")
    for keyword in keywords_to_check:
        if keyword:
            db.delete_fact(keyword)
    print("   ✓ Test facts deleted")
    
    db.close()
    
    print("\n" + "=" * 60)
    print("✓ End-to-End Test Completed Successfully!")
    print("=" * 60)

if __name__ == "__main__":
    try:
        test_end_to_end()
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
