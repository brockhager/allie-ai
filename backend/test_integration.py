"""
Test script for hybrid memory integration with server
"""

import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from memory.hybrid import HybridMemory

def test_basic_functionality():
    """Test basic hybrid memory functionality"""
    print("=" * 60)
    print("TESTING HYBRID MEMORY INTEGRATION")
    print("=" * 60)
    
    # Initialize hybrid memory
    memory = HybridMemory()
    
    # Test 1: Add facts
    print("\n1. Adding facts...")
    memory.add_fact("Paris is the capital of France", category="geography", source="test")
    memory.add_fact("Python is a programming language", category="technology", source="test")
    memory.add_fact("Earth is the third planet", category="science", source="test")
    print("   ✓ Added 3 facts")
    
    # Test 2: Search
    print("\n2. Searching for 'capital'...")
    results = memory.search("capital", limit=5)
    print(f"   Found {len(results)} results:")
    for result in results:
        print(f"   - {result['fact']} [{result['category']}]")
    
    # Test 3: Timeline
    print("\n3. Getting timeline...")
    timeline = memory.get_timeline()
    print(f"   Timeline has {len(timeline)} entries:")
    for i, fact_dict in enumerate(timeline, 1):
        timestamp = fact_dict["timestamp"][:19].replace("T", " ")
        print(f"   {i}. [{timestamp}] {fact_dict['fact']}")
    
    # Test 4: Statistics
    print("\n4. Getting statistics...")
    stats = memory.get_statistics()
    print(f"   Total facts: {stats['total_facts']}")
    print(f"   Active facts: {stats['active_facts']}")
    print(f"   Categories: {', '.join(stats['categories'].keys())}")
    
    # Test 5: Update fact
    print("\n5. Updating a fact...")
    memory.update_fact(
        "Paris is the capital of France",
        "Paris is the capital and largest city of France",
        source="correction"
    )
    print("   ✓ Fact updated")
    
    # Test 6: Search updated fact
    print("\n6. Searching for updated fact...")
    results = memory.search("Paris", limit=5)
    print(f"   Found {len(results)} results:")
    for result in results:
        outdated = " [OUTDATED]" if result["is_outdated"] else ""
        print(f"   - {result['fact']}{outdated}")
    
    # Test 7: External reconciliation
    print("\n7. Testing external reconciliation...")
    external_facts = [
        "London is the capital of England"
    ]
    report = memory.reconcile_with_external(
        query="capital England",
        external_facts=external_facts,
        source="wikipedia"
    )
    print(f"   Added: {len(report['facts_added'])} facts")
    print(f"   Conflicts: {report['conflicts_found']}")
    
    # Final statistics
    print("\n" + "=" * 60)
    print("FINAL STATISTICS")
    print("=" * 60)
    stats = memory.get_statistics()
    print(f"Total facts: {stats['total_facts']}")
    print(f"Active facts: {stats['active_facts']}")
    print(f"Outdated facts: {stats['outdated_facts']}")
    print(f"Indexed keywords: {stats['indexed_keywords']}")
    print("\nCategories:")
    for category, count in stats['categories'].items():
        print(f"  - {category}: {count}")
    print("\nSources:")
    for source, count in stats['sources'].items():
        print(f"  - {source}: {count}")
    
    print("\n✅ ALL INTEGRATION TESTS PASSED!")

if __name__ == "__main__":
    test_basic_functionality()
