"""
Test Suite for Hybrid Memory System

Demonstrates all features of the hybrid memory system.
"""

import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))

from memory import HybridMemory


def test_basic_operations():
    """Test basic add and search operations"""
    print("=" * 60)
    print("TEST 1: Basic Operations")
    print("=" * 60)
    
    memory = HybridMemory()
    
    # Add facts
    print("\n1. Adding facts...")
    result1 = memory.add_fact("Paris is the capital of France", category="geography", source="user")
    print(f"   ✓ {result1['message']}: {result1['fact']}")
    
    result2 = memory.add_fact("Phoenix is the capital of Arizona", category="geography", source="user")
    print(f"   ✓ {result2['message']}: {result2['fact']}")
    
    result3 = memory.add_fact("Python is a programming language", category="technology", source="user")
    print(f"   ✓ {result3['message']}: {result3['fact']}")
    
    # Search
    print("\n2. Searching for 'capital'...")
    results = memory.search("capital")
    print(f"   Found {len(results)} facts:")
    for r in results:
        print(f"   - {r['fact']} [{r['category']}]")
    
    print("\n3. Searching for 'programming'...")
    results = memory.search("programming")
    print(f"   Found {len(results)} facts:")
    for r in results:
        print(f"   - {r['fact']} [{r['category']}]")


def test_chronological_timeline():
    """Test timeline traversal"""
    print("\n" + "=" * 60)
    print("TEST 2: Chronological Timeline")
    print("=" * 60)
    
    memory = HybridMemory()
    
    # Add facts in order
    facts = [
        ("Earth is the third planet from the Sun", "science"),
        ("Mars is the fourth planet from the Sun", "science"),
        ("Jupiter is the largest planet", "science"),
        ("The Moon orbits Earth", "science"),
    ]
    
    print("\n1. Adding facts in order...")
    for fact, category in facts:
        memory.add_fact(fact, category=category)
        print(f"   ✓ Added: {fact}")
    
    # Get timeline
    print("\n2. Timeline of all facts:")
    timeline = memory.get_timeline()
    for i, entry in enumerate(timeline, 1):
        print(f"   {i}. [{entry['timestamp'][:19]}] {entry['fact']}")


def test_fact_updates():
    """Test updating facts and conflict resolution"""
    print("\n" + "=" * 60)
    print("TEST 3: Fact Updates and Versioning")
    print("=" * 60)
    
    memory = HybridMemory()
    
    # Add initial fact
    print("\n1. Adding initial fact...")
    result = memory.add_fact("The capital of Arizona is Tucson", category="geography")
    print(f"   ✓ {result['fact']}")
    
    # Correct the fact
    print("\n2. Correcting with updated fact...")
    result = memory.add_fact("The capital of Arizona is Phoenix", category="geography", source="correction")
    print(f"   ✓ {result['fact']}")
    if result.get('updated'):
        print(f"   📝 Updated from: {result['old_fact']}")
    
    # Search should return only the latest
    print("\n3. Searching for 'capital arizona'...")
    results = memory.search("capital arizona")
    print(f"   Found {len(results)} active fact(s):")
    for r in results:
        print(f"   - {r['fact']} (outdated: {r['is_outdated']})")
    
    # Timeline shows both (with outdated marker)
    print("\n4. Full timeline (including outdated):")
    timeline = memory.get_timeline(include_outdated=True)
    for i, entry in enumerate(timeline, 1):
        status = " [OUTDATED]" if entry['is_outdated'] else ""
        print(f"   {i}. {entry['fact']}{status}")


def test_external_reconciliation():
    """Test reconciliation with external sources"""
    print("\n" + "=" * 60)
    print("TEST 4: External Source Reconciliation")
    print("=" * 60)
    
    memory = HybridMemory()
    
    # Add some initial facts
    print("\n1. Adding initial memory facts...")
    memory.add_fact("Paris is the capital of France", category="geography", source="user")
    memory.add_fact("Phoenix is the capital of Texas", category="geography", source="user")  # Wrong!
    print("   ✓ Added 2 facts to memory")
    
    # Simulate external search results
    external_facts = [
        "Paris is the capital of France",  # Confirms memory
        "Phoenix is the capital of Arizona",  # Conflicts with memory
        "London is the capital of England",  # New fact
    ]
    
    print("\n2. Reconciling with external sources...")
    result = memory.reconcile_with_external(
        query="capitals",
        external_facts=external_facts,
        source="wikipedia"
    )
    
    print(f"   Conflicts found: {result['conflicts_found']}")
    print(f"   Facts updated: {len(result['facts_updated'])}")
    for update in result['facts_updated']:
        print(f"     - '{update['old']}' → '{update['new']}'")
    
    print(f"   Facts added: {len(result['facts_added'])}")
    for fact in result['facts_added']:
        print(f"     - {fact}")
    
    print(f"   Memory confirmed: {len(result['memory_confirmed'])}")
    for fact in result['memory_confirmed']:
        print(f"     - {fact}")
    
    # Search again
    print("\n3. Searching for 'Phoenix' after reconciliation...")
    results = memory.search("Phoenix")
    for r in results:
        print(f"   - {r['fact']} [source: {r['source']}]")


def test_statistics():
    """Test memory statistics"""
    print("\n" + "=" * 60)
    print("TEST 5: Memory Statistics")
    print("=" * 60)
    
    memory = HybridMemory()
    
    # Add diverse facts
    print("\n1. Adding diverse facts...")
    facts = [
        ("Paris is the capital of France", "geography", "user"),
        ("Phoenix is the capital of Arizona", "geography", "wikipedia"),
        ("Python is a programming language", "technology", "user"),
        ("Earth orbits the Sun", "science", "wikipedia"),
        ("The square root of 144 is 12", "science", "calculator"),
    ]
    
    for fact, category, source in facts:
        memory.add_fact(fact, category=category, source=source)
    
    # Get statistics
    stats = memory.get_statistics()
    
    print("\n2. Memory Statistics:")
    print(f"   Total facts: {stats['total_facts']}")
    print(f"   Active facts: {stats['active_facts']}")
    print(f"   Outdated facts: {stats['outdated_facts']}")
    print(f"   Indexed keywords: {stats['indexed_keywords']}")
    
    print("\n3. Category breakdown:")
    for category, count in stats['categories'].items():
        print(f"   - {category}: {count}")
    
    print("\n4. Source breakdown:")
    for source, count in stats['sources'].items():
        print(f"   - {source}: {count}")


def test_performance():
    """Test performance of index vs sequential search"""
    print("\n" + "=" * 60)
    print("TEST 6: Performance Comparison")
    print("=" * 60)
    
    import time
    
    memory = HybridMemory()
    
    # Add many facts
    print("\n1. Adding 100 facts...")
    for i in range(100):
        memory.add_fact(
            f"Fact number {i} about topic {i % 10}",
            category="test"
        )
    print(f"   ✓ Added 100 facts")
    
    # Test index search (O(1))
    print("\n2. Testing index search (dictionary lookup)...")
    start = time.time()
    results = memory.search("topic")
    index_time = time.time() - start
    print(f"   ✓ Found {len(results)} results in {index_time*1000:.3f}ms")
    
    # Test sequential search (O(n))
    print("\n3. Testing sequential search (linked list traversal)...")
    start = time.time()
    timeline = memory.get_timeline()
    matching = [f for f in timeline if "topic" in f['fact'].lower()]
    sequential_time = time.time() - start
    print(f"   ✓ Found {len(matching)} results in {sequential_time*1000:.3f}ms")
    
    print(f"\n4. Speedup: {sequential_time/index_time:.1f}x faster with index")


def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("HYBRID MEMORY SYSTEM - TEST SUITE")
    print("=" * 60)
    
    test_basic_operations()
    test_chronological_timeline()
    test_fact_updates()
    test_external_reconciliation()
    test_statistics()
    test_performance()
    
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
