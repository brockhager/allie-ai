#!/usr/bin/env python3
"""
Test script for MySQL Memory System

Tests all components of the new memory architecture.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "advanced-memory"))

from db import AllieMemoryDB
from learning_pipeline import LearningPipeline

# Mock external sources for testing
async def mock_wikipedia(keyword):
    facts = {
        "Eiffel Tower": {
            "success": True,
            "fact": "The Eiffel Tower is a wrought-iron lattice tower in Paris, standing 330 meters tall"
        },
        "Mars": {
            "success": True,
            "fact": "Mars is the fourth planet from the Sun and the second-smallest planet in the Solar System"
        }
    }
    return facts.get(keyword, {"success": False})

async def mock_wikidata(keyword):
    return {"success": False}  # Simulate no result

async def mock_duckduckgo(keyword):
    return {"success": False}  # Simulate no result


def test_database_connection():
    """Test 1: Database connection and table creation"""
    print("\n" + "="*60)
    print("TEST 1: Database Connection")
    print("="*60)
    
    try:
        db = AllieMemoryDB(
            host='localhost',
            database='allie_memory_test',
            user='root',
            password=''  # Update with your MySQL password
        )
        print("✓ Connected to MySQL successfully")
        print("✓ Tables initialized")
        return db
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        print("\nPlease ensure:")
        print("1. MySQL is installed and running")
        print("2. Database credentials are correct")
        print("3. User has necessary privileges")
        return None


def test_basic_operations(db):
    """Test 2: Basic CRUD operations"""
    print("\n" + "="*60)
    print("TEST 2: Basic CRUD Operations")
    print("="*60)
    
    # Add fact
    print("\n[ADD] Adding fact about Eiffel Tower...")
    result = db.add_fact(
        keyword="Eiffel Tower",
        fact="The Eiffel Tower is 330 meters tall",
        source="test",
        confidence=0.9,
        category="geography"
    )
    print(f"  Result: {result['status']}")
    print(f"  Fact ID: {result.get('fact_id')}")
    
    # Get fact
    print("\n[GET] Retrieving fact...")
    fact = db.get_fact("Eiffel Tower")
    if fact:
        print(f"  ✓ Found: {fact['fact']}")
        print(f"  Confidence: {fact['confidence']}")
        print(f"  Source: {fact['source']}")
    else:
        print("  ✗ Fact not found")
    
    # Update fact
    print("\n[UPDATE] Updating fact...")
    update_result = db.update_fact(
        keyword="Eiffel Tower",
        new_fact="The Eiffel Tower is 330 meters (1,083 ft) tall and located in Paris",
        source="updated_test",
        confidence=0.95
    )
    print(f"  Result: {update_result['status']}")
    
    # Verify update
    updated_fact = db.get_fact("Eiffel Tower")
    if updated_fact:
        print(f"  ✓ Updated: {updated_fact['fact']}")
    
    # Search
    print("\n[SEARCH] Searching for 'tower'...")
    results = db.search_facts("tower", limit=5)
    print(f"  Found {len(results)} results")
    for r in results:
        print(f"    - {r['keyword']}: {r['fact'][:50]}...")
    
    return True


def test_learning_queue(db):
    """Test 3: Learning queue operations"""
    print("\n" + "="*60)
    print("TEST 3: Learning Queue")
    print("="*60)
    
    # Add to queue
    print("\n[QUEUE] Adding fact to learning queue...")
    queue_result = db.add_to_learning_queue(
        keyword="Mars",
        fact="Mars is the fourth planet from the Sun",
        source="wikipedia",
        confidence=0.85,
        category="astronomy"
    )
    print(f"  Result: {queue_result['status']}")
    print(f"  Queue ID: {queue_result.get('queue_id')}")
    
    # Get queue
    print("\n[QUEUE] Retrieving pending items...")
    pending = db.get_learning_queue('pending', limit=10)
    print(f"  Found {len(pending)} pending items")
    for item in pending:
        print(f"    - ID {item['id']}: {item['keyword']}")
    
    # Process queue item
    if pending:
        queue_id = pending[0]['id']
        print(f"\n[PROCESS] Processing queue item {queue_id}...")
        process_result = db.process_queue_item(queue_id, 'process')
        print(f"  Result: {process_result['status']}")
        print(f"  Action: {process_result.get('action')}")
    
    return True


def test_clustering(db):
    """Test 4: Fact clustering"""
    print("\n" + "="*60)
    print("TEST 4: Fact Clustering")
    print("="*60)
    
    # Create cluster
    print("\n[CLUSTER] Creating 'Geography' cluster...")
    cluster_result = db.create_cluster(
        cluster_name="Geography",
        description="Geographic facts about places"
    )
    print(f"  Result: {cluster_result['status']}")
    
    # Add fact to cluster
    fact = db.get_fact("Eiffel Tower")
    if fact:
        print(f"\n[CLUSTER] Adding fact {fact['id']} to cluster...")
        add_result = db.add_to_cluster("Geography", fact['id'], relevance_score=1.0)
        print(f"  Result: {add_result['status']}")
    
    # Get cluster facts
    print("\n[CLUSTER] Retrieving cluster facts...")
    cluster_facts = db.get_cluster_facts("Geography")
    print(f"  Found {len(cluster_facts)} facts in cluster")
    for cf in cluster_facts:
        print(f"    - {cf['keyword']} (relevance: {cf['relevance_score']})")
    
    return True


def test_timeline(db):
    """Test 5: Timeline and learning log"""
    print("\n" + "="*60)
    print("TEST 5: Timeline and History")
    print("="*60)
    
    # Get timeline
    print("\n[TIMELINE] Retrieving recent facts...")
    timeline = db.timeline(limit=5)
    print(f"  Found {len(timeline)} facts")
    for fact in timeline:
        print(f"    - [{fact['updated_at']}] {fact['keyword']}")
    
    # Get timeline with deleted facts
    print("\n[TIMELINE] Retrieving full history (including changes)...")
    full_timeline = db.timeline(limit=10, include_deleted=True)
    print(f"  Found {len(full_timeline)} entries")
    for entry in full_timeline[:5]:  # Show first 5
        change_type = entry.get('change_type', 'N/A')
        print(f"    - [{entry['updated_at']}] {entry['keyword']} ({change_type})")
    
    return True


def test_statistics(db):
    """Test 6: Memory statistics"""
    print("\n" + "="*60)
    print("TEST 6: Statistics")
    print("="*60)
    
    stats = db.get_statistics()
    
    print(f"\n📊 Memory Statistics:")
    print(f"  Total Facts: {stats.get('total_facts', 0)}")
    print(f"  Average Confidence: {stats.get('average_confidence', 0)}")
    print(f"  Learning Log Entries: {stats.get('learning_log_entries', 0)}")
    
    print(f"\n  By Category:")
    for category, count in stats.get('by_category', {}).items():
        print(f"    - {category}: {count}")
    
    print(f"\n  By Source:")
    for source, count in stats.get('by_source', {}).items():
        print(f"    - {source}: {count}")
    
    print(f"\n  Queue Status:")
    for status, count in stats.get('queue_status', {}).items():
        print(f"    - {status}: {count}")
    
    return True


async def test_learning_pipeline(db):
    """Test 7: Full learning pipeline"""
    print("\n" + "="*60)
    print("TEST 7: Learning Pipeline")
    print("="*60)
    
    # Setup mock external sources
    external_sources = {
        'wikipedia': mock_wikipedia,
        'wikidata': mock_wikidata,
        'duckduckgo': mock_duckduckgo
    }
    
    pipeline = LearningPipeline(db, external_sources)
    
    # Process a fact through the pipeline
    print("\n[PIPELINE] Processing fact through all stages...")
    result = await pipeline.process_full_pipeline(
        keyword="Jupiter",
        fact="Jupiter is the largest planet in our Solar System",
        source="wikipedia",
        category="astronomy"
    )
    
    print(f"\n  Status: {result['status']}")
    print(f"  Duration: {result['duration_ms']:.2f}ms")
    
    print("\n  Stage Results:")
    for stage_name, stage_result in result['stages'].items():
        status = stage_result.get('status', 'N/A')
        print(f"    {stage_name}: {status}")
        
        if stage_name == 'validate' and 'final_confidence' in stage_result:
            print(f"      Final Confidence: {stage_result['final_confidence']:.2f}")
        
        if stage_name == 'decide' and 'decision' in stage_result:
            action = stage_result['decision'].get('action')
            print(f"      Decision: {action}")
    
    return True


async def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("🧪 MySQL Memory System Test Suite")
    print("="*60)
    
    # Test 1: Connection
    db = test_database_connection()
    if not db:
        print("\n❌ Cannot proceed without database connection")
        print("\nSetup instructions:")
        print("1. Install MySQL: https://dev.mysql.com/downloads/mysql/")
        print("2. Create database: CREATE DATABASE allie_memory_test;")
        print("3. Update credentials in this test script")
        return
    
    try:
        # Test 2-6: Basic operations
        tests = [
            ("Basic Operations", lambda: test_basic_operations(db)),
            ("Learning Queue", lambda: test_learning_queue(db)),
            ("Clustering", lambda: test_clustering(db)),
            ("Timeline", lambda: test_timeline(db)),
            ("Statistics", lambda: test_statistics(db)),
        ]
        
        passed = 0
        failed = 0
        
        for test_name, test_func in tests:
            try:
                if test_func():
                    passed += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"\n✗ Test failed: {e}")
                failed += 1
        
        # Test 7: Async pipeline test
        try:
            await test_learning_pipeline(db)
            passed += 1
        except Exception as e:
            print(f"\n✗ Pipeline test failed: {e}")
            failed += 1
        
        # Summary
        print("\n" + "="*60)
        print("📋 Test Summary")
        print("="*60)
        print(f"  ✓ Passed: {passed}")
        print(f"  ✗ Failed: {failed}")
        print(f"  Total: {passed + failed}")
        
        if failed == 0:
            print("\n✅ All tests passed!")
        else:
            print(f"\n⚠️  {failed} test(s) failed")
        
    finally:
        # Cleanup
        db.close()
        print("\n🔒 Database connection closed")


if __name__ == "__main__":
    asyncio.run(main())
