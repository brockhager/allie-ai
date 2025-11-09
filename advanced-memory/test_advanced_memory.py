#!/usr/bin/env python3
"""
Test script for Allie's Advanced Memory System

Tests all major functionality including:
- Database connection
- Basic CRUD operations
- Learning pipeline
- Queue management
- Clustering
- Statistics
"""

import sys
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def print_header(text):
    """Print a formatted header"""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def print_success(text):
    """Print success message"""
    print(f"✓ {text}")


def print_error(text):
    """Print error message"""
    print(f"✗ {text}")


def print_info(text):
    """Print info message"""
    print(f"  {text}")


def test_database_connection():
    """Test 1: Database connection"""
    print_header("TEST 1: Database Connection")
    
    try:
        from db import AllieMemoryDB
        memory = AllieMemoryDB()
        print_success("Connected to MySQL database")
        print_info(f"Database: {memory.database}")
        print_info(f"User: {memory.user}")
        return memory
    except Exception as e:
        print_error(f"Connection failed: {e}")
        return None


def test_basic_operations(memory):
    """Test 2: Basic CRUD operations"""
    print_header("TEST 2: Basic CRUD Operations")
    
    # Add fact
    print_info("Adding test fact...")
    result = memory.add_fact(
        keyword='test_python',
        fact='Python is an interpreted programming language created by Guido van Rossum',
        source='test_script',
        confidence=0.95,
        category='testing'
    )
    
    if result['status'] == 'added':
        print_success(f"Added fact with ID: {result['fact_id']}")
    else:
        print_error(f"Failed to add fact: {result}")
        return False
    
    # Get fact
    print_info("Retrieving fact...")
    fact = memory.get_fact('test_python')
    if fact:
        print_success("Retrieved fact successfully")
        print_info(f"  Keyword: {fact['keyword']}")
        print_info(f"  Fact: {fact['fact']}")
        print_info(f"  Confidence: {fact['confidence']}")
    else:
        print_error("Failed to retrieve fact")
        return False
    
    # Update fact
    print_info("Updating fact...")
    result = memory.update_fact(
        keyword='test_python',
        new_fact='Python is an interpreted, high-level programming language created by Guido van Rossum in 1991',
        source='test_script_updated',
        confidence=0.98
    )
    
    if result['status'] == 'updated':
        print_success("Updated fact successfully")
    else:
        print_error(f"Failed to update fact: {result}")
        return False
    
    # Search facts
    print_info("Searching for 'python'...")
    results = memory.search_facts('python', limit=5)
    print_success(f"Found {len(results)} matching facts")
    for r in results:
        print_info(f"  - {r['keyword']}: {r['fact'][:50]}...")
    
    return True


def test_learning_pipeline(memory):
    """Test 3: Learning pipeline"""
    print_header("TEST 3: Learning Pipeline")
    
    try:
        from learning_pipeline import LearningPipeline
        pipeline = LearningPipeline(memory)
        print_success("Initialized learning pipeline")
    except Exception as e:
        print_error(f"Failed to initialize pipeline: {e}")
        return False
    
    # Test single fact processing
    print_info("Processing single fact through pipeline...")
    result = pipeline.process_fact(
        keyword='test_javascript',
        fact='JavaScript is a scripting language that runs in web browsers',
        source='user',
        base_confidence=0.9,
        category='testing',
        auto_resolve=True
    )
    
    print_success(f"Pipeline status: {result['final_status']}")
    print_info(f"  Confidence: {result['confidence']}")
    print_info(f"  Stages completed: {len(result['stages'])}")
    
    # Test batch processing
    print_info("Processing batch of facts...")
    test_facts = [
        {
            'keyword': 'test_sql',
            'fact': 'SQL is a language for managing relational databases',
            'source': 'quick_teach',
            'confidence': 0.95,
            'category': 'testing'
        },
        {
            'keyword': 'test_html',
            'fact': 'HTML is the markup language for web pages',
            'source': 'user',
            'confidence': 0.9,
            'category': 'testing'
        },
        {
            'keyword': 'test_css',
            'fact': 'CSS is used to style web pages',
            'source': 'conversation',
            'confidence': 0.85,
            'category': 'testing'
        }
    ]
    
    batch_result = pipeline.process_batch(test_facts)
    print_success("Batch processing complete")
    print_info(f"  Total: {batch_result['total']}")
    print_info(f"  Added: {batch_result['added']}")
    print_info(f"  Updated: {batch_result['updated']}")
    print_info(f"  Skipped: {batch_result['skipped']}")
    print_info(f"  Rejected: {batch_result['rejected']}")
    
    return True


def test_queue_management(memory):
    """Test 4: Learning queue"""
    print_header("TEST 4: Learning Queue Management")
    
    # Add to queue
    print_info("Adding fact to learning queue...")
    result = memory.add_to_learning_queue(
        keyword='test_queue',
        fact='This is a test fact in the queue',
        source='test_script',
        confidence=0.6,
        category='testing'
    )
    
    if result['status'] == 'queued':
        queue_id = result['queue_id']
        print_success(f"Added to queue with ID: {queue_id}")
    else:
        print_error(f"Failed to add to queue: {result}")
        return False
    
    # Get queue items
    print_info("Retrieving pending queue items...")
    pending = memory.get_learning_queue('pending', limit=10)
    print_success(f"Found {len(pending)} pending items")
    
    if pending:
        for item in pending[:3]:  # Show first 3
            print_info(f"  - {item['keyword']}: {item['fact'][:40]}... (conf: {item['confidence']})")
    
    # Process queue item
    if queue_id:
        print_info(f"Processing queue item {queue_id}...")
        result = memory.process_queue_item(queue_id, 'validate', confidence=0.8)
        if result['status'] == 'success':
            print_success("Queue item validated successfully")
        else:
            print_error(f"Failed to process queue item: {result}")
    
    return True


def test_clustering(memory):
    """Test 5: Fact clustering"""
    print_header("TEST 5: Fact Clustering")
    
    # Create cluster
    print_info("Creating test cluster...")
    result = memory.create_cluster(
        cluster_name='test_programming',
        description='Test cluster for programming-related facts'
    )
    
    if result['status'] in ['created', 'exists']:
        print_success(f"Cluster ready: {result['cluster_name']}")
    else:
        print_error(f"Failed to create cluster: {result}")
        return False
    
    # Get some fact IDs
    test_facts = memory.search_facts('test', limit=5)
    
    if test_facts:
        print_info(f"Adding {len(test_facts)} facts to cluster...")
        for fact in test_facts:
            result = memory.add_to_cluster('test_programming', fact['id'], relevance_score=0.9)
            if result['status'] == 'added':
                print_success(f"  Added '{fact['keyword']}' to cluster")
    
    # Get cluster facts
    print_info("Retrieving cluster facts...")
    cluster_facts = memory.get_cluster_facts('test_programming')
    print_success(f"Cluster contains {len(cluster_facts)} facts")
    
    for fact in cluster_facts[:3]:  # Show first 3
        print_info(f"  - {fact['keyword']}: relevance {fact['relevance_score']}")
    
    return True


def test_statistics(memory):
    """Test 6: Statistics"""
    print_header("TEST 6: Statistics Dashboard")
    
    print_info("Gathering memory statistics...")
    stats = memory.get_statistics()
    
    print_success("Statistics retrieved successfully")
    print_info(f"  Total facts: {stats.get('total_facts', 0)}")
    print_info(f"  Average confidence: {stats.get('average_confidence', 0)}")
    print_info(f"  Learning log entries: {stats.get('learning_log_entries', 0)}")
    
    if stats.get('by_category'):
        print_info("  Facts by category:")
        for category, count in stats['by_category'].items():
            print_info(f"    - {category}: {count}")
    
    if stats.get('by_source'):
        print_info("  Facts by source:")
        for source, count in list(stats['by_source'].items())[:5]:  # Show top 5
            print_info(f"    - {source}: {count}")
    
    if stats.get('queue_status'):
        print_info("  Queue status:")
        for status, count in stats['queue_status'].items():
            print_info(f"    - {status}: {count}")
    
    return True


def test_timeline(memory):
    """Test 7: Timeline"""
    print_header("TEST 7: Memory Timeline")
    
    print_info("Retrieving recent facts...")
    recent = memory.timeline(limit=10, include_deleted=False)
    
    print_success(f"Retrieved {len(recent)} recent facts")
    for fact in recent[:5]:  # Show first 5
        updated = fact.get('updated_at', 'N/A')
        print_info(f"  - {fact['keyword']} (updated: {updated})")
    
    return True


def cleanup_test_data(memory):
    """Clean up test data"""
    print_header("CLEANUP: Removing Test Data")
    
    print_info("Removing test facts...")
    test_keywords = [
        'test_python', 'test_javascript', 'test_sql',
        'test_html', 'test_css', 'test_queue'
    ]
    
    removed = 0
    for keyword in test_keywords:
        result = memory.delete_fact(keyword)
        if result['status'] == 'deleted':
            removed += 1
    
    print_success(f"Removed {removed} test facts")


def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("  ALLIE ADVANCED MEMORY SYSTEM - TEST SUITE")
    print("=" * 60)
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    tests_passed = 0
    tests_failed = 0
    
    # Test 1: Connection
    memory = test_database_connection()
    if memory:
        tests_passed += 1
    else:
        tests_failed += 1
        print("\n❌ Cannot proceed without database connection")
        return
    
    # Test 2: Basic operations
    if test_basic_operations(memory):
        tests_passed += 1
    else:
        tests_failed += 1
    
    # Test 3: Learning pipeline
    if test_learning_pipeline(memory):
        tests_passed += 1
    else:
        tests_failed += 1
    
    # Test 4: Queue management
    if test_queue_management(memory):
        tests_passed += 1
    else:
        tests_failed += 1
    
    # Test 5: Clustering
    if test_clustering(memory):
        tests_passed += 1
    else:
        tests_failed += 1
    
    # Test 6: Statistics
    if test_statistics(memory):
        tests_passed += 1
    else:
        tests_failed += 1
    
    # Test 7: Timeline
    if test_timeline(memory):
        tests_passed += 1
    else:
        tests_failed += 1
    
    # Cleanup
    cleanup_test_data(memory)
    
    # Final report
    print_header("TEST SUMMARY")
    print_success(f"Tests passed: {tests_passed}")
    if tests_failed > 0:
        print_error(f"Tests failed: {tests_failed}")
    else:
        print_success("All tests passed! ✨")
    
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60 + "\n")
    
    # Close connection
    memory.close()


if __name__ == '__main__':
    try:
        run_all_tests()
    except KeyboardInterrupt:
        print("\n\n⚠ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
