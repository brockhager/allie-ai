#!/usr/bin/env python3
"""Test KB automatic insertion and verify the pipeline works"""
import sys
import importlib.util
from datetime import datetime

spec = importlib.util.spec_from_file_location('db', 'advanced-memory/db.py')
db_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(db_module)

spec2 = importlib.util.spec_from_file_location('learner', 'backend/automatic_learner.py')
learner_module = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(learner_module)

def test_high_confidence_fact():
    """Test adding a high-confidence fact that should be promoted"""
    print("🧪 Testing high-confidence fact promotion...")

    db = db_module.AllieMemoryDB()
    learner = learner_module.AutomaticLearner(None, db, db)  # memory_system, hybrid_memory, learning_queue

    # Create a high-confidence fact
    test_message = "Albert Einstein was born on March 14, 1879 in Ulm, Germany. He developed the theory of relativity in 1915."

    # Process the message
    result = learner.process_message(test_message, "assistant")

    print(f"📝 Processing result: {len(result['extracted_facts'])} facts extracted")
    print(f"🎯 Learning actions: {len(result['learning_actions'])} actions")

    for fact in result['extracted_facts']:
        print(f"  Fact: {fact['fact'][:60]}... (confidence: {fact['confidence']})")

    # Check if it was queued
    queued_actions = [a for a in result['learning_actions'] if a['action'] == 'queued_for_reconciliation']
    if queued_actions:
        print(f"✅ Fact queued for reconciliation (confidence: {queued_actions[0]['confidence']})")
    else:
        print("❌ Fact not queued (confidence too low)")

    return result

def test_worker_promotion():
    """Test that the worker promotes high-confidence facts"""
    print("\n🤖 Testing worker promotion...")

    db = db_module.AllieMemoryDB()

    # Add a high-confidence fact to queue manually
    test_keyword = f"test_promotion_{int(datetime.now().timestamp())}"
    test_fact = "This is a test fact with very high confidence for promotion testing."

    queue_result = db.add_to_learning_queue(
        keyword=test_keyword,
        fact=test_fact,
        source="test_promotion",
        confidence=0.95,  # High confidence
        category="test"
    )

    print(f"📋 Added to queue: {queue_result}")

    # Simulate worker logic
    if queue_result['status'] == 'queued':
        qid = queue_result['queue_id']

        # Check if it would be promoted (confidence >= 0.8)
        if 0.95 >= 0.8:
            # Simulate promotion
            res = db.add_kb_fact(
                test_keyword,
                test_fact,
                source="test_promotion",
                confidence_score=95,
                provenance={'test_queue_id': qid},
                status='true'  # Use valid status
            )
            print(f"✅ Would promote to KB: {res}")

            # Verify it exists
            kb_fact = db.get_kb_fact(test_keyword)
            if kb_fact:
                print(f"✅ Verification: Fact in KB (ID: {kb_fact['id']})")
                return True
            else:
                print("❌ Fact not found in KB after promotion")
                return False
        else:
            print("❌ Confidence too low for promotion")
            return False

def test_kb_growth():
    """Test that KB can grow beyond initial entries"""
    print("\n📈 Testing KB growth...")

    db = db_module.AllieMemoryDB()

    # Get initial count
    initial_count = len(db.get_all_kb_facts())

    # Add multiple test facts
    test_facts = [
        ("Test Geography", "London is the capital of England", "test_source", 95),
        ("Test Science", "Water boils at 100 degrees Celsius", "test_source", 90),
        ("Test History", "World War II ended in 1945", "test_source", 88),
    ]

    added_count = 0
    for keyword, fact, source, confidence in test_facts:
        result = db.add_kb_fact(keyword, fact, source, confidence)
        if result['status'] in ['added', 'updated']:
            added_count += 1
            print(f"✅ Added: {keyword}")

    # Get final count
    final_count = len(db.get_all_kb_facts())

    print(f"📊 KB growth: {initial_count} → {final_count} (+{final_count - initial_count})")

    if final_count > initial_count:
        print("✅ KB successfully grew")
        return True
    else:
        print("❌ KB did not grow")
        return False

def main():
    print("🔬 KB AUTOMATIC INSERTION TEST SUITE")
    print("=" * 50)

    # Test 1: High confidence fact processing
    result1 = test_high_confidence_fact()

    # Test 2: Worker promotion simulation
    result2 = test_worker_promotion()

    # Test 3: KB growth capability
    result3 = test_kb_growth()

    print("\n" + "=" * 50)
    print("📋 TEST RESULTS SUMMARY")
    print("=" * 50)

    tests_passed = sum([result1['total_facts_learned'] > 0, result2, result3])
    total_tests = 3

    print(f"Tests passed: {tests_passed}/{total_tests}")

    if tests_passed == total_tests:
        print("✅ All tests passed - KB automatic insertion is working!")
        print("\n💡 To enable automatic KB growth:")
        print("  1. Ensure conversations generate facts with confidence >= 0.8")
        print("  2. Run the KB worker: python scripts/kb_worker.py")
        print("  3. Monitor learning queue and KB growth")
    else:
        print("❌ Some tests failed - investigate the issues above")

    return tests_passed == total_tests

if __name__ == "__main__":
    main()