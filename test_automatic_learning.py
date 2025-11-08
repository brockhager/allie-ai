#!/usr/bin/env python3
"""
Test script for automatic learning functionality
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from automatic_learner import AutomaticLearner

class MockMemorySystem:
    """Mock memory system for testing"""
    def __init__(self):
        self.knowledge_base = {"facts": []}
        self.facts_added = []

    def add_fact(self, fact, importance=0.5, category="general"):
        """Mock add_fact method"""
        fact_entry = {
            "fact": fact,
            "importance": importance,
            "category": category,
            "timestamp": "2025-11-07T22:47:00"
        }
        self.knowledge_base["facts"].append(fact_entry)
        self.facts_added.append(fact_entry)
        print(f"✓ Stored: {fact} (category: {category}, importance: {importance})")

    def recall_facts(self, query, limit=5):
        """Mock recall method"""
        return [f["fact"] for f in self.knowledge_base["facts"] if query.lower() in f["fact"].lower()][:limit]

def test_automatic_learning():
    """Test the automatic learning system"""
    print("Testing Automatic Learning System")
    print("=" * 50)

    # Create mock memory system
    memory_system = MockMemorySystem()
    learner = AutomaticLearner(memory_system)

    # Test cases with factual information
    test_messages = [
        "Paris is the capital of France and has about 2.2 million people living there.",
        "Albert Einstein was born in Germany and developed the theory of relativity.",
        "The first computer was invented by Charles Babbage in the 19th century.",
        "Python is a programming language created by Guido van Rossum.",
        "World War II ended in 1945 with the defeat of Nazi Germany.",
        "The human heart pumps about 2,000 gallons of blood each day.",
        "Tokyo is the largest city in Japan with over 13 million residents.",
        "Photosynthesis is the process by which plants convert sunlight into energy.",
        "The iPhone was released by Apple in 2007.",
        "Mount Everest is the highest mountain in the world at 29,029 feet."
    ]

    total_facts_learned = 0

    for i, message in enumerate(test_messages, 1):
        print(f"\nTest {i}: Processing message: '{message}'")
        print("-" * 60)

        result = learner.process_message(message, "user")

        print(f"Extracted {len(result['extracted_facts'])} facts:")
        for fact in result['extracted_facts']:
            print(f"  - {fact['fact']} (category: {fact['category']}, confidence: {fact['confidence']:.2f})")

        print(f"Learning actions: {len(result['learning_actions'])}")
        for action in result['learning_actions']:
            print(f"  - {action['action']}: {action['fact'][:50]}...")

        total_facts_learned += result['total_facts_learned']

        # Generate learning response
        learning_response = learner.generate_learning_response(result['learning_actions'])
        if learning_response:
            print(f"Learning confirmation: {learning_response.strip()}")

    print(f"\n{'='*50}")
    print(f"Total facts learned: {total_facts_learned}")
    print(f"Total facts stored: {len(memory_system.facts_added)}")

    print(f"\nTesting recall functionality:")
    # Test geography facts
    geography_facts = [f for f in memory_system.facts_added if f.get('category') == 'geography']
    print(f"Geography facts: {len(geography_facts)}")
    for fact in geography_facts[:3]:
        print(f"  - {fact['fact']}")

    # Test biography facts
    biography_facts = [f for f in memory_system.facts_added if f.get('category') == 'biography']
    print(f"Biography facts: {len(biography_facts)}")
    for fact in biography_facts[:3]:
        print(f"  - {fact['fact']}")

if __name__ == "__main__":
    test_automatic_learning()