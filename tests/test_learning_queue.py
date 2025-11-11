#!/usr/bin/env python3
"""
Test script to verify automatic learner adds facts to learning queue
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from backend.automatic_learner import AutomaticLearner
from allie_memory import AllieMemory
from advanced_memory.db import AllieMemoryDB

def test_automatic_learning():
    # Initialize components
    memory_file = project_root / 'data' / 'allie_memory.json'
    allie_memory = AllieMemory(str(memory_file))
    advanced_memory = AllieMemoryDB()

    # Initialize learner with learning queue
    learner = AutomaticLearner(allie_memory, advanced_memory, learning_queue=advanced_memory)

    # Test with a simple fact
    test_message = 'The capital of France is Paris.'
    result = learner.process_message(test_message, 'test')

    print('Learning result:')
    print(f'Total facts learned: {result["total_facts_learned"]}')
    print(f'Learning actions: {len(result["learning_actions"])}')

    for action in result['learning_actions']:
        print(f'- {action}')

    # Check if anything was added to learning queue
    queue_items = advanced_memory.get_learning_queue('pending')
    print(f'\nPending queue items after test: {len(queue_items)}')

    if queue_items:
        print('New queue items:')
        for item in queue_items[-3:]:  # Show last 3 items
            print(f'  ID: {item["id"]}, Keyword: {item["keyword"]}, Fact: {item["fact"][:50]}...')

if __name__ == '__main__':
    test_automatic_learning()