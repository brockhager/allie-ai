import sys
import os
sys.path.append('backend')

from backend.automatic_learner import AutomaticLearner
from backend.memory.hybrid import HybridMemory
from pathlib import Path

# Initialize memory system
memory = HybridMemory()
learner = AutomaticLearner(memory)

# Test the new patterns
test_conversations = [
    'John Smith graduated from MIT in 2010 and worked at Google for 5 years.',
    'Albert Einstein died in 1955 and published several books on physics.',
    'Jane Doe worked at Microsoft from 2015 to 2020.',
    'Mark Twain published his first novel in 2005.'
]

print('Testing new pattern expansion:')
for conv in test_conversations:
    print(f'\nTesting: {conv}')
    result = learner.process_message(conv, 'user')
    for fact in result['extracted_facts']:
        print(f'  Fact: {fact["keyword"]} -> {fact["fact"]} (confidence: {fact["confidence"]})')