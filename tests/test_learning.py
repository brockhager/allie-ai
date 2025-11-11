import sys
sys.path.insert(0, 'backend')
from backend.automatic_learner import AutomaticLearner
from backend.allie_memory import AllieMemory
from pathlib import Path

# Test automatic learning
memory = AllieMemory(Path('data/allie_memory.json'))
learner = AutomaticLearner(memory)

# Test with a sample message that should extract facts
test_message = 'Paris is the capital of France and it has about 2.2 million people.'
result = learner.process_message(test_message, 'user')

print('Test learning result:')
print(f'Facts extracted: {len(result["extracted_facts"])}')
print(f'Learning actions: {len(result["learning_actions"])}')

for fact in result['extracted_facts']:
    print(f'  - {fact["fact"]} (confidence: {fact["confidence"]})')

for action in result['learning_actions']:
    print(f'  Action: {action["action"]} - {action["fact"][:50]}...')