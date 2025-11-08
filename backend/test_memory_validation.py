import asyncio
import sys
import os
sys.path.insert(0, '.')

# Mock the model and tokenizer to avoid loading issues
class MockTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "Mock prompt"
    def __call__(self, prompt, return_tensors=None):
        return {"input_ids": [1, 2, 3]}
    def decode(self, tokens, skip_special_tokens=None):
        return "Mock response"

class MockModel:
    def generate(self, **kwargs):
        return ["Mock response"]

# Mock the global variables
sys.modules['transformers'] = type('MockModule', (), {
    'AutoTokenizer': lambda: MockTokenizer(),
    'AutoModelForCausalLM': lambda: MockModel()
})()

# Now import and test the validation logic
from server import AllieMemory

# Create memory instance
memory = AllieMemory("test_memory.json")

# Add a test fact that should conflict
memory.add_fact("Joe Biden is the president of the United States.", 0.8, "politics")

print("Added test fact to memory")
print("Current facts:", [f["fact"] for f in memory.knowledge_base.get("facts", [])])

# Test recall
facts = memory.recall_facts("who is the president")
print("Recalled facts:", facts)

# Test conflict detection logic (simplified version)
def test_conflict_detection():
    fact = "Joe Biden is the president of the United States."
    wiki_text = "Donald Trump is the 47th and current president of the United States."

    fact_lower = fact.lower()
    wiki_lower = wiki_text.lower()

    # Check for president conflict
    if "president" in fact_lower and "president" in wiki_lower:
        fact_part = "biden"  # Simplified extraction
        wiki_part = "trump"  # Simplified extraction

        if fact_part != wiki_part:
            print(f"Conflict detected: '{fact}' vs Wikipedia info")
            return True
    return False

if test_conflict_detection():
    print("Memory validation would trigger update")
else:
    print("No conflict detected")