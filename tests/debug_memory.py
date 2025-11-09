import sys
sys.path.insert(0, '.')
from backend.memory.hybrid import HybridMemory
hybrid_memory = HybridMemory()
timeline = hybrid_memory.get_timeline(include_outdated=True)
print('Recent memory facts related to elections, programming, food:')
keywords = ['election', 'programming', 'meal', 'food', 'date', 'landmark', 'king', 'voting', 'los angeles', 'la brea', 'tokyo', 'bakery']
for fact in timeline[-100:]:
    fact_text = fact['fact'].lower()
    if any(keyword in fact_text for keyword in keywords):
        print(f'  {fact["timestamp"][:19]} [{fact["category"]}] {fact["fact"][:150]}...')