from memory.hybrid import HybridMemory

# Test Rocky Mountains search
m = HybridMemory()
results = m.search('rocky mountains', limit=5)
print('Rocky Mountains search results:')
for r in results:
    print(f'  - {r["fact"][:80]}...')

# Test Eiffel Tower search
results2 = m.search('eiffel tower', limit=5)
print('\nEiffel Tower search results:')
for r in results2:
    print(f'  - {r["fact"][:80]}...')

# Check memory statistics
stats = m.get_statistics()
print(f'\nMemory stats: {stats["total_facts"]} facts')
