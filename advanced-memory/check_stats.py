from db import AllieMemoryDB

mem = AllieMemoryDB()
stats = mem.get_statistics()

print(f"Total facts: {stats['total_facts']}")
print(f"Average confidence: {stats['average_confidence']}")
print(f"Learning log entries: {stats['learning_log_entries']}")
print(f"\nTop categories:")
for cat, count in list(stats['by_category'].items())[:8]:
    print(f"  - {cat}: {count}")

mem.close()
