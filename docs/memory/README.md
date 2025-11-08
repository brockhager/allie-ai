# Allie AI Memory System Documentation

## Overview

Allie AI's memory system enables persistent learning and intelligent fact retrieval. This directory contains documentation for the memory architecture and its components.

## Documentation

### [Hybrid Memory System Guide](../HYBRID_MEMORY_GUIDE.md)

Comprehensive guide to the Hybrid Memory System that combines:
- **Linked List**: Chronological fact storage with timeline queries
- **Hash Map Index**: O(1) keyword lookup for fast searches
- **Fact Versioning**: Track updates and corrections
- **External Reconciliation**: Validate facts against Wikipedia and other sources

**Topics Covered:**
- Architecture and design
- User commands (`show memory timeline`, `memory statistics`)
- API endpoints for all operations
- Integration with chat system
- Testing and benchmarks
- Technical implementation details

## Memory System Components

### Core Modules

```
backend/memory/
├── __init__.py           # Module exports
├── linked_list.py        # Chronological storage (FactNode, FactLinkedList)
├── index.py              # Keyword indexing (KeywordIndex)
└── hybrid.py             # Integration layer (HybridMemory)
```

### Key Classes

- **`FactNode`**: Individual fact with metadata (timestamp, category, confidence, source)
- **`FactLinkedList`**: Chronological linked list for timeline queries
- **`KeywordIndex`**: Dictionary-based O(1) keyword lookup
- **`HybridMemory`**: Unified interface combining both structures

## Features

### 🔍 Fast Search
- O(1) keyword lookup via dictionary indexing
- Automatic keyword extraction with stop-word filtering
- Relevance scoring for better results

### 📅 Timeline View
- Chronological ordering preserved
- View learning progression over time
- Filter outdated vs. active facts

### 🔄 Fact Versioning
- Track fact updates with `is_outdated` flag
- Maintain history of corrections
- Link updated facts via `updated_by` reference

### 🌐 External Validation
- Reconcile with Wikipedia and other sources
- Automatic conflict detection
- Prefer fresher data from authoritative sources

### 📊 Rich Metadata
- **Category**: geography, technology, science, etc.
- **Source**: user, wikipedia, calculator, correction
- **Confidence**: 0.0 to 1.0 rating
- **Timestamp**: When the fact was learned

## Quick Start

### Using the Memory System

```python
from memory.hybrid import HybridMemory

# Initialize
memory = HybridMemory()

# Add a fact
memory.add_fact(
    "Paris is the capital of France",
    category="geography",
    confidence=1.0,
    source="user"
)

# Search for facts
results = memory.search("capital", limit=5)
for fact_dict in results:
    print(f"{fact_dict['fact']} [{fact_dict['category']}]")

# Get chronological timeline
timeline = memory.get_timeline(include_outdated=False)

# Update a fact
memory.update_fact(
    old_fact="Paris is the capital of France",
    new_fact="Paris is the capital and largest city of France",
    source="correction"
)

# Get statistics
stats = memory.get_statistics()
print(f"Total: {stats['total_facts']} facts")
```

### User Commands

In chat with Allie:
- `show memory timeline` - Display all facts chronologically
- `memory statistics` - Show detailed analytics

## API Reference

### REST Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/hybrid-memory/add` | POST | Add a new fact |
| `/api/hybrid-memory/search` | GET | Search for facts by query |
| `/api/hybrid-memory/timeline` | GET | Get chronological timeline |
| `/api/hybrid-memory/update` | PUT | Update an existing fact |
| `/api/hybrid-memory/reconcile` | POST | Reconcile with external sources |
| `/api/hybrid-memory/statistics` | GET | Get memory statistics |

See the [full guide](../HYBRID_MEMORY_GUIDE.md) for detailed API documentation with examples.

## Testing

Run the test suites:

```bash
# Comprehensive unit tests
python backend/test_hybrid_memory.py

# Integration tests with server
python backend/test_integration.py
```

Tests cover:
- ✅ Basic operations (add, search, timeline)
- ✅ Chronological ordering
- ✅ Fact updates and versioning
- ✅ External reconciliation
- ✅ Statistics and analytics
- ✅ Performance benchmarks

## Performance

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Keyword Search | O(1) | Dictionary lookup |
| Sequential Search | O(n) | Full traversal |
| Add Fact | O(1) | Append to list + index |
| Update Fact | O(1) | Mark outdated |
| Timeline Query | O(n) | Traverse linked list |

Benchmark (100 facts):
- Index search: ~0.034ms
- Sequential search: ~0.010ms

## Architecture Benefits

### Hybrid Memory vs. Legacy System

| Feature | Legacy | Hybrid |
|---------|--------|--------|
| Search Speed | O(n) | O(1) |
| Timeline | ❌ | ✅ |
| Versioning | ❌ | ✅ |
| Keyword Index | ❌ | ✅ |
| Rich Metadata | Limited | Full |
| External Validation | Basic | Advanced |

## Future Enhancements

Planned improvements:
- [ ] Persistence layer (save to disk)
- [ ] Fact importance weighting
- [ ] Topic clustering
- [ ] Semantic search integration
- [ ] Confidence decay over time
- [ ] Multi-source validation
- [ ] User feedback loop

## Contributing

When working with the memory system:

1. **Add tests** for new features in `test_hybrid_memory.py`
2. **Update documentation** when changing APIs
3. **Maintain backward compatibility** with legacy memory system
4. **Log operations** for debugging and transparency
5. **Benchmark performance** for optimization opportunities

## Related Documentation

- [Main README](../../README.md)
- [Hybrid Memory System Guide](../HYBRID_MEMORY_GUIDE.md)
- [API Documentation](../API.md) *(if exists)*
- [Architecture Overview](../ARCHITECTURE.md) *(if exists)*

## Support

For issues or questions:
1. Check the [Hybrid Memory Guide](../HYBRID_MEMORY_GUIDE.md)
2. Review test files for usage examples
3. Examine the source code with inline documentation

---

**Status**: ✅ Production Ready  
**Last Updated**: November 8, 2025  
**Version**: 1.0.0
