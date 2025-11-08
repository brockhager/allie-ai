# Multi-Source Knowledge Retrieval System

## Overview

Allie AI now features a **modular multi-source retrieval system** that replaces the deprecated Wikipedia API with multiple authoritative knowledge sources.

## Problem Solved

**Previous Issues:**
- Wikipedia API returning 403 Forbidden errors
- Single source dependency (fragile)
- Limited knowledge coverage
- No synthesis of multiple sources

**New Solution:**
- Multiple independent source modules
- Parallel querying for speed
- Intelligent synthesis of results
- Automatic fact storage in hybrid memory

## Architecture

```
┌──────────────────────────────────────────────────────┐
│         Knowledge Retrieval Orchestrator            │
├──────────────────────────────────────────────────────┤
│                                                      │
│  1. Memory First (Hybrid Memory System)              │
│     ↓                                                │
│  2. External Sources (Parallel):                     │
│     ┌──────────────┬──────────────┬───────────────┐ │
│     │  DuckDuckGo  │   Wikidata   │    DBpedia    │ │
│     │  (Web Search)│  (Structured)│  (Semantic)   │ │
│     └──────────────┴──────────────┴───────────────┘ │
│     ┌──────────────┐                                │
│     │ OpenLibrary  │                                │
│     │  (Cultural)  │                                │
│     └──────────────┘                                │
│     ↓                                                │
│  3. Synthesis & Storage                              │
│     - Combine results intelligently                  │
│     - Store new facts in hybrid memory               │
│     - Return coherent answer                         │
└──────────────────────────────────────────────────────┘
```

## Source Modules

### 1. DuckDuckGo (`sources/duckduckgo.py`)
- **Purpose**: General web search and instant answers
- **API**: DuckDuckGo Instant Answer API (no key required)
- **Best For**: Current events, general knowledge, quick facts
- **Response Time**: ~100-500ms

### 2. Wikidata (`sources/wikidata.py`)
- **Purpose**: Structured factual data
- **API**: MediaWiki API + SPARQL endpoint
- **Best For**: Entity facts, relationships, population, geography
- **Response Time**: ~500-1000ms
- **Data Format**: Structured JSON with properties and values

### 3. DBpedia (`sources/dbpedia.py`)
- **Purpose**: Semantic encyclopedia data
- **API**: DBpedia Lookup + SPARQL endpoint
- **Best For**: Encyclopedic knowledge, ontologies, semantic queries
- **Response Time**: ~500-1500ms
- **Data Format**: RDF triples, semantic relationships

### 4. OpenLibrary (`sources/openlibrary.py`)
- **Purpose**: Cultural and bibliographic information
- **API**: OpenLibrary Search API
- **Best For**: Books, authors, publications, cultural references
- **Response Time**: ~300-800ms
- **Data Format**: Book/author metadata with subjects

## Retrieval Workflow

### Step 1: Check Memory First
```python
memory_results = hybrid_memory.search(query, limit=5)
```
- If ≥2 relevant facts found → **Use memory (fastest)**
- If <2 facts → Continue to external sources

### Step 2: Parallel External Search
```python
results = await asyncio.gather(
    search_duckduckgo(query, 3),
    search_wikidata(query, 3),
    search_dbpedia(query, 3),
    search_openlibrary(query, 3)
)
```
All sources queried simultaneously for speed.

### Step 3: Synthesize Results
```python
synthesis = synthesize_results(
    query=query,
    memory_results=memory_results,
    duckduckgo=ddg_result,
    wikidata=wd_result,
    dbpedia=dbp_result,
    openlibrary=ol_result
)
```
- Combine information from all successful sources
- Remove duplicates
- Prioritize authoritative sources
- Create coherent narrative

### Step 4: Store New Facts
```python
for fact in synthesis["facts_to_store"]:
    hybrid_memory.add_fact(
        fact["fact"],
        category=fact["category"],
        confidence=fact["confidence"],
        source=fact["source"]
    )
```
- Automatically store new knowledge
- Tag with source and confidence
- Categorize appropriately

## Usage Examples

### Basic Search
```python
from sources.retrieval import search_all_sources

results = await search_all_sources(
    query="What is the capital of Arizona?",
    max_results_per_source=3
)

print(results["synthesized_text"])
# Output: "Phoenix is the capital of Arizona..."
```

### With Memory Integration
```python
from sources.retrieval import search_with_memory_first

results = await search_with_memory_first(
    query="What is the population of Phoenix?",
    memory_search_func=hybrid_memory.search,
    max_results=5
)
```

### Individual Source Search
```python
from sources.duckduckgo import search_duckduckgo
from sources.wikidata import search_wikidata

# Quick web search
ddg = await search_duckduckgo("Arizona cities", max_results=5)

# Structured entity data
wd = await search_wikidata("Phoenix Arizona", max_results=3)
```

## API Integration

The system is automatically integrated into Allie's chat endpoint:

```python
# In server.py /api/generate endpoint

if needs_external_search:
    multi_source_results = await search_all_sources(
        query=prompt,
        memory_results=relevant_facts,
        max_results_per_source=3
    )
    
    # Automatic fact storage
    for fact_data in multi_source_results["facts_to_store"]:
        hybrid_memory.add_fact(
            fact_data["fact"],
            category=fact_data.get("category", "general"),
            confidence=fact_data.get("confidence", 0.8),
            source=fact_data.get("source", "external")
        )
```

## Response Format

Each source module returns a consistent format:

```json
{
  "success": true,
  "source": "duckduckgo",
  "query": "capital of Arizona",
  "results": [
    {
      "title": "Phoenix",
      "text": "Phoenix is the capital and most populous city of Arizona",
      "url": "https://...",
      "source": "duckduckgo_instant"
    }
  ],
  "instant_answer": "Phoenix" // DuckDuckGo only
}
```

Synthesized response:
```json
{
  "success": true,
  "query": "capital of Arizona",
  "memory_used": false,
  "sources_used": ["duckduckgo", "wikidata", "dbpedia"],
  "synthesized_text": "Phoenix is the capital of Arizona. According to Wikidata, it has a population of 1,680,992 as of 2020...",
  "facts_to_store": [
    {
      "fact": "Phoenix is the capital of Arizona",
      "source": "duckduckgo",
      "category": "geography",
      "confidence": 1.0
    }
  ]
}
```

## Error Handling

Each source handles failures gracefully:

```python
{
  "success": false,
  "source": "wikipedia",
  "error": "HTTP 403 Forbidden",
  "results": []
}
```

The synthesis continues with remaining successful sources.

## Performance

**Typical Query Times:**
- Memory only: 1-5ms
- Memory + DuckDuckGo: 100-500ms
- Full multi-source: 500-2000ms (parallel)

**Optimization:**
- Parallel queries (4 sources simultaneously)
- 5-minute cache (reduces repeat queries)
- Early termination if memory sufficient
- Timeout protection (15s per source)

## Fact Categorization

Automatic categorization based on query keywords:

| Category | Keywords |
|----------|----------|
| geography | capital, city, country, location, population |
| history | when, history, war, ancient, founded |
| science | planet, element, theory, scientific |
| technology | computer, software, program, algorithm |
| biography | who is, who was, born, died, person |
| cultural | book, author, music, art, culture |
| mathematics | calculate, equation, math, number |

## Confidence Levels

Source-based confidence scores:

| Source | Confidence | Reasoning |
|--------|-----------|-----------|
| Memory (validated) | 1.0 | Already verified |
| DuckDuckGo Instant | 1.0 | Highly reliable |
| Wikidata | 0.95 | Structured, curated |
| DBpedia | 0.90 | Derived from Wikipedia |
| DuckDuckGo Web | 0.85 | General web |
| OpenLibrary | 0.85 | Bibliographic data |

## Advantages Over Wikipedia-Only

| Aspect | Wikipedia Only | Multi-Source |
|--------|---------------|--------------|
| Availability | ❌ 403 errors | ✅ Multiple fallbacks |
| Coverage | Limited | Comprehensive |
| Speed | Single query | Parallel queries |
| Freshness | Static | Mixed (web + structured) |
| Synthesis | None | Intelligent |
| Storage | Manual | Automatic |
| Transparency | Single source | Multi-source attribution |

## Configuration

All source modules support configuration:

```python
# In sources/duckduckgo.py
DUCKDUCKGO_API_URL = "https://api.duckduckgo.com/"
TIMEOUT = 10.0  # seconds

# In sources/wikidata.py
WIKIDATA_API_URL = "https://www.wikidata.org/w/api.php"
WIKIDATA_SPARQL_URL = "https://query.wikidata.org/sparql"
TIMEOUT = 15.0

# Similar for other sources
```

## Future Enhancements

Planned improvements:
- [ ] Add Wolfram Alpha for mathematical/computational queries
- [ ] Add arXiv for scientific papers
- [ ] Add news APIs for current events
- [ ] Implement caching per source
- [ ] Add source reliability scoring
- [ ] Implement fact conflict resolution
- [ ] Add user preferences for source priorities
- [ ] Implement semantic deduplication

## Testing

Run tests for individual sources:

```bash
# Test all sources
python -c "from sources import *; print('All modules loaded')"

# Test retrieval orchestrator
python -c "from sources.retrieval import search_all_sources; print('Retrieval ready')"
```

## Troubleshooting

### Import Errors
```bash
# Ensure sources directory is in path
export PYTHONPATH="${PYTHONPATH}:/path/to/backend"
```

### Source Timeouts
- Increase timeout in individual source files
- Check network connectivity
- Verify API endpoints are accessible

### No Results
- Check logs for individual source failures
- Verify query format
- Try simpler/broader queries

## Migration from Wikipedia

**Old Code:**
```python
wiki_results = await search_wikipedia(query)
```

**New Code:**
```python
multi_results = await search_all_sources(query)
```

The old `search_wikipedia()` function is deprecated but still works as a wrapper to DuckDuckGo for backward compatibility.

## Credits

- DuckDuckGo: https://duckduckgo.com/api
- Wikidata: https://www.wikidata.org/
- DBpedia: https://www.dbpedia.org/
- OpenLibrary: https://openlibrary.org/

---

**Status**: ✅ Production Ready  
**Last Updated**: November 8, 2025  
**Version**: 2.0.0
