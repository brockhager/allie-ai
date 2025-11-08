# Knowledge Sources Configuration

Allie AI now integrates with **10 different knowledge sources** for comprehensive information retrieval.

## Core Sources (No Configuration Required)

These sources work out of the box:

1. **DuckDuckGo** - Instant answers and quick facts
2. **Wikipedia** - Comprehensive encyclopedia articles
3. **Wikidata** - Structured knowledge graph
4. **DBpedia** - Semantic encyclopedia data extracted from Wikipedia
5. **OpenLibrary** - Books, authors, and literary information
6. **ConceptNet** - Semantic network of common knowledge and relationships

## Optional Sources (Require API Keys)

These sources provide additional data but need configuration:

### 7. Google Knowledge Graph API

**Setup:**
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing
3. Enable "Knowledge Graph Search API"
4. Create API credentials
5. Set environment variable:
   ```powershell
   $env:GOOGLE_KG_API_KEY = "your-api-key-here"
   ```

**Benefits:** High-quality entity information with descriptions and links

---

### 8. YAGO Knowledge Graph

**Setup:**
- Uses public SPARQL endpoint
- No API key required but may have rate limits
- Endpoint: `https://yago-knowledge.org/sparql`

**Benefits:** Large semantic knowledge base with multilingual support

---

### 9. Freebase

**Note:** Freebase was officially shut down by Google in 2016. This integration:
- Uses community-maintained endpoints
- May have limited availability
- Data is partially archived in Wikidata

**Benefits:** Historical knowledge graph data (if available)

---

### 10. Bing Web Search API

**Setup:**
1. Go to [Azure Portal](https://portal.azure.com/)
2. Create a "Bing Search v7" resource
3. Get your API key from "Keys and Endpoint"
4. Set environment variable:
   ```powershell
   $env:BING_API_KEY = "your-api-key-here"
   ```

**Benefits:** Comprehensive web search results

---

## Search Priority Order

When Allie searches for information, sources are consulted in this priority:

1. **Memory** (fastest - already stored facts)
2. **DuckDuckGo** (instant answers)
3. **Wikipedia** (comprehensive articles)
4. **Google KG** (if configured - high-quality entities)
5. **Wikidata** (structured data)
6. **DBpedia** (semantic facts)
7. **YAGO** (if available)
8. **ConceptNet** (relationships and common sense)
9. **OpenLibrary** (books and authors)
10. **Freebase** (if available)

## Performance Notes

- All sources are queried **in parallel** for speed
- Failed sources are handled gracefully (won't break queries)
- Query cleaning removes question words for better search results
- Results are synthesized with confidence scores

## Testing Your Configuration

After setting up API keys, restart the server and ask:
```
"what is the eiffel tower"
```

Check the logs to see which sources successfully returned results.
