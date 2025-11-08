# Knowledge Sources Configuration

Allie AI now integrates with **11 different knowledge sources** for comprehensive information retrieval.

## Core Sources (No Configuration Required)

These sources work out of the box:

1. **Nominatim (OpenStreetMap)** - Geographic distances and location data (NEW!)
2. **DuckDuckGo** - Instant answers and quick facts
3. **Wikipedia** - Comprehensive encyclopedia articles
4. **Wikidata** - Structured knowledge graph
5. **DBpedia** - Semantic encyclopedia data extracted from Wikipedia
6. **OpenLibrary** - Books, authors, and literary information
7. **ConceptNet** - Semantic network of common knowledge and relationships

## Optional Sources (Require API Keys)

These sources provide additional data but need configuration:

### 8. Google Knowledge Graph API

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

### 9. YAGO Knowledge Graph

**Setup:**
- Uses public SPARQL endpoint
- No API key required but may have rate limits
- Endpoint: `https://yago-knowledge.org/sparql`

**Benefits:** Large semantic knowledge base with multilingual support

---

### 10. Freebase

**Note:** Freebase was officially shut down by Google in 2016. This integration:
- Uses community-maintained endpoints
- May have limited availability
- Data is partially archived in Wikidata

**Benefits:** Historical knowledge graph data (if available)

---

### 11. Bing Web Search API

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

1. **Nominatim** (for distance queries - highest confidence 0.99)
2. **Memory** (fastest - already stored facts)
3. **DuckDuckGo** (instant answers)
4. **Wikipedia** (comprehensive articles)
5. **Google KG** (if configured - high-quality entities)
6. **Wikidata** (structured data)
7. **DBpedia** (semantic facts)
8. **YAGO** (if available)
9. **ConceptNet** (relationships and common sense)
10. **OpenLibrary** (books and authors)
11. **Freebase** (if available)

## Special Features

### Distance Calculations (Nominatim)

Nominatim automatically detects and handles distance queries:
- "how far is it from X to Y"
- "distance from X to Y"  
- "what is the distance between X and Y"

Returns:
- Distance in miles and kilometers
- Estimated driving time
- Full location names
- Very high confidence (0.99)

Example:
```
Q: "how far is it from san diego to yuma az"
A: "The distance from San Diego, California to Yuma, Arizona is approximately 
    172.3 miles (277.3 km). Estimated driving time at 60 mph: 2h 52m"
```

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
