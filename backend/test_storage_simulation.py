import asyncio
import sys
sys.path.insert(0, '.')

from server import query_duckduckgo, query_wikidata, query_dbpedia, hybrid_memory
import httpx

async def test_fact_storage_simulation():
    print('Testing fact storage simulation (like quick-topics endpoint)...')

    async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as http_client:
        topic = 'election in usa'

        # Query multiple sources in parallel (like the endpoint does)
        queries = await asyncio.gather(
            query_duckduckgo(http_client, topic),
            query_wikidata(http_client, topic),
            query_dbpedia(http_client, topic),
            return_exceptions=True
        )

        print(f'Query results: {len(queries)} total')

        # Collect successful results (like the endpoint does)
        external_results = []
        for query_result in queries:
            if isinstance(query_result, Exception):
                print(f'Query failed: {query_result}')
                continue
            if query_result.get("success") and query_result.get("results"):
                external_results.extend(query_result["results"])
                print(f'Added {len(query_result["results"])} results from {query_result.get("query", "unknown")}')

        print(f'Total external results: {len(external_results)}')

        # Extract facts from results and store them (like the endpoint does)
        facts_learned = 0
        stored_facts = []

        for result in external_results:
            text = result.get("text", "").strip()
            print(f'Processing result: "{text[:50]}..." (len={len(text)})')

            if text and len(text) > 10:  # Minimum length check
                try:
                    print(f'  Attempting to store fact...')
                    # Store fact in hybrid memory
                    fact_result = hybrid_memory.add_fact(
                        fact=text,
                        category=topic,  # Use topic as category
                        confidence=result.get("confidence", 0.7),
                        source=result.get("source", "external_research")
                    )
                    print(f'  Storage result: {fact_result}')
                    facts_learned += 1
                    stored_facts.append({
                        "fact": text,
                        "source": result.get("source", "unknown"),
                        "confidence": result.get("confidence", 0.7)
                    })
                except Exception as e:
                    print(f'  ✗ Failed to store fact: {e}')
            else:
                print(f'  ✗ Skipped: too short or empty')

        print(f'\nSummary:')
        print(f'  Facts learned: {facts_learned}')
        print(f'  Stored facts: {len(stored_facts)}')

        # Check memory statistics
        stats = hybrid_memory.get_statistics()
        print(f'  Memory stats after: total_facts={stats.get("total_facts", 0)}')

if __name__ == "__main__":
    asyncio.run(test_fact_storage_simulation())