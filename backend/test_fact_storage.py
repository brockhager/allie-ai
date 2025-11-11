import asyncio
import sys
sys.path.insert(0, '.')

# Test the query functions directly
from server import query_duckduckgo, query_wikidata, query_dbpedia
import httpx

async def test_fact_storage():
    print('Testing fact extraction and storage...')

    async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as http_client:
        # Test DBpedia (which we know returns results)
        dbpedia_result = await query_dbpedia(http_client, 'election in usa')
        print(f'DBpedia results: {len(dbpedia_result.get("results", []))}')

        if dbpedia_result.get('results'):
            for i, result in enumerate(dbpedia_result['results'][:2]):
                text = result.get('text', '')
                print(f'  Result {i+1}: {text[:100]}...')

                # Test if this would be stored
                if text and len(text) > 10:
                    print(f'    ✓ Would store: {len(text)} chars, confidence: {result.get("confidence", 0)}')
                else:
                    print(f'    ✗ Would skip: too short or empty')

if __name__ == "__main__":
    asyncio.run(test_fact_storage())