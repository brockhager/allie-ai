import asyncio
from server import query_duckduckgo, query_wikidata, query_dbpedia
import httpx

async def test_queries():
    topic = 'election in usa'

    async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as http_client:
        print(f'Testing queries for: {topic}')

        # Test DuckDuckGo
        ddg = await query_duckduckgo(http_client, topic)
        print(f'DuckDuckGo: success={ddg.get("success")}, results={len(ddg.get("results", []))}')
        if ddg.get('results'):
            print(f'  Sample result: {ddg["results"][0].get("text", "")[:100]}...')

        # Test Wikidata
        wiki = await query_wikidata(http_client, topic)
        print(f'Wikidata: success={wiki.get("success")}, results={len(wiki.get("results", []))}')
        if wiki.get('results'):
            print(f'  Sample result: {wiki["results"][0].get("text", "")[:100]}...')

        # Test DBpedia
        dbpedia = await query_dbpedia(http_client, topic)
        print(f'DBpedia: success={dbpedia.get("success")}, results={len(dbpedia.get("results", []))}')
        if dbpedia.get('results'):
            print(f'  Sample result: {dbpedia["results"][0].get("text", "")[:100]}...')

if __name__ == "__main__":
    asyncio.run(test_queries())