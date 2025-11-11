import asyncio
from server import query_duckduckgo, query_wikidata, query_dbpedia
import httpx
import json

async def test_queries_debug():
    topic = 'election in usa'

    async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as http_client:
        print(f'Testing queries for: {topic}')
        print('=' * 50)

        # Test DuckDuckGo
        print('\n--- DuckDuckGo ---')
        try:
            url = f"https://api.duckduckgo.com/?q={topic}&format=json&no_html=1&skip_disambig=1"
            response = await http_client.get(url)
            print(f'Status: {response.status_code}')
            if response.status_code == 200:
                data = response.json()
                print(f'Keys in response: {list(data.keys())}')
                if 'Answer' in data:
                    print(f'Answer: {data["Answer"][:200]}...')
                if 'AbstractText' in data:
                    print(f'AbstractText: {data["AbstractText"][:200]}...')
                if 'RelatedTopics' in data:
                    print(f'RelatedTopics count: {len(data["RelatedTopics"])}')
                    if data["RelatedTopics"]:
                        print(f'First topic: {data["RelatedTopics"][0].get("Text", "No Text")[:100]}...')
            else:
                print(f'Error: {response.text}')
        except Exception as e:
            print(f'Exception: {e}')

        # Test Wikidata
        print('\n--- Wikidata ---')
        try:
            search_url = f"https://www.wikidata.org/w/api.php?action=wbsearchentities&search={topic}&language=en&format=json&limit=3"
            response = await http_client.get(search_url)
            print(f'Status: {response.status_code}')
            if response.status_code == 200:
                data = response.json()
                print(f'Keys in response: {list(data.keys())}')
                if 'search' in data:
                    print(f'Search results: {len(data["search"])}')
                    for i, entity in enumerate(data["search"][:2]):
                        print(f'  {i+1}: {entity.get("label", "")} - {entity.get("description", "")}')
            else:
                print(f'Error: {response.text[:200]}...')
        except Exception as e:
            print(f'Exception: {e}')

        # Test DBpedia
        print('\n--- DBpedia ---')
        try:
            spotlight_url = f"https://api.dbpedia-spotlight.org/en/annotate?text={topic}&confidence=0.5&support=20"
            response = await http_client.get(spotlight_url, headers={"Accept": "application/json"})
            print(f'Status: {response.status_code}')
            if response.status_code == 200:
                data = response.json()
                print(f'Keys in response: {list(data.keys())}')
                if 'Resources' in data:
                    print(f'Resources count: {len(data["Resources"])}')
                    for i, resource in enumerate(data["Resources"][:2]):
                        print(f'  {i+1}: {resource.get("@surfaceForm", "")} (confidence: {resource.get("@similarityScore", "")})')
                else:
                    print('No Resources key found')
            else:
                print(f'Error: {response.text[:200]}...')
        except Exception as e:
            print(f'Exception: {e}')

if __name__ == "__main__":
    asyncio.run(test_queries_debug())