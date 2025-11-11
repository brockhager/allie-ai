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
                print(f'Answer: "{data.get("Answer", "NOT_FOUND")}"')
                print(f'AbstractText: "{data.get("AbstractText", "NOT_FOUND")}"')
                print(f'RelatedTopics count: {len(data.get("RelatedTopics", []))}')

                # Check if RelatedTopics has actual content
                if data.get("RelatedTopics"):
                    for i, topic in enumerate(data["RelatedTopics"][:3]):
                        text = topic.get("Text", "")
                        if text:
                            print(f'  Topic {i+1}: "{text[:100]}..."')
                        else:
                            print(f'  Topic {i+1}: No Text field')

                # Check Results array
                if data.get("Results"):
                    print(f'Results count: {len(data["Results"])}')
                    for i, result in enumerate(data["Results"][:2]):
                        text = result.get("Text", "")
                        if text:
                            print(f'  Result {i+1}: "{text[:100]}..."')
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
                print(f'Full response keys: {list(data.keys())}')
                print(f'@text: "{data.get("@text", "NOT_FOUND")}"')
                print(f'@confidence: {data.get("@confidence", "NOT_FOUND")}')
                print(f'@support: {data.get("@support", "NOT_FOUND")}')

                # Check if this is actually an annotation response
                if '@text' in data and len(data['@text']) > len(topic):
                    print('This appears to be an annotation response with extracted text!')
                    print(f'Extracted text length: {len(data["@text"])}')
            else:
                print(f'Error: {response.text[:200]}...')
        except Exception as e:
            print(f'Exception: {e}')

if __name__ == "__main__":
    asyncio.run(test_queries_debug())