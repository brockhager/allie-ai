import asyncio
import httpx

async def test_sources():
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            'http://localhost:8001/api/generate',
            json={'prompt': 'What is the Eiffel Tower?'},
            timeout=30.0
        )
        text = resp.json()['text']
        print("Response:")
        print(text)
        print("\n" + "="*60)
        if "Sources:" in text:
            print("✓ Sources are present in response")
        else:
            print("✗ Sources are NOT present in response")

asyncio.run(test_sources())
