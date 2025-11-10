#!/usr/bin/env python3
"""
Manual test script to verify server endpoints work correctly
"""
import asyncio
import httpx
import json

async def test_endpoints():
    """Test both problematic endpoints"""
    
    print("🔍 Testing server endpoints...")
    
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            # Test 1: Facts endpoint
            print("\n1. Testing /api/facts endpoint...")
            response = await client.get("http://127.0.0.1:8001/api/facts?limit=3")
            print(f"Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"✓ Facts loaded: {data.get('total', 0)} total, {len(data.get('facts', []))} returned")
                if data.get('facts'):
                    print(f"  Sample fact: {data['facts'][0]['fact'][:100]}...")
            else:
                print(f"✗ Error: {response.text}")
                
        except Exception as e:
            print(f"✗ Facts endpoint failed: {e}")
        
        try:
            # Test 2: Quick topics endpoint
            print("\n2. Testing /api/learning/quick-topics endpoint...")
            payload = {"topics": ["python", "programming"]}
            response = await client.post(
                "http://127.0.0.1:8001/api/learning/quick-topics",
                json=payload,
                timeout=60
            )
            print(f"Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                successful = data.get('successful', 0)
                total_facts = data.get('total_facts_learned', 0)
                print(f"✓ Topics processed: {data.get('topics_processed', 0)}")
                print(f"✓ Successful: {successful}")
                print(f"✓ Facts learned: {total_facts}")
                
                if data.get('results'):
                    for result in data['results']:
                        status = "✓" if result.get('success') else "✗"
                        topic = result.get('topic', 'unknown')
                        facts = result.get('facts_learned', 0)
                        error = result.get('error', '')
                        print(f"  {status} {topic}: {facts} facts {error}")
            else:
                print(f"✗ Error: {response.text}")
                
        except Exception as e:
            print(f"✗ Quick topics endpoint failed: {e}")

if __name__ == "__main__":
    print("📡 Make sure server is running on http://127.0.0.1:8001")
    print("   Run: cd backend && python server.py")
    print("   Then run this test in another terminal")
    input("\nPress Enter when server is ready...")
    
    asyncio.run(test_endpoints())