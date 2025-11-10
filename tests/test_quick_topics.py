#!/usr/bin/env python3
"""Test the new quick-topics endpoint"""

import requests
import json

def test_quick_topics():
    """Test the /api/learning/quick-topics endpoint"""
    try:
        print("Testing /api/learning/quick-topics endpoint...")

        response = requests.post(
            'http://localhost:8001/api/learning/quick-topics',
            json={'topics': ['Python programming']},
            timeout=30
        )

        print(f'Status: {response.status_code}')

        if response.status_code == 200:
            result = response.json()
            print('✅ Success!')
            print(f'  Topics processed: {result.get("topics_processed")}')
            print(f'  Successful: {result.get("successful")}')
            print(f'  Total facts learned: {result.get("total_facts_learned")}')

            if result.get("results"):
                print(f'  Results: {len(result["results"])} topics')
                for r in result["results"]:
                    print(f'    - {r["topic"]}: {r["facts_learned"]} facts')

        else:
            print(f'❌ Error: {response.text}')

    except requests.exceptions.ConnectionError:
        print("❌ Connection error: Is the server running on localhost:8001?")
    except Exception as e:
        print(f'❌ Error: {e}')

if __name__ == "__main__":
    test_quick_topics()