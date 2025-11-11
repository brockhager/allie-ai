#!/usr/bin/env python3
"""
Test the quick-topics research endpoint
"""

import requests
import json
import time

def test_quick_topics():
    """Test the quick-topics research endpoint"""
    print("=" * 60)
    print("Testing Quick-Topics Research Endpoint")
    print("=" * 60)

    # Test data
    test_topics = ["election in usa", "artificial intelligence"]

    payload = {
        "topics": test_topics
    }

    print(f"\nTesting with topics: {test_topics}")
    print(f"Payload: {json.dumps(payload, indent=2)}")

    try:
        # Make request to the endpoint
        url = "http://localhost:8001/api/learning/quick-topics"
        print(f"\nMaking POST request to: {url}")

        response = requests.post(url, json=payload, timeout=60)

        print(f"Response status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print(f"\nResponse data:")
            print(json.dumps(result, indent=2))

            # Check results
            topics_processed = result.get("topics_processed", 0)
            successful = result.get("successful", 0)
            total_facts = result.get("total_facts_learned", 0)

            print(f"\nSummary:")
            print(f"  Topics processed: {topics_processed}")
            print(f"  Successful: {successful}")
            print(f"  Total facts learned: {total_facts}")

            if total_facts > 0:
                print("✓ SUCCESS: Facts were learned!")
                return True
            else:
                print("✗ FAILURE: No facts were learned")
                return False
        else:
            print(f"✗ HTTP Error: {response.status_code}")
            print(f"Response: {response.text}")
            return False

    except requests.exceptions.ConnectionError:
        print("✗ Connection Error: Server not running")
        print("Please start the server first with: python server.py")
        return False
    except Exception as e:
        print(f"✗ Test failed with exception: {e}")
        return False

if __name__ == "__main__":
    success = test_quick_topics()
    exit(0 if success else 1)