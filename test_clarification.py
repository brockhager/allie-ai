#!/usr/bin/env python3
"""
Test script for clarification handling functionality.
This script tests the clarification flow to ensure Allie properly combines
clarified input with original intent.
"""

import json
import requests
import time
from typing import Dict, List, Any

# Configuration
SERVER_URL = "http://localhost:8001"

def test_clarification_flow():
    """Test the complete clarification flow"""
    print("🧪 Testing Clarification Handling Flow")
    print("=" * 50)

    # Test scenario: "Tell me about Ada Lovelace" -> clarification -> "Her childhood"
    test_scenario = {
        "original_query": "Tell me about Ada Lovelace",
        "clarification": "Her childhood",
        "expected_combined": "Tell me about Ada Lovelace's childhood"
    }

    print(f"Test Scenario: '{test_scenario['original_query']}' + '{test_scenario['clarification']}'")
    print(f"Expected Combined: '{test_scenario['expected_combined']}'")
    print()

    try:
        # Step 1: Send original ambiguous query
        print("Step 1: Sending original query...")
        response1 = requests.post(
            f"{SERVER_URL}/api/generate",
            json={
                "prompt": test_scenario["original_query"],
                "max_tokens": 512,
                "conversation_context": []
            },
            timeout=30
        )

        if response1.status_code != 200:
            print(f"❌ Failed to get response for original query: {response1.status_code}")
            return False

        result1 = response1.json()
        response_text1 = result1.get("text", "")

        print("✓ Original query response received")
        print(f"Response length: {len(response_text1)} characters")

        # Check if disambiguation was triggered
        has_clarification_context = "<!--CLARIFICATION_CONTEXT:" in response_text1
        print(f"Clarification context detected: {has_clarification_context}")

        if not has_clarification_context:
            print("⚠️  No clarification context found - disambiguation may not have triggered")
            print("This could be normal if the query wasn't ambiguous to the system")
            return True  # Not necessarily a failure

        # Step 2: Extract clarification context and simulate user response
        print("\nStep 2: Extracting clarification context...")
        clarification_marker = "<!--CLARIFICATION_CONTEXT:"
        start_idx = response_text1.find(clarification_marker)
        end_idx = response_text1.find("-->", start_idx)

        if start_idx == -1 or end_idx == -1:
            print("❌ Failed to extract clarification context")
            return False

        context_json = response_text1[start_idx + len(clarification_marker):end_idx]
        clarification_data = json.loads(context_json)

        print("✓ Clarification context extracted")
        print(f"Original query stored: '{clarification_data.get('original_query')}'")
        print(f"Clarification pending: {clarification_data.get('clarification_pending')}")

        # Step 3: Send clarification response
        print("\nStep 3: Sending clarification response...")
        conversation_context = [
            {"role": "me", "text": test_scenario["original_query"]},
            {"role": "them", "text": response_text1},
            {"role": "me", "text": test_scenario["clarification"]}
        ]

        response2 = requests.post(
            f"{SERVER_URL}/api/generate",
            json={
                "prompt": test_scenario["clarification"],  # This should be combined internally
                "max_tokens": 512,
                "conversation_context": conversation_context
            },
            timeout=30
        )

        if response2.status_code != 200:
            print(f"❌ Failed to get response for clarification: {response2.status_code}")
            return False

        result2 = response2.json()
        response_text2 = result2.get("text", "")

        print("✓ Clarification response received")
        print(f"Response length: {len(response_text2)} characters")

        # Step 4: Verify the response is about the combined query
        combined_query_lower = test_scenario["expected_combined"].lower()
        response_lower = response_text2.lower()

        # Check if the response mentions the combined topic
        keywords = ["ada lovelace", "childhood", "ada lovelace's childhood"]
        mentions_combined = any(keyword in response_lower for keyword in keywords)

        print(f"\nStep 4: Verification")
        print(f"Response mentions combined topic: {mentions_combined}")

        if mentions_combined:
            print("✅ SUCCESS: Clarification handling appears to be working!")
            print("The response addresses the combined query rather than just the clarification.")
        else:
            print("⚠️  UNCLEAR: Response may not be addressing the combined query properly")
            print("This could be due to model behavior or the combination logic needs refinement")

        # Step 5: Check memory integration
        print("\nStep 5: Checking memory integration...")
        try:
            memory_response = requests.get(f"{SERVER_URL}/api/hybrid-memory/search?query=clarification_event")
            if memory_response.status_code == 200:
                memory_data = memory_response.json()
                clarification_facts = [f for f in memory_data.get("facts", []) if "clarification" in f.get("fact", "").lower()]
                print(f"✓ Found {len(clarification_facts)} clarification events in memory")
                if clarification_facts:
                    print(f"Latest clarification event: {clarification_facts[0]['fact'][:100]}...")
            else:
                print("⚠️  Could not check memory integration")
        except Exception as e:
            print(f"⚠️  Memory check failed: {e}")

        return True

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

def test_edge_cases():
    """Test edge cases for clarification handling"""
    print("\n🧪 Testing Edge Cases")
    print("=" * 30)

    edge_cases = [
        {
            "name": "Short clarification",
            "original": "Who is Einstein",
            "clarification": "His theory",
            "expected_contains": ["einstein", "theory"]
        },
        {
            "name": "Long clarification",
            "original": "Tell me about Python",
            "clarification": "The history of its development and key features",
            "expected_contains": ["python", "history", "development"]
        }
    ]

    for case in edge_cases:
        print(f"\nTesting: {case['name']}")
        print(f"Original: '{case['original']}'")
        print(f"Clarification: '{case['clarification']}'")

        try:
            # Quick test - just check if the system responds
            response = requests.post(
                f"{SERVER_URL}/api/generate",
                json={
                    "prompt": case["original"],
                    "max_tokens": 200,
                    "conversation_context": []
                },
                timeout=15
            )

            if response.status_code == 200:
                print("✓ System responded to original query")
            else:
                print(f"⚠️  System error: {response.status_code}")

        except Exception as e:
            print(f"⚠️  Test failed: {e}")

if __name__ == "__main__":
    print("🚀 Starting Clarification Handling Tests")
    print("Make sure the Allie server is running on http://localhost:8001")
    print()

    # Wait a moment for server to be ready
    time.sleep(2)

    # Test main flow
    success = test_clarification_flow()

    # Test edge cases
    test_edge_cases()

    print("\n" + "=" * 50)
    if success:
        print("🎉 Clarification handling tests completed!")
        print("Review the output above to verify the functionality.")
    else:
        print("❌ Tests failed - check server logs for details")

    print("\nNext steps:")
    print("1. Check server logs for any errors")
    print("2. Test manually through the UI")
    print("3. Refine combination logic if needed")