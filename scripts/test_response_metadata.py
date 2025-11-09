#!/usr/bin/env python3
"""
Test script to verify source URLs and confidence scores are included in responses
"""

import requests
import json
import time

def test_generate_endpoint():
    """Test the modified generate endpoint"""

    test_prompts = [
        'What is the capital of France?',
        'Who are you?',
        'What is 2+2?'
    ]

    print("Testing Allie Generate Endpoint with Source URLs & Confidence Scores")
    print("=" * 70)

    for prompt in test_prompts:
        try:
            print(f"\nTesting prompt: '{prompt}'")
            print("-" * 50)

            response = requests.post('http://localhost:8001/api/generate',
                                   json={'prompt': prompt},
                                   timeout=30)

            if response.status_code == 200:
                result = response.json()
                text = result.get('text', '')

                # Extract the source/confidence section
                if '---' in text:
                    parts = text.split('---')
                    main_response = parts[0].strip()
                    metadata = parts[1].strip() if len(parts) > 1 else ""

                    print(f"Main response: {main_response[:100]}...")
                    print(f"Metadata section:\n{metadata}")
                else:
                    print(f"Full response: {text}")
                    print("WARNING: No metadata section found!")

            else:
                print(f"Error: HTTP {response.status_code}")
                print(f"Response: {response.text}")

        except Exception as e:
            print(f"Failed: {e}")

        print()

if __name__ == "__main__":
    test_generate_endpoint()