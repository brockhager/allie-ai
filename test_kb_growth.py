#!/usr/bin/env python3
"""Test script to send factual conversations to Allie server and monitor KB growth."""

import requests
import time
import json

def send_conversation(prompt):
    """Send a conversation to the Allie server."""
    url = "http://localhost:8000/api/conversations"
    payload = {"prompt": prompt}

    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Conversation sent: {prompt[:50]}...")
            print(f"   Response: {result['response'][:100]}...")
            return True
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False

def main():
    print("🧪 TESTING KB AUTOMATIC GROWTH WITH REAL CONVERSATIONS")
    print("=" * 60)

    # Factual conversations that should trigger learning
    conversations = [
        "Marie Curie was born in Warsaw, Poland in 1867 and was a pioneering physicist and chemist.",
        "The Great Wall of China was built over many dynasties, with most construction during the Ming dynasty.",
        "Leonardo da Vinci painted the Mona Lisa between 1503 and 1519, and it hangs in the Louvre Museum in Paris.",
        "The Amazon River is the longest river in the world, flowing through South America.",
        "Shakespeare wrote Romeo and Juliet around 1595, and it's one of his most famous tragedies."
    ]

    print(f"Sending {len(conversations)} factual conversations...")

    for i, prompt in enumerate(conversations, 1):
        print(f"\n💬 Conversation {i}:")
        success = send_conversation(prompt)
        if success:
            # Wait a bit for learning to process
            time.sleep(2)

    print("\n" + "=" * 60)
    print("✅ All conversations sent!")
    print("⏳ Waiting for learning pipeline to process...")
    time.sleep(5)  # Give time for worker to process

    print("\n🔍 CHECKING KB GROWTH...")
    print("Run: python tests/kb_diagnostic.py")
    print("Expected: KB should have grown by several entries")

if __name__ == "__main__":
    main()