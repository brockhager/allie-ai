#!/usr/bin/env python3
"""
Quick Teach - Help Allie learn faster!

This script allows you to quickly teach Allie about multiple topics at once.
"""

import requests
import json
import sys

SERVER_URL = "http://localhost:8001"

def bulk_learn_facts(facts):
    """Teach Allie a list of facts directly"""
    try:
        response = requests.post(
            f"{SERVER_URL}/api/learning/bulk-learn",
            json={"facts": facts},
            timeout=30
        )
        response.raise_for_status()
        result = response.json()
        print(f"✓ Learned {result['results']['facts_learned']} facts")
        return result
    except Exception as e:
        print(f"✗ Error: {e}")
        return None

def quick_research_topics(topics):
    """Teach Allie about multiple topics quickly (researches each topic)"""
    try:
        print(f"🔍 Researching {len(topics)} topics...")
        response = requests.post(
            f"{SERVER_URL}/api/learning/quick-topics",
            json={"topics": topics},
            timeout=120
        )
        response.raise_for_status()
        result = response.json()
        print(f"\n✓ Research complete!")
        print(f"  Topics processed: {result['topics_processed']}")
        print(f"  Successful: {result['successful']}")
        print(f"  Total facts learned: {result['total_facts_learned']}")
        
        print("\nDetails:")
        for r in result['results']:
            if r['success']:
                print(f"  ✓ {r['topic']}: {r['facts_learned']} facts")
            else:
                print(f"  ✗ {r['topic']}: {r.get('error', 'Unknown error')}")
        
        return result
    except Exception as e:
        print(f"✗ Error: {e}")
        return None

def interactive_mode():
    """Interactive teaching mode"""
    print("=" * 60)
    print("🎓 Quick Teach - Help Allie Learn Faster!")
    print("=" * 60)
    print()
    print("Choose an option:")
    print("1. Teach facts directly (one per line)")
    print("2. Research topics (Allie will research and learn)")
    print("3. Quick topics (common knowledge areas)")
    print("4. Exit")
    print()
    
    choice = input("Enter choice (1-4): ").strip()
    
    if choice == "1":
        print("\nEnter facts (one per line, empty line to finish):")
        facts = []
        while True:
            fact = input("> ").strip()
            if not fact:
                break
            facts.append(fact)
        
        if facts:
            print(f"\nTeaching {len(facts)} facts...")
            bulk_learn_facts(facts)
        else:
            print("No facts provided.")
    
    elif choice == "2":
        print("\nEnter topics to research (one per line, empty line to finish):")
        topics = []
        while True:
            topic = input("> ").strip()
            if not topic:
                break
            topics.append(topic)
        
        if topics:
            quick_research_topics(topics)
        else:
            print("No topics provided.")
    
    elif choice == "3":
        print("\n📚 Teaching common knowledge topics...")
        common_topics = [
            "Solar System planets",
            "World War II",
            "Python programming language",
            "Artificial Intelligence",
            "Climate change",
            "United States presidents",
            "European countries and capitals",
            "Human anatomy basics",
            "Periodic table elements",
            "Famous scientists"
        ]
        quick_research_topics(common_topics)
    
    elif choice == "4":
        print("Goodbye!")
        sys.exit(0)
    
    else:
        print("Invalid choice.")

def main():
    if len(sys.argv) > 1:
        # Command line mode
        command = sys.argv[1]
        
        if command == "facts" and len(sys.argv) > 2:
            facts = sys.argv[2:]
            bulk_learn_facts(facts)
        
        elif command == "topics" and len(sys.argv) > 2:
            topics = sys.argv[2:]
            quick_research_topics(topics)
        
        elif command == "common":
            common_topics = [
                "Solar System planets",
                "World War II",
                "Python programming language",
                "Artificial Intelligence",
                "United States presidents",
                "European countries and capitals"
            ]
            quick_research_topics(common_topics)
        
        else:
            print("Usage:")
            print("  python quick_teach.py                          # Interactive mode")
            print("  python quick_teach.py facts <fact1> <fact2>... # Teach facts")
            print("  python quick_teach.py topics <topic1> <topic2> # Research topics")
            print("  python quick_teach.py common                   # Teach common knowledge")
    
    else:
        # Interactive mode
        try:
            while True:
                interactive_mode()
                print()
        except KeyboardInterrupt:
            print("\n\nGoodbye!")

if __name__ == "__main__":
    main()
