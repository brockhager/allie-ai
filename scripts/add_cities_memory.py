#!/usr/bin/env python3
"""
Script to add city information to Allie's memory
"""

import requests
import json
import time

def add_memory_fact(fact, importance=0.8, category="geography"):
    """Add a fact to Allie's memory"""
    url = "http://localhost:8001/api/memory/add"
    data = {
        "fact": fact,
        "importance": importance,
        "category": category
    }

    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            print(f"✓ Added: {fact[:50]}...")
            return True
        else:
            print(f"✗ Failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def main():
    # City data by state
    cities_data = {
        "Arizona": [
            "Phoenix", "Tucson", "Mesa", "Chandler", "Glendale",
            "Scottsdale", "Gilbert", "Tempe", "Peoria", "Surprise",
            "Yuma", "Avondale", "Goodyear", "Flagstaff", "Casa Grande"
        ],
        "California": [
            "Los Angeles", "San Diego", "San Jose", "San Francisco", "Fresno",
            "Sacramento", "Long Beach", "Oakland", "Bakersfield", "Anaheim",
            "Santa Ana", "Riverside", "Stockton", "Chula Vista", "Irvine"
        ],
        "Colorado": [
            "Denver", "Colorado Springs", "Aurora", "Fort Collins", "Lakewood",
            "Thornton", "Arvada", "Westminster", "Pueblo", "Centennial",
            "Boulder", "Greeley", "Longmont", "Loveland", "Grand Junction"
        ]
    }

    print("Adding city information to Allie's memory...")

    for state, cities in cities_data.items():
        fact = f"Top 15 biggest cities in {state}: {', '.join(cities)}"
        success = add_memory_fact(fact, importance=0.9, category="geography")
        if not success:
            print(f"Failed to add {state} cities")
        time.sleep(0.1)  # Small delay between requests

    print("\nMemory addition complete!")

if __name__ == "__main__":
    main()