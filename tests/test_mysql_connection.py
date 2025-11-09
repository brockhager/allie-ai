#!/usr/bin/env python3
"""Quick test of MySQL memory connection"""

import sys
from pathlib import Path

# Add advanced-memory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "advanced-memory"))

from db import MemoryDB

try:
    # Connect to your existing database
    db = MemoryDB(
        host='localhost',
        user='allie',
        password='StrongPassword123!',  # Your existing password
        database='allie_memory'
    )
    
    print("✓ Connected to MySQL successfully")
    
    # Test adding a fact
    db.add_fact(
        keyword="test_connection",
        fact="Connection test successful",
        source="setup_test"
    )
    print("✓ Added test fact")
    
    # Retrieve it
    fact = db.get_fact("test_connection")
    if fact:
        print(f"✓ Retrieved fact: {fact}")
    
    # Get timeline
    timeline = db.timeline()
    print(f"✓ Timeline has {len(timeline)} facts")
    
    print("\n✅ MySQL memory system is working!")
    print("\nYour existing database is ready to use.")
    print("The basic MemoryDB class is active with methods:")
    print("  - add_fact(keyword, fact, source)")
    print("  - get_fact(keyword)")
    print("  - update_fact(keyword, new_fact, source)")
    print("  - delete_fact(keyword)")
    print("  - timeline()")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("\nPlease check:")
    print("1. MySQL service is running")
    print("2. Database 'allie_memory' exists")
    print("3. Credentials match your setup")
