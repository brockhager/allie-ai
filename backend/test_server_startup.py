"""
Quick test to verify server starts with MySQL integration
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

def test_server_startup():
    """Test that server components initialize with MySQL"""
    print("=" * 60)
    print("Testing Server Startup with MySQL")
    print("=" * 60)
    
    print("\n1. Testing HybridMemory initialization...")
    from hybrid import HybridMemory
    try:
        memory = HybridMemory()
        print(f"   ✓ HybridMemory initialized: {memory}")
        
        stats = memory.get_statistics()
        print(f"   ✓ Stats retrieved from MySQL:")
        print(f"     - Total facts: {stats.get('total_facts', 0)}")
        print(f"     - Sources: {stats.get('sources', {})}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False
    
    print("\n2. Testing AutomaticLearner...")
    try:
        from automatic_learner import AutomaticLearner
        learner = AutomaticLearner(memory)
        print(f"   ✓ AutomaticLearner initialized")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False
    
    print("\n3. Testing retrieval system...")
    try:
        from sources.retrieval import search_all_sources
        print(f"   ✓ Search system available")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False
    
    print("\n4. Verifying MySQL connection...")
    from db import MemoryDB
    try:
        db = MemoryDB()
        if db.connection:
            print(f"   ✓ MySQL connected")
            
            # Quick test
            test_fact = db.get_fact("test_startup")
            if test_fact:
                print(f"   Found existing test fact")
            else:
                print(f"   No existing test facts (clean database)")
            
            db.close()
        else:
            print(f"   ✗ MySQL connection failed")
            return False
    except Exception as e:
        print(f"   ✗ MySQL error: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✓ All server components ready!")
    print("MySQL integration is active and functional.")
    print("=" * 60)
    print("\nYou can now start the server with:")
    print("  python server.py")
    print("\nOr use the batch file:")
    print("  run_server.bat")
    
    return True

if __name__ == "__main__":
    success = test_server_startup()
    sys.exit(0 if success else 1)
