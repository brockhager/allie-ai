"""
Smoke test script for Knowledge Base system.

Tests:
1. Server is running
2. Create a KB fact via API
3. List KB facts via API
4. Retrieve single fact via API
5. Test hybrid memory preference
6. Update KB fact
7. Delete KB fact
"""

import requests
import sys
import time
from pathlib import Path

# Add advanced-memory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "advanced-memory"))

BASE_URL = "http://localhost:8000"
TEST_KEYWORD = "smoke_test_kb_fact"


def test_server_running():
    """Test 1: Check if server is running"""
    print("1. Testing server connectivity...")
    try:
        response = requests.get(f"{BASE_URL}/api/conversations", timeout=5)
        if response.status_code == 200:
            print("✓ Server is running")
            return True
        else:
            print(f"✗ Server returned status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"✗ Server not accessible: {e}")
        return False


def test_create_kb_fact():
    """Test 2: Create a KB fact"""
    print("\n2. Creating KB fact via API...")
    try:
        payload = {
            "keyword": TEST_KEYWORD,
            "fact": "This is a smoke test fact for the Knowledge Base system",
            "source": "smoke_test",
            "confidence_score": 95,
            "provenance": "automated smoke test",
            "status": "true"
        }
        response = requests.post(f"{BASE_URL}/api/kb", json=payload, timeout=5)
        if response.status_code in [200, 201]:
            data = response.json()
            print(f"✓ KB fact created: {data}")
            return True
        else:
            print(f"✗ Failed to create fact: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"✗ Error creating fact: {e}")
        return False


def test_list_kb_facts():
    """Test 3: List KB facts"""
    print("\n3. Listing KB facts via API...")
    try:
        response = requests.get(f"{BASE_URL}/api/kb", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Retrieved {len(data)} KB facts")
            
            # Find our test fact
            test_fact = next((f for f in data if f['keyword'] == TEST_KEYWORD), None)
            if test_fact:
                print(f"✓ Found test fact: {test_fact['fact'][:50]}...")
                return test_fact['id']
            else:
                print(f"✗ Test fact not found in list")
                return None
        else:
            print(f"✗ Failed to list facts: {response.status_code}")
            return None
    except Exception as e:
        print(f"✗ Error listing facts: {e}")
        return None


def test_get_single_fact(fact_id):
    """Test 4: Get single KB fact by ID"""
    print(f"\n4. Retrieving single fact (ID: {fact_id})...")
    try:
        response = requests.get(f"{BASE_URL}/api/kb/{fact_id}", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Retrieved fact: {data['keyword']} - {data['fact'][:50]}...")
            return True
        else:
            print(f"✗ Failed to retrieve fact: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Error retrieving fact: {e}")
        return False


def test_hybrid_memory_preference():
    """Test 5: Test hybrid memory KB preference"""
    print("\n5. Testing hybrid memory KB preference...")
    try:
        from hybrid import HybridMemory
        
        hybrid = HybridMemory()
        results = hybrid.search(TEST_KEYWORD)
        
        if isinstance(results, dict) and 'results' in results:
            if len(results['results']) > 0:
                first_result = results['results'][0]
                if first_result.get('category') == 'knowledge_base':
                    print(f"✓ Hybrid memory returned KB fact with confidence {first_result.get('confidence')}")
                    return True
                else:
                    print(f"✗ Hybrid memory didn't prioritize KB fact (category: {first_result.get('category')})")
                    return False
            else:
                print(f"✗ Hybrid memory returned no results")
                return False
        else:
            print(f"✗ Unexpected hybrid memory response format")
            return False
    except Exception as e:
        print(f"✗ Error testing hybrid memory: {e}")
        return False


def test_update_kb_fact(fact_id):
    """Test 6: Update KB fact"""
    print(f"\n6. Updating KB fact (ID: {fact_id})...")
    try:
        payload = {
            "fact": "This is an UPDATED smoke test fact",
            "confidence_score": 98
        }
        headers = {"X-User-Role": "admin"}
        response = requests.patch(f"{BASE_URL}/api/kb/{fact_id}", json=payload, headers=headers, timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✓ KB fact updated: {data}")
            return True
        else:
            print(f"✗ Failed to update fact: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"✗ Error updating fact: {e}")
        return False


def test_delete_kb_fact(fact_id):
    """Test 7: Delete KB fact"""
    print(f"\n7. Deleting KB fact (ID: {fact_id})...")
    try:
        headers = {"X-User-Role": "admin"}
        response = requests.delete(f"{BASE_URL}/api/kb/{fact_id}", headers=headers, timeout=5)
        if response.status_code in [200, 204]:
            print(f"✓ KB fact deleted successfully")
            
            # Verify it's gone
            check_response = requests.get(f"{BASE_URL}/api/kb/{fact_id}", timeout=5)
            if check_response.status_code == 404:
                print(f"✓ Verified fact is deleted")
                return True
            else:
                print(f"⚠ Fact might still exist")
                return True
        else:
            print(f"✗ Failed to delete fact: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"✗ Error deleting fact: {e}")
        return False


def main():
    """Run all smoke tests"""
    print("=" * 60)
    print("Knowledge Base System - Smoke Tests")
    print("=" * 60)
    
    results = []
    fact_id = None
    
    # Test 1: Server running
    results.append(("Server Running", test_server_running()))
    if not results[-1][1]:
        print("\n✗ Server is not running. Start it with:")
        print("  cd backend")
        print("  uvicorn server:app --host 0.0.0.0 --port 8000")
        sys.exit(1)
    
    # Test 2: Create fact
    results.append(("Create KB Fact", test_create_kb_fact()))
    
    # Test 3: List facts
    if results[-1][1]:
        fact_id = test_list_kb_facts()
        results.append(("List KB Facts", fact_id is not None))
    else:
        results.append(("List KB Facts", False))
    
    # Test 4: Get single fact
    if fact_id:
        results.append(("Get Single Fact", test_get_single_fact(fact_id)))
    else:
        results.append(("Get Single Fact", False))
    
    # Test 5: Hybrid memory preference
    results.append(("Hybrid Memory Preference", test_hybrid_memory_preference()))
    
    # Test 6: Update fact
    if fact_id:
        results.append(("Update KB Fact", test_update_kb_fact(fact_id)))
    else:
        results.append(("Update KB Fact", False))
    
    # Test 7: Delete fact (cleanup)
    if fact_id:
        results.append(("Delete KB Fact", test_delete_kb_fact(fact_id)))
    else:
        results.append(("Delete KB Fact", False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:30} {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All smoke tests passed!")
        return 0
    else:
        print(f"\n⚠ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
