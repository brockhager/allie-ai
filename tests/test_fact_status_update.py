#!/usr/bin/env python3
"""
Test script to verify fact status updates work correctly

This tests:
1. PATCH /api/facts/{fact_id} endpoint exists
2. Status updates are persisted to database
3. Invalid status values are rejected
4. Confidence score updates work
"""

import requests
import json
import sys
import time

BASE_URL = "http://localhost:8001"

def test_fact_status_update():
    """Test fact status update functionality"""
    print("=" * 60)
    print("Testing Fact Status Update")
    print("=" * 60)
    
    # Step 1: Get list of facts
    print("\n1. Fetching facts from database...")
    try:
        response = requests.get(f"{BASE_URL}/api/facts?limit=5")
        if not response.ok:
            print(f"❌ Failed to fetch facts: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
        
        data = response.json()
        if data.get("status") != "success" or not data.get("facts"):
            print(f"❌ No facts found in database")
            print(f"   Add some facts first using the UI or API")
            return False
        
        facts = data["facts"]
        print(f"✅ Found {len(facts)} facts")
        
        # Pick the first fact with a valid ID
        test_fact = None
        for fact in facts:
            if fact.get("id") and isinstance(fact["id"], int):
                test_fact = fact
                break
        
        if not test_fact:
            print(f"❌ No valid fact IDs found")
            return False
        
        fact_id = test_fact["id"]
        original_status = test_fact.get("status", "not_verified")
        original_confidence = test_fact.get("confidence_score", 50)
        
        print(f"\n   Testing with fact ID: {fact_id}")
        print(f"   Original status: {original_status}")
        print(f"   Original confidence: {original_confidence}")
        
    except Exception as e:
        print(f"❌ Error fetching facts: {e}")
        return False
    
    # Step 2: Test valid status update
    print(f"\n2. Testing valid status update (true)...")
    try:
        response = requests.patch(
            f"{BASE_URL}/api/facts/{fact_id}",
            json={"status": "true"},
            headers={"Content-Type": "application/json"}
        )
        
        if not response.ok:
            print(f"❌ PATCH request failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
        
        data = response.json()
        if data.get("status") != "success":
            print(f"❌ Update failed: {data}")
            return False
        
        print(f"✅ Status updated successfully")
        print(f"   New status: {data.get('new_status')}")
        
    except Exception as e:
        print(f"❌ Error updating status: {e}")
        return False
    
    # Step 3: Verify status was updated in database
    print(f"\n3. Verifying status was persisted to database...")
    time.sleep(0.5)  # Small delay to ensure DB write completes
    
    try:
        response = requests.get(f"{BASE_URL}/api/facts?limit=100")
        data = response.json()
        facts = data.get("facts", [])
        
        updated_fact = next((f for f in facts if f.get("id") == fact_id), None)
        if not updated_fact:
            print(f"❌ Fact {fact_id} not found after update")
            return False
        
        if updated_fact.get("status") != "true":
            print(f"❌ Status was not persisted: {updated_fact.get('status')}")
            return False
        
        print(f"✅ Status persisted correctly in database")
        
    except Exception as e:
        print(f"❌ Error verifying update: {e}")
        return False
    
    # Step 4: Test confidence score update
    print(f"\n4. Testing confidence score update (75)...")
    try:
        response = requests.patch(
            f"{BASE_URL}/api/facts/{fact_id}",
            json={"confidence_score": 75},
            headers={"Content-Type": "application/json"}
        )
        
        if not response.ok:
            print(f"❌ PATCH request failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
        
        data = response.json()
        if data.get("status") != "success":
            print(f"❌ Update failed: {data}")
            return False
        
        print(f"✅ Confidence updated successfully")
        print(f"   New confidence: {data.get('new_confidence_score')}")
        
    except Exception as e:
        print(f"❌ Error updating confidence: {e}")
        return False
    
    # Step 5: Test invalid status value (should fail)
    print(f"\n5. Testing invalid status value (should be rejected)...")
    try:
        response = requests.patch(
            f"{BASE_URL}/api/facts/{fact_id}",
            json={"status": "invalid_status"},
            headers={"Content-Type": "application/json"}
        )
        
        if response.ok:
            print(f"❌ Invalid status was accepted (should have failed)")
            return False
        
        print(f"✅ Invalid status correctly rejected (HTTP {response.status_code})")
        
    except Exception as e:
        print(f"❌ Error testing invalid status: {e}")
        return False
    
    # Step 6: Test invalid confidence value (should fail)
    print(f"\n6. Testing invalid confidence value (should be rejected)...")
    try:
        response = requests.patch(
            f"{BASE_URL}/api/facts/{fact_id}",
            json={"confidence_score": 150},  # Out of range
            headers={"Content-Type": "application/json"}
        )
        
        if response.ok:
            print(f"❌ Invalid confidence was accepted (should have failed)")
            return False
        
        print(f"✅ Invalid confidence correctly rejected (HTTP {response.status_code})")
        
    except Exception as e:
        print(f"❌ Error testing invalid confidence: {e}")
        return False
    
    # Step 7: Restore original values
    print(f"\n7. Restoring original values...")
    try:
        response = requests.patch(
            f"{BASE_URL}/api/facts/{fact_id}",
            json={
                "status": original_status,
                "confidence_score": original_confidence
            },
            headers={"Content-Type": "application/json"}
        )
        
        if response.ok:
            print(f"✅ Original values restored")
        else:
            print(f"⚠️  Warning: Failed to restore original values")
        
    except Exception as e:
        print(f"⚠️  Warning: Error restoring original values: {e}")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    print("\nFact Status Update Test")
    print("Ensure the server is running on http://localhost:8001\n")
    
    try:
        success = test_fact_status_update()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
