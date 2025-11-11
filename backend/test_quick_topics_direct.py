import asyncio
import sys
sys.path.insert(0, '.')

from server import quick_topics_research
import json

async def test_quick_topics_direct():
    """Test the quick_topics_research function directly"""
    print("=" * 60)
    print("Testing Quick-Topics Research Function Directly")
    print("=" * 60)

    # Test payload
    payload = {
        "topics": ["election in usa", "artificial intelligence"]
    }

    print(f"Test payload: {json.dumps(payload, indent=2)}")

    try:
        # Call the function directly (simulating the endpoint)
        result = await quick_topics_research(payload)

        print(f"\nResponse:")
        print(json.dumps(result, indent=2))

        # Check results
        topics_processed = result.get("topics_processed", 0)
        successful = result.get("successful", 0)
        total_facts_learned = result.get("total_facts_learned", 0)

        print(f"\nSummary:")
        print(f"  Topics processed: {topics_processed}")
        print(f"  Successful: {successful}")
        print(f"  Total facts learned: {total_facts_learned}")

        if total_facts_learned > 0:
            print("✓ SUCCESS: Facts were learned!")
            return True
        else:
            print("✗ FAILURE: No facts were learned")
            return False

    except Exception as e:
        print(f"✗ Function call failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_quick_topics_direct())
    exit(0 if success else 1)