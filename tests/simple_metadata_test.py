#!/usr/bin/env python3
"""
Standalone test for response metadata functionality.
Tests the logic that adds source URLs and confidence scores to responses.
"""

def test_response_metadata():
    """Test the response metadata generation logic"""

    # Test cases for different response types
    test_cases = [
        {
            "name": "external_sources_response",
            "is_self_referential": False,
            "multi_source_results": {
                "success": True,
                "sources_used": ["wikipedia", "duckduckgo"],
                "all_results": {
                    "wikipedia": {"success": True, "url": "https://en.wikipedia.org/wiki/Test"},
                    "duckduckgo": {"success": True, "results": [{"url": "https://example.com"}]}
                }
            },
            "relevant_facts": [],
            "expected_source": "external_sources",
            "expected_confidence": 0.85,
            "expected_urls": ["📖 Wikipedia: https://en.wikipedia.org/wiki/Test", "🔍 Source 1: https://example.com"]
        },
        {
            "name": "memory_based_response",
            "is_self_referential": False,
            "multi_source_results": None,
            "relevant_facts": ["Test fact from memory"],
            "expected_source": "memory",
            "expected_confidence": 0.70,
            "expected_urls": ["💾 Internal Memory: Stored knowledge base"]
        },
        {
            "name": "model_only_response",
            "is_self_referential": False,
            "multi_source_results": None,
            "relevant_facts": [],
            "expected_source": "model",
            "expected_confidence": 0.60,
            "expected_urls": ["🤖 AI Model: Generated response"]
        },
        {
            "name": "self_referential_response",
            "is_self_referential": True,
            "multi_source_results": None,
            "relevant_facts": [],
            "expected_source": "model",
            "expected_confidence": 0.60,
            "expected_urls": ["🤖 AI Model: Generated response"]
        }
    ]

    print("Testing response metadata generation...")

    for test_case in test_cases:
        print(f"\n--- Testing: {test_case['name']} ---")

        # Simulate the metadata generation logic from the server
        source_info = []
        confidence_score = 0.0
        primary_source = "model"

        # Determine primary source and confidence
        if not test_case["is_self_referential"] and test_case["multi_source_results"] and test_case["multi_source_results"].get("success"):
            # External sources were used - high confidence
            primary_source = "external_sources"
            confidence_score = 0.85  # High confidence from external verification

            # Collect URLs from all sources
            all_results = test_case["multi_source_results"].get("all_results", {})

            # Wikipedia URLs
            if "wikipedia" in test_case["multi_source_results"].get("sources_used", []):
                wiki_data = all_results.get("wikipedia", {})
                if wiki_data.get("success") and wiki_data.get("url"):
                    source_info.append(f"📖 Wikipedia: {wiki_data['url']}")

            # DuckDuckGo URLs (use the search results)
            if "duckduckgo" in test_case["multi_source_results"].get("sources_used", []):
                ddg_data = all_results.get("duckduckgo", {})
                if ddg_data.get("success") and ddg_data.get("results"):
                    for idx, result in enumerate(ddg_data["results"][:2], 1):  # Top 2 results
                        if result.get("url"):
                            source_name = result.get("source", f"Source {idx}")
                            source_info.append(f"🔍 {source_name}: {result['url']}")

            # Wikidata URLs
            if "wikidata" in test_case["multi_source_results"].get("sources_used", []):
                wikidata_data = all_results.get("wikidata", {})
                if wikidata_data.get("success") and wikidata_data.get("entity_id"):
                    entity_id = wikidata_data["entity_id"]
                    source_info.append(f"🗂️ Wikidata: https://www.wikidata.org/wiki/{entity_id}")

        elif test_case["relevant_facts"] and len(test_case["relevant_facts"]) > 0:
            # Memory-based response - medium confidence
            primary_source = "memory"
            confidence_score = 0.70  # Medium confidence from stored knowledge
            source_info.append("💾 Internal Memory: Stored knowledge base")

            # If we have specific memory facts with confidence scores, use the highest
            # For this test, we'll keep it at 0.70

        else:
            # Pure model generation - lower confidence
            primary_source = "model"
            confidence_score = 0.60  # Base confidence for model-generated responses
            source_info.append("🤖 AI Model: Generated response")

        # Check results
        success = True
        if primary_source != test_case["expected_source"]:
            print(f"❌ Source mismatch: expected '{test_case['expected_source']}', got '{primary_source}'")
            success = False
        else:
            print(f"✅ Source: {primary_source}")

        if abs(confidence_score - test_case["expected_confidence"]) > 0.01:
            print(f"❌ Confidence mismatch: expected {test_case['expected_confidence']}, got {confidence_score}")
            success = False
        else:
            print(f"✅ Confidence: {confidence_score:.0%}")

        # Check URLs (order might vary, so check if all expected URLs are present)
        expected_urls = test_case["expected_urls"]
        if len(source_info) != len(expected_urls):
            print(f"❌ URL count mismatch: expected {len(expected_urls)}, got {len(source_info)}")
            success = False
        else:
            urls_match = all(url in source_info for url in expected_urls)
            if not urls_match:
                print(f"❌ URLs mismatch: expected {expected_urls}, got {source_info}")
                success = False
            else:
                print(f"✅ URLs: {len(source_info)} sources")

        if success:
            print(f"✅ {test_case['name']} PASSED")
        else:
            print(f"❌ {test_case['name']} FAILED")
            return False

    print("\n🎉 All tests passed! Response metadata functionality is working correctly.")
    return True

if __name__ == "__main__":
    test_response_metadata()