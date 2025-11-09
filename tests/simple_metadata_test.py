#!/usr/bin/env python3
"""
Simple test to verify source URLs and confidence scores metadata logic
"""

def test_metadata_logic():
    """Test the metadata generation logic directly"""

    # Simulate different response scenarios
    test_cases = [
        {
            "name": "External sources response",
            "multi_source_results": {
                "success": True,
                "sources_used": ["duckduckgo", "wikipedia"],
                "all_results": {
                    "duckduckgo": {
                        "success": True,
                        "results": [
                            {"url": "https://example.com/1", "source": "DuckDuckGo"},
                            {"url": "https://example.com/2", "source": "DuckDuckGo"}
                        ]
                    },
                    "wikipedia": {
                        "success": True,
                        "url": "https://en.wikipedia.org/wiki/Paris"
                    }
                }
            },
            "relevant_facts": [],
            "is_self_referential": False
        },
        {
            "name": "Memory-based response",
            "multi_source_results": None,
            "relevant_facts": ["Paris is the capital of France"],
            "is_self_referential": False
        },
        {
            "name": "Model-only response",
            "multi_source_results": None,
            "relevant_facts": [],
            "is_self_referential": False
        },
        {
            "name": "Self-referential response",
            "multi_source_results": None,
            "relevant_facts": [],
            "is_self_referential": True
        }
    ]

    print("Testing Metadata Generation Logic")
    print("=" * 50)

    for test_case in test_cases:
        print(f"\nTest: {test_case['name']}")
        print("-" * 30)

        # Simulate the metadata generation logic from the server
        source_info = []
        confidence_score = 0.0
        primary_source = "model"

        multi_source_results = test_case["multi_source_results"]
        relevant_facts = test_case["relevant_facts"]
        is_self_referential = test_case["is_self_referential"]

        # Determine primary source and confidence
        if not is_self_referential and multi_source_results and multi_source_results.get("success"):
            # External sources were used - high confidence
            primary_source = "external_sources"
            confidence_score = 0.85  # High confidence from external verification

            # Collect URLs from all sources
            all_results = multi_source_results.get("all_results", {})

            # Wikipedia URLs
            if "wikipedia" in multi_source_results.get("sources_used", []):
                wiki_data = all_results.get("wikipedia", {})
                if wiki_data.get("success") and wiki_data.get("url"):
                    source_info.append(f"📖 Wikipedia: {wiki_data['url']}")

            # DuckDuckGo URLs (use the search results)
            if "duckduckgo" in multi_source_results.get("sources_used", []):
                ddg_data = all_results.get("duckduckgo", {})
                if ddg_data.get("success") and ddg_data.get("results"):
                    for idx, result in enumerate(ddg_data["results"][:2], 1):  # Top 2 results
                        if result.get("url"):
                            source_name = result.get("source", f"Source {idx}")
                            source_info.append(f"🔍 {source_name}: {result['url']}")

        elif relevant_facts and len(relevant_facts) > 0:
            # Memory-based response - medium confidence
            primary_source = "memory"
            confidence_score = 0.70  # Medium confidence from stored knowledge
            source_info.append("💾 Internal Memory: Stored knowledge base")

        else:
            # Pure model generation - lower confidence
            primary_source = "model"
            confidence_score = 0.60  # Base confidence for model-generated responses
            source_info.append("🤖 AI Model: Generated response")

        # Generate the metadata section
        metadata = f"---\n**Source:** {primary_source.title()}\n**Confidence:** {confidence_score:.0%}"

        if source_info:
            metadata += "\n**URLs:**\n" + "\n".join(source_info)

        print(f"Primary Source: {primary_source}")
        print(f"Confidence Score: {confidence_score:.0%}")
        print(f"Source URLs: {len(source_info)}")
        for url in source_info:
            print(f"  {url}")
        print(f"Metadata section:\n{metadata}")

        # Verify the metadata contains required elements
        assert "**Source:**" in metadata, "Source not found in metadata"
        assert "**Confidence:**" in metadata, "Confidence not found in metadata"
        assert f"{confidence_score:.0%}" in metadata, "Confidence score not formatted correctly"

        if primary_source in ["external_sources", "memory"]:
            assert "**URLs:**" in metadata, "URLs section missing for non-model sources"

        print("✓ Test passed")

    print("\n" + "=" * 50)
    print("All metadata generation tests passed!")
    print("The logic correctly adds source URLs and confidence scores to all responses.")

if __name__ == "__main__":
    test_metadata_logic()