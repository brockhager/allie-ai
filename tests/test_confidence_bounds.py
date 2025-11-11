#!/usr/bin/env python3
"""
Test script to verify confidence score bounds and disambiguation functionality
"""

import sys
import os
from pathlib import Path

# Add the advanced-memory directory to the path
sys.path.insert(0, str(Path(__file__).parent / "advanced-memory"))

from disambiguation import DisambiguationEngine
from hybrid import HybridMemory

def test_confidence_bounds():
    """Test that confidence scores are properly bounded"""
    print("Testing confidence score bounds...")

    # Create disambiguation engine and clear cache
    engine = DisambiguationEngine()
    engine.cache = {}
    engine._save_cache()

    # Test case 1: Single low-credibility source should be capped at 50%
    print("\n1. Testing single low-credibility source (should be ≤50%)")
    results = [{
        "fact": "Python is a programming language",
        "source": "user",  # Low credibility (0.60)
        "confidence": 0.9,
        "status": "not_verified"
    }]

    disambiguation = engine.detect_ambiguity("python", results)
    interpretation = disambiguation["interpretations"][0]
    confidence = interpretation["confidence_score"]

    print(f"   Single user source confidence: {confidence} (raw), {confidence*100:.0f}%")
    assert confidence <= 0.5, f"Expected ≤0.5, got {confidence}"
    print("   ✅ PASS: Confidence properly capped")

    # Test case 2: Two credible sources should allow higher confidence
    print("\n2. Testing two credible sources (should allow >50%)")
    results = [
        {
            "fact": "Python is a programming language used for software development",
            "source": "dbpedia",  # High credibility (0.85)
            "confidence": 0.9,
            "status": "true"
        },
        {
            "fact": "Python programming language is widely used in data science",
            "source": "arxiv",  # High credibility (0.80)
            "confidence": 0.85,
            "status": "true"
        }
    ]

    disambiguation = engine.detect_ambiguity("python language", results)
    print(f"   Debug: disambiguation result: {disambiguation}")
    interpretation = disambiguation["interpretations"][0]
    confidence = interpretation["confidence_score"]

    print(f"   Two credible sources confidence: {confidence} (raw), {confidence*100:.0f}%")
    print(f"   Sources in result: {interpretation.get('sources_consulted', [])}")
    assert confidence > 0.5, f"Expected >0.5 with 2+ credible sources, got {confidence}"
    print("   ✅ PASS: Confidence allowed above 50% with corroboration")

    # Test case 3: False fact should have very low confidence
    print("\n3. Testing false fact (should be ≤10%)")
    results = [{
        "fact": "Python is a type of snake that lives in trees",
        "source": "user",
        "confidence": 0.8,
        "status": "false"  # Explicitly marked false
    }]

    disambiguation = engine.detect_ambiguity("python snake", results)
    interpretation = disambiguation["interpretations"][0]
    confidence = interpretation["confidence_score"]

    print(f"   False fact confidence: {confidence} (raw), {confidence*100:.0f}%")
    assert confidence <= 0.1, f"Expected ≤0.1 for false facts, got {confidence}"
    print("   ✅ PASS: False facts have very low confidence")

    # Test case 4: Ambiguous query should be detected
    print("\n4. Testing ambiguous query detection")
    ambiguous_results = [
        {
            "fact": "Python is a high-level programming language created by Guido van Rossum",
            "source": "wikipedia",
            "confidence": 0.9,
            "status": "true"
        },
        {
            "fact": "Python is a genus of non-venomous snakes in the family Pythonidae found in Africa and Asia",
            "source": "wikipedia",
            "confidence": 0.85,
            "status": "true"
        }
    ]

    # Debug: check categorization and merging
    engine = DisambiguationEngine()
    engine.cache = {}  # Clear cache for this specific test
    groups = engine._group_by_meaning("python", ambiguous_results)
    print(f"   Debug: Groups created: {list(groups.keys())}")
    for group_name, facts in groups.items():
        print(f"   - {group_name}: {[f['fact'] for f in facts]}")

    # Debug: check interpretations
    interpretations = engine._analyze_search_results("python", ambiguous_results)
    print(f"   Debug: Interpretations created: {len(interpretations)}")
    for i, interp in enumerate(interpretations):
        print(f"   - {i+1}: {interp['meaning_label']} (confidence: {interp['confidence_score']:.2f})")

    disambiguation = engine.detect_ambiguity("python", ambiguous_results)
    print(f"   Query 'python' is ambiguous: {disambiguation['is_ambiguous']}")
    print(f"   Number of interpretations: {len(disambiguation['interpretations'])}")
    for interp in disambiguation['interpretations']:
        print(f"   - {interp['meaning_label']}: {interp['summary'][:50]}...")
    assert disambiguation["is_ambiguous"] == True, "Expected 'python' to be detected as ambiguous"
    assert len(disambiguation["interpretations"]) >= 2, "Expected at least 2 interpretations for ambiguous query"
    print("   ✅ PASS: Ambiguous query properly detected")

    print("\n🎉 All confidence bound tests passed!")

def test_hybrid_memory_integration():
    """Test hybrid memory integration with disambiguation"""
    print("\nTesting hybrid memory integration...")

    # Create hybrid memory instance
    memory = HybridMemory()

    # Add some test facts
    memory.add_fact("Python is a programming language", source="wikipedia", confidence=0.95, status="true")
    memory.add_fact("Python is a snake genus", source="wikipedia", confidence=0.90, status="true")
    memory.add_fact("Java is a programming language", source="wikipedia", confidence=0.95, status="true")

    # Test search with disambiguation
    result = memory.search("python", include_disambiguation=True)

    print(f"   Search results: {len(result['results'])} facts")
    print(f"   Disambiguation detected: {result['disambiguation'] is not None}")
    if result['disambiguation']:
        disamb = result['disambiguation']
        print(f"   Is ambiguous: {disamb['is_ambiguous']}")
        print(f"   Interpretations: {len(disamb['interpretations'])}")
        print(f"   Overall confidence: {disamb['confidence']:.2f}")

    assert result['disambiguation'] is not None, "Expected disambiguation analysis"
    assert result['disambiguation']['is_ambiguous'] == True, "Expected 'python' to be ambiguous"

    print("   ✅ PASS: Hybrid memory integration works")

if __name__ == "__main__":
    try:
        test_confidence_bounds()
        test_hybrid_memory_integration()
        print("\n🎉 All tests completed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)