#!/usr/bin/env python3
"""Debug automatic learner fact extraction"""
import sys
import importlib.util

spec = importlib.util.spec_from_file_location('learner', 'backend/automatic_learner.py')
learner_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(learner_module)

def debug_fact_extraction():
    """Debug why facts aren't being extracted"""
    print("🔍 DEBUGGING FACT EXTRACTION")
    print("=" * 50)

    learner = learner_module.AutomaticLearner(None, None, None)

    # Test messages that should extract facts
    test_messages = [
        "Albert Einstein was born on March 14, 1879 in Ulm, Germany.",
        "Paris is the capital of France.",
        "Python was created by Guido van Rossum in 1991.",
        "World War II ended in 1945.",
        "Water boils at 100 degrees Celsius at sea level.",
    ]

    for msg in test_messages:
        print(f"\n📝 Testing message: {msg}")

        # Check if message passes initial filters
        msg_lower = msg.lower()
        skip_phrases = [
            "i think", "i believe", "i feel", "in my opinion", "i'm not sure",
            "maybe", "perhaps", "possibly", "i don't know", "not sure",
            "let me check", "i'll look it up", "can you tell me"
        ]

        if any(phrase in msg_lower for phrase in skip_phrases):
            print("❌ Message skipped (contains uncertainty phrases)")
            continue

        # Split into sentences
        import re
        sentences = re.split(r'[.!?]+', msg)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 5]

        print(f"📋 Sentences: {sentences}")

        for sentence in sentences:
            print(f"  Processing: {sentence}")

            # Check quality filters
            if not learner._passes_quality_filters(sentence):
                print("  ❌ Failed quality filters")
                continue

            # Try pattern matching
            facts_found = []
            for category, patterns in learner.fact_patterns.items():
                for pattern in patterns:
                    matches = re.findall(pattern, sentence, re.IGNORECASE)
                    if matches:
                        for match in matches:
                            fact_text = match.strip()
                            if fact_text and learner._passes_quality_filters(fact_text):
                                confidence = learner._calculate_fact_confidence(fact_text, category, pattern)
                                facts_found.append({
                                    'fact': fact_text,
                                    'category': category,
                                    'confidence': confidence,
                                    'pattern': pattern
                                })

            if facts_found:
                print(f"  ✅ Found {len(facts_found)} facts:")
                for fact in facts_found:
                    print(f"    - {fact['fact']} (cat: {fact['category']}, conf: {fact['confidence']:.2f})")
            else:
                print("  ❌ No facts extracted from this sentence")

def test_biography_pattern():
    """Test the biography pattern specifically"""
    print("\n🎭 TESTING BIOGRAPHY PATTERN")
    print("=" * 50)

    import re
    learner = learner_module.AutomaticLearner(None, None, None)

    test_fact = "Albert Einstein was born on March 14, 1879 in Ulm, Germany"

    print(f"Testing fact: {test_fact}")

    # Test each biography pattern
    for i, pattern in enumerate(learner.fact_patterns['biography']):
        print(f"Pattern {i+1}: {pattern}")
        matches = re.findall(pattern, test_fact, re.IGNORECASE)
        if matches:
            print(f"  ✅ Matches: {matches}")
            for match in matches:
                confidence = learner._calculate_fact_confidence(match, 'biography', pattern)
                print(f"    Confidence: {confidence}")
        else:
            print("  ❌ No match")
def main():
    debug_fact_extraction()
    test_biography_pattern()

if __name__ == "__main__":
    main()