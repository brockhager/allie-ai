#!/usr/bin/env python3
"""
Integration test for learning score reflecting KB growth through conversation

This test simulates:
1. Starting with minimal KB
2. Processing a conversation that generates KB facts
3. Verifying score increases appropriately
"""

import pytest
import sys
import asyncio
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "advanced-memory"))

from db import AllieMemoryDB


class TestLearningScoreIntegration:
    """Integration tests for learning score through conversation flow"""
    
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Setup and cleanup for each test"""
        self.db = AllieMemoryDB()
        
        # Clean up test data
        cursor = self.db.connection.cursor()
        cursor.execute("DELETE FROM knowledge_base WHERE keyword LIKE 'integration_test_%'")
        cursor.close()
        
        yield
        
        # Clean up after test
        cursor = self.db.connection.cursor()
        cursor.execute("DELETE FROM knowledge_base WHERE keyword LIKE 'integration_test_%'")
        cursor.close()
        self.db.close()
    
    def calculate_score(self, stats):
        """Calculate learning score using same formula as UI"""
        kb_total = stats.get('kb_total', 0)
        kb_active = stats.get('kb_active', 0)
        kb_recent_7d = stats.get('kb_recent_7d', 0)
        kb_recent_24h = stats.get('kb_recent_24h', 0)
        kb_avg_confidence = float(stats.get('kb_avg_confidence', 0))  # Convert Decimal to float
        
        if kb_total == 0:
            return 1
        
        raw_score = 0
        raw_score += min((kb_active / 10) * 10, 40)
        quality_ratio = kb_active / kb_total
        raw_score += quality_ratio * 25
        
        if kb_total > 0:
            growth_ratio = kb_recent_7d / kb_total
            raw_score += min(growth_ratio * 100, 25)
        
        if kb_avg_confidence > 0:
            raw_score += (kb_avg_confidence / 100) * 10
        
        raw_score = max(0, min(100, raw_score))
        final_score = round(raw_score * 10)
        growth_bonus = kb_recent_24h * 10
        final_score += growth_bonus
        
        return min(1000, max(1, final_score))
    
    def test_conversation_generates_kb_growth(self):
        """Simulate conversation that generates KB facts and verify score increases"""
        # Get baseline score
        initial_stats = self.db.get_kb_statistics()
        initial_score = self.calculate_score(initial_stats)
        
        print(f"\n=== Initial State ===")
        print(f"Initial KB stats: {initial_stats}")
        print(f"Initial learning score: {initial_score}")
        
        # Simulate conversation extracting facts
        # Example: User asks "What is the capital of France?"
        # System learns: "Paris is the capital of France"
        conversation_facts = [
            {
                "keyword": "integration_test_paris",
                "fact": "Paris is the capital of France",
                "confidence": 90
            },
            {
                "keyword": "integration_test_france",
                "fact": "France is a country in Europe",
                "confidence": 85
            },
            {
                "keyword": "integration_test_population",
                "fact": "Paris has a population of over 2 million people",
                "confidence": 80
            }
        ]
        
        print(f"\n=== Simulating Conversation ===")
        print(f"Learning {len(conversation_facts)} facts from conversation...")
        
        # Add facts to KB (simulating learning pipeline)
        for fact_data in conversation_facts:
            result = self.db.add_kb_fact(
                keyword=fact_data["keyword"],
                fact=fact_data["fact"],
                source="conversation",
                confidence_score=fact_data["confidence"],
                status="true"
            )
            print(f"  Added: {fact_data['keyword']}")
        
        # Get new stats and score
        new_stats = self.db.get_kb_statistics()
        new_score = self.calculate_score(new_stats)
        
        print(f"\n=== After Conversation ===")
        print(f"New KB stats: {new_stats}")
        print(f"New learning score: {new_score}")
        print(f"Score increase: {new_score - initial_score}")
        
        # Assertions
        assert new_stats['kb_total'] > initial_stats['kb_total'], "KB should have more facts"
        assert new_stats['kb_recent_24h'] >= 3, "Should have 3 recent facts"
        assert new_stats['kb_recent_7d'] >= 3, "Should have 3 recent facts in 7-day window"
        assert new_score > initial_score, f"Score should increase: {initial_score} -> {new_score}"
        
        # Should get at least 30 points from growth bonus (3 facts * 10)
        assert new_score >= initial_score + 30, f"Should get growth bonus of at least 30 points"
        
        print(f"\n✅ Integration test passed! Score increased by {new_score - initial_score} points")
    
    def test_multiple_conversations_compound_growth(self):
        """Test that multiple conversations compound KB growth"""
        initial_stats = self.db.get_kb_statistics()
        initial_score = self.calculate_score(initial_stats)
        
        print(f"\n=== Initial Score: {initial_score} ===")
        
        # First conversation
        for i in range(2):
            self.db.add_kb_fact(
                keyword=f"integration_test_conv1_{i}",
                fact=f"Conversation 1 fact {i}",
                source="conversation",
                confidence_score=85,
                status="true"
            )
        
        stats_after_conv1 = self.db.get_kb_statistics()
        score_after_conv1 = self.calculate_score(stats_after_conv1)
        print(f"After conversation 1: {score_after_conv1} (+{score_after_conv1 - initial_score})")
        
        # Second conversation
        for i in range(2):
            self.db.add_kb_fact(
                keyword=f"integration_test_conv2_{i}",
                fact=f"Conversation 2 fact {i}",
                source="conversation",
                confidence_score=90,
                status="true"
            )
        
        stats_after_conv2 = self.db.get_kb_statistics()
        score_after_conv2 = self.calculate_score(stats_after_conv2)
        print(f"After conversation 2: {score_after_conv2} (+{score_after_conv2 - score_after_conv1})")
        
        # Third conversation
        for i in range(2):
            self.db.add_kb_fact(
                keyword=f"integration_test_conv3_{i}",
                fact=f"Conversation 3 fact {i}",
                source="conversation",
                confidence_score=88,
                status="true"
            )
        
        stats_after_conv3 = self.db.get_kb_statistics()
        score_after_conv3 = self.calculate_score(stats_after_conv3)
        print(f"After conversation 3: {score_after_conv3} (+{score_after_conv3 - score_after_conv2})")
        
        print(f"\n=== Final Stats ===")
        print(f"Total KB facts: {stats_after_conv3['kb_total']}")
        print(f"Recent 24h: {stats_after_conv3['kb_recent_24h']}")
        print(f"Recent 7d: {stats_after_conv3['kb_recent_7d']}")
        print(f"Total score increase: {score_after_conv3 - initial_score}")
        
        # Verify compound growth
        assert score_after_conv3 > score_after_conv2 > score_after_conv1 > initial_score, \
            "Score should increase with each conversation"
        assert stats_after_conv3['kb_total'] >= 6, "Should have at least 6 facts"
        
        print(f"\n✅ Compound growth test passed!")
    
    def test_score_reflects_confidence_quality(self):
        """Test that score reflects the quality (confidence) of KB facts"""
        # Add low confidence facts
        for i in range(3):
            self.db.add_kb_fact(
                keyword=f"integration_test_low_{i}",
                fact=f"Low confidence fact {i}",
                source="conversation",
                confidence_score=45,
                status="true"
            )
        
        stats_low = self.db.get_kb_statistics()
        score_low = self.calculate_score(stats_low)
        
        print(f"\n=== Low Confidence Scenario ===")
        print(f"Average confidence: {stats_low['kb_avg_confidence']}")
        print(f"Score: {score_low}")
        
        # Clean and add high confidence facts
        cursor = self.db.connection.cursor()
        cursor.execute("DELETE FROM knowledge_base WHERE keyword LIKE 'integration_test_low_%'")
        cursor.close()
        
        for i in range(3):
            self.db.add_kb_fact(
                keyword=f"integration_test_high_{i}",
                fact=f"High confidence fact {i}",
                source="conversation",
                confidence_score=92,
                status="true"
            )
        
        stats_high = self.db.get_kb_statistics()
        score_high = self.calculate_score(stats_high)
        
        print(f"\n=== High Confidence Scenario ===")
        print(f"Average confidence: {stats_high['kb_avg_confidence']}")
        print(f"Score: {score_high}")
        print(f"Score difference: {score_high - score_low}")
        
        # High confidence should yield better score
        assert score_high > score_low, \
            f"Higher confidence should yield better score: {score_high} vs {score_low}"
        
        print(f"\n✅ Confidence quality test passed!")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
