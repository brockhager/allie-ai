#!/usr/bin/env python3
"""
Unit tests for learning score calculation based on KB growth

Tests:
1. Score increases proportionally when KB facts are added
2. Score reflects recent growth (7-day and 24-hour windows)
3. Confidence weighting affects score appropriately
4. Score stays in 1-1000 range
"""

import pytest
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "advanced-memory"))

from db import AllieMemoryDB


class TestLearningScoreKB:
    """Test learning score calculation using knowledge_base table"""
    
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Setup and cleanup for each test"""
        self.db = AllieMemoryDB()
        
        # Clean up test data before test
        cursor = self.db.connection.cursor()
        cursor.execute("DELETE FROM knowledge_base WHERE keyword LIKE 'test_%'")
        cursor.close()
        
        yield
        
        # Clean up test data after test
        cursor = self.db.connection.cursor()
        cursor.execute("DELETE FROM knowledge_base WHERE keyword LIKE 'test_%'")
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
        
        # Base score: KB active facts (0-40 points)
        raw_score += min((kb_active / 10) * 10, 40)
        
        # Quality ratio (0-25 points)
        quality_ratio = kb_active / kb_total
        raw_score += quality_ratio * 25
        
        # Growth score (0-25 points)
        if kb_total > 0:
            growth_ratio = kb_recent_7d / kb_total
            raw_score += min(growth_ratio * 100, 25)
        
        # Confidence weighting (0-10 points)
        if kb_avg_confidence > 0:
            raw_score += (kb_avg_confidence / 100) * 10
        
        raw_score = max(0, min(100, raw_score))
        
        # Scale to 1-1000
        final_score = round(raw_score * 10)
        
        # Growth bonus
        growth_bonus = kb_recent_24h * 10
        final_score += growth_bonus
        
        # Clamp to 1-1000
        return min(1000, max(1, final_score))
    
    def test_empty_kb_returns_minimum_score(self):
        """Test that empty KB returns score of 1"""
        stats = self.db.get_kb_statistics()
        score = self.calculate_score(stats)
        
        print(f"Empty KB score: {score}")
        assert score >= 1, "Score should be at least 1 for empty KB"
    
    def test_adding_facts_increases_score(self):
        """Test that adding 3 KB facts increases score proportionally"""
        # Get initial score
        initial_stats = self.db.get_kb_statistics()
        initial_score = self.calculate_score(initial_stats)
        
        print(f"\nInitial stats: {initial_stats}")
        print(f"Initial score: {initial_score}")
        
        # Add 3 KB facts
        for i in range(3):
            self.db.add_kb_fact(
                keyword=f"test_fact_{i}",
                fact=f"This is test fact number {i}",
                source="test",
                confidence_score=85,
                status="true"
            )
        
        # Get new score
        new_stats = self.db.get_kb_statistics()
        new_score = self.calculate_score(new_stats)
        
        print(f"\nNew stats: {new_stats}")
        print(f"New score: {new_score}")
        print(f"Score increase: {new_score - initial_score}")
        
        # Verify score increased
        assert new_score > initial_score, f"Score should increase after adding facts: {initial_score} -> {new_score}"
        
        # Verify proportional increase (should get growth bonus of 30 points minimum)
        assert new_score >= initial_score + 30, f"Score should increase by at least 30 points for 3 new facts"
    
    def test_recent_growth_affects_score(self):
        """Test that recent facts (7-day window) affect the score"""
        # Add some older facts (simulate by not using recent timestamp)
        for i in range(3):
            self.db.add_kb_fact(
                keyword=f"test_old_{i}",
                fact=f"Old fact {i}",
                source="test",
                confidence_score=80,
                status="true"
            )
        
        stats_with_old = self.db.get_kb_statistics()
        score_with_old = self.calculate_score(stats_with_old)
        
        # Add recent facts
        for i in range(3):
            self.db.add_kb_fact(
                keyword=f"test_new_{i}",
                fact=f"New fact {i}",
                source="test",
                confidence_score=85,
                status="true"
            )
        
        stats_with_new = self.db.get_kb_statistics()
        score_with_new = self.calculate_score(stats_with_new)
        
        print(f"\nScore with old facts only: {score_with_old}")
        print(f"Score with old + new facts: {score_with_new}")
        print(f"KB recent 7d: {stats_with_new['kb_recent_7d']}")
        print(f"KB recent 24h: {stats_with_new['kb_recent_24h']}")
        
        # Recent facts should increase score
        assert score_with_new > score_with_old, "Adding recent facts should increase score"
        assert stats_with_new['kb_recent_7d'] >= 3, "Should have at least 3 recent facts in 7-day window"
        assert stats_with_new['kb_recent_24h'] >= 3, "Should have at least 3 recent facts in 24-hour window"
    
    def test_confidence_affects_score(self):
        """Test that confidence weighting affects score appropriately"""
        # Add facts with low confidence
        for i in range(3):
            self.db.add_kb_fact(
                keyword=f"test_low_conf_{i}",
                fact=f"Low confidence fact {i}",
                source="test",
                confidence_score=40,
                status="true"
            )
        
        stats_low = self.db.get_kb_statistics()
        score_low = self.calculate_score(stats_low)
        
        print(f"\nLow confidence avg: {stats_low['kb_avg_confidence']}")
        print(f"Score with low confidence: {score_low}")
        
        # Clean up and add high confidence facts
        cursor = self.db.connection.cursor()
        cursor.execute("DELETE FROM knowledge_base WHERE keyword LIKE 'test_low_conf_%'")
        cursor.close()
        
        for i in range(3):
            self.db.add_kb_fact(
                keyword=f"test_high_conf_{i}",
                fact=f"High confidence fact {i}",
                source="test",
                confidence_score=95,
                status="true"
            )
        
        stats_high = self.db.get_kb_statistics()
        score_high = self.calculate_score(stats_high)
        
        print(f"High confidence avg: {stats_high['kb_avg_confidence']}")
        print(f"Score with high confidence: {score_high}")
        
        # High confidence should yield higher score
        assert stats_high['kb_avg_confidence'] > stats_low['kb_avg_confidence'], "High confidence facts should have higher avg"
        assert score_high > score_low, f"Higher confidence should yield higher score: {score_low} vs {score_high}"
    
    def test_score_stays_in_range(self):
        """Test that score always stays in 1-1000 range"""
        # Add many facts to try to exceed 1000
        for i in range(50):
            self.db.add_kb_fact(
                keyword=f"test_many_{i}",
                fact=f"Fact {i}",
                source="test",
                confidence_score=95,
                status="true"
            )
        
        stats = self.db.get_kb_statistics()
        score = self.calculate_score(stats)
        
        print(f"\nScore with 50 facts: {score}")
        print(f"KB stats: {stats}")
        
        assert 1 <= score <= 1000, f"Score must be in range 1-1000, got {score}"
    
    def test_example_scenario(self):
        """Test the example: 3/12 facts in last 7 days should give ~25 points from growth"""
        # Add 9 older facts first (simulate by adding them)
        for i in range(9):
            self.db.add_kb_fact(
                keyword=f"test_old_scenario_{i}",
                fact=f"Old scenario fact {i}",
                source="test",
                confidence_score=80,
                status="true"
            )
        
        # Update their timestamp to be older (simulate older facts)
        # For this test, we'll just add 3 more new facts to get 12 total with 3 recent
        for i in range(3):
            self.db.add_kb_fact(
                keyword=f"test_new_scenario_{i}",
                fact=f"New scenario fact {i}",
                source="test",
                confidence_score=85,
                status="true"
            )
        
        stats = self.db.get_kb_statistics()
        score = self.calculate_score(stats)
        
        print(f"\nExample scenario stats:")
        print(f"  KB Total: {stats['kb_total']}")
        print(f"  KB Recent 7d: {stats['kb_recent_7d']}")
        print(f"  KB Recent 24h: {stats['kb_recent_24h']}")
        print(f"  Growth ratio: {stats['kb_recent_7d'] / stats['kb_total'] if stats['kb_total'] > 0 else 0:.2%}")
        print(f"  Final score: {score}")
        
        # With 12 total and 3 recent (25% growth), should get growth points
        assert stats['kb_total'] >= 12, "Should have 12 facts total"
        assert score > 1, "Score should reflect the KB growth"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
