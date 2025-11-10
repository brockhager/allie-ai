import pytest
import json
import tempfile
import sys
from pathlib import Path
from datetime import datetime

# Add project root and advanced-memory to path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(project_root / "advanced-memory") not in sys.path:
    sys.path.insert(0, str(project_root / "advanced-memory"))

from hybrid import HybridMemory


class TestHybridMemory:
    """Unit tests for HybridMemory class"""

    def setup_method(self):
        """Create temporary directory and HybridMemory instance for each test"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.memory_file = self.temp_dir / "test_memory.json"
        self.memory = HybridMemory(str(self.memory_file))

    def teardown_method(self):
        """Clean up temporary files"""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_initialization_empty(self):
        """Test initialization with empty memory file"""
        assert self.memory.facts == []
        assert self.memory.data_file == self.memory_file

    def test_initialization_with_existing_file(self):
        """Test initialization with existing memory file"""
        # Create a memory file with some data
        initial_data = {
            "version": "1.0",
            "saved_at": datetime.now().isoformat(),
            "fact_count": 2,
            "facts": [
                {
                    "fact": "Test fact 1",
                    "timestamp": datetime.now().isoformat(),
                    "category": "test",
                    "confidence": 0.8,
                    "source": "test",
                    "status": "not_verified",
                    "confidence_score": 80,
                    "is_outdated": False
                },
                {
                    "fact": "Test fact 2",
                    "timestamp": datetime.now().isoformat(),
                    "category": "test",
                    "confidence": 0.9,
                    "source": "test",
                    "status": "true",
                    "confidence_score": 90,
                    "is_outdated": False
                }
            ]
        }

        with open(self.memory_file, 'w') as f:
            json.dump(initial_data, f)

        # Create new memory instance
        memory = HybridMemory(str(self.memory_file))

        assert len(memory.facts) == 2
        assert memory.facts[0]["fact"] == "Test fact 1"
        assert memory.facts[1]["fact"] == "Test fact 2"

    def test_add_fact_basic(self):
        """Test adding a basic fact"""
        result = self.memory.add_fact("This is a test fact")

        assert result["status"] == "stored"
        assert result["fact"] == "This is a test fact"
        assert len(self.memory.facts) == 1

        fact = self.memory.facts[0]
        assert fact["fact"] == "This is a test fact"
        assert fact["category"] == "general"
        assert fact["confidence"] == 0.9
        assert fact["source"] == "user"
        assert fact["status"] == "not_verified"
        assert fact["confidence_score"] == 90
        assert fact["is_outdated"] is False
        assert "timestamp" in fact

    def test_add_fact_with_parameters(self):
        """Test adding a fact with custom parameters"""
        result = self.memory.add_fact(
            "Custom fact",
            category="science",
            confidence=0.75,
            source="research",
            status="true",
            confidence_score=85
        )

        assert result["status"] == "stored"
        assert len(self.memory.facts) == 1

        fact = self.memory.facts[0]
        assert fact["fact"] == "Custom fact"
        assert fact["category"] == "science"
        assert fact["confidence"] == 0.75
        assert fact["source"] == "research"
        assert fact["status"] == "true"
        assert fact["confidence_score"] == 85

    def test_add_fact_confidence_score_calculation(self):
        """Test that confidence_score is calculated from confidence when not provided"""
        result = self.memory.add_fact("Test fact", confidence=0.65)

        fact = self.memory.facts[0]
        assert fact["confidence"] == 0.65
        assert fact["confidence_score"] == 65

    def test_add_fact_duplicate_detection(self):
        """Test that adding similar facts marks old ones as outdated"""
        # Add first fact
        self.memory.add_fact("Paris is the capital of France and has many museums")

        # Add similar but different fact (same first 6 words)
        result = self.memory.add_fact("Paris is the capital of France and has great food")

        assert result["status"] == "stored"
        assert "updated" in result
        assert result["updated"] is True
        assert "old_fact" in result

        # Check that first fact is marked outdated
        assert self.memory.facts[1]["is_outdated"] is True  # First fact (now second in list)
        assert self.memory.facts[0]["is_outdated"] is False  # New fact (first in list)

    def test_search_basic(self):
        """Test basic search functionality"""
        self.memory.add_fact("Paris is the capital of France")
        self.memory.add_fact("London is the capital of England")
        self.memory.add_fact("Berlin is the capital of Germany")

        results = self.memory.search("capital")

        assert len(results) == 3
        # Results should be sorted by status priority, then confidence, then recency
        assert all("capital" in result["fact"].lower() for result in results)

    def test_search_empty_query(self):
        """Test search with empty query"""
        self.memory.add_fact("Test fact")
        results = self.memory.search("")
        assert results == []

    def test_search_no_matches(self):
        """Test search with no matching results"""
        self.memory.add_fact("Paris is the capital of France")
        results = self.memory.search("moon")
        assert results == []

    def test_search_with_limit(self):
        """Test search with result limit"""
        for i in range(10):
            self.memory.add_fact(f"Fact number {i} about capitals")

        results = self.memory.search("capitals", limit=3)
        assert len(results) == 3

    def test_search_status_priority(self):
        """Test that search prioritizes facts by status"""
        # Add facts with different statuses
        self.memory.add_fact("Fact with true status", status="true", confidence=0.8)
        self.memory.add_fact("Fact with experimental status", status="experimental", confidence=0.8)
        self.memory.add_fact("Fact with not_verified status", status="not_verified", confidence=0.8)
        self.memory.add_fact("Fact with needs_review status", status="needs_review", confidence=0.8)
        self.memory.add_fact("Fact with false status", status="false", confidence=0.8)

        results = self.memory.search("status")

        # Should be ordered: true, experimental, not_verified, needs_review, false
        assert results[0]["status"] == "true"
        assert results[1]["status"] == "experimental"
        assert results[2]["status"] == "not_verified"
        assert results[3]["status"] == "needs_review"
        assert results[4]["status"] == "false"

    def test_get_timeline_active_only(self):
        """Test getting timeline with only active facts"""
        self.memory.add_fact("Active fact 1")
        self.memory.add_fact("Active fact 2")
        self.memory.add_fact("Will be outdated", status="true")

        # Mark one as outdated
        self.memory.facts[2]["is_outdated"] = True

        timeline = self.memory.get_timeline(include_outdated=False)

        assert len(timeline) == 2
        assert all(not fact.get("is_outdated", False) for fact in timeline)

    def test_get_timeline_include_outdated(self):
        """Test getting timeline including outdated facts"""
        self.memory.add_fact("Active fact")
        self.memory.add_fact("Outdated fact")
        self.memory.facts[1]["is_outdated"] = True

        timeline = self.memory.get_timeline(include_outdated=True)

        assert len(timeline) == 2
        assert timeline[0]["is_outdated"] is False
        assert timeline[1]["is_outdated"] is True

    def test_get_statistics(self):
        """Test getting memory statistics"""
        # Add some test facts
        self.memory.add_fact("Science fact 1", category="science", source="research")
        self.memory.add_fact("History fact 1", category="history", source="book")
        self.memory.add_fact("Science fact 2", category="science", source="research")
        self.memory.add_fact("Outdated fact", category="general", source="unknown")
        self.memory.facts[3]["is_outdated"] = True

        stats = self.memory.get_statistics()

        assert stats["total_facts"] == 4
        assert stats["active_facts"] == 3
        assert stats["outdated_facts"] == 1
        assert stats["categories"]["science"] == 2
        assert stats["categories"]["history"] == 1
        assert stats["categories"]["general"] == 1
        assert stats["sources"]["research"] == 2
        assert stats["sources"]["book"] == 1
        assert stats["sources"]["unknown"] == 1
        assert "indexed_keywords" in stats

    def test_reconcile_with_external_new_facts(self):
        """Test reconciling with external facts - adding new facts"""
        external_facts = ["New fact from external source", "Another new fact"]

        result = self.memory.reconcile_with_external("test query", external_facts, "external_api")

        assert result["facts_added"] == external_facts
        assert result["conflicts_found"] == 0
        assert result["memory_confirmed"] == []
        assert len(self.memory.facts) == 2

    def test_reconcile_with_external_conflicts(self):
        """Test reconciling with external facts - handling conflicts"""
        # Add existing fact
        self.memory.add_fact("Paris is the capital of France", source="memory")

        # External source has slightly different version with same prefix
        external_facts = ["Paris is the capital of France in Europe"]

        result = self.memory.reconcile_with_external("test query", external_facts, "external_api")

        assert result["conflicts_found"] == 1
        assert len(result["facts_updated"]) == 1
        assert result["facts_added"] == []
        assert len(self.memory.facts) == 2  # Original (outdated) + new

        # Check that old fact is outdated and new one exists
        assert self.memory.facts[1]["is_outdated"] is True  # Old fact
        assert self.memory.facts[0]["is_outdated"] is False  # New fact
        assert self.memory.facts[0]["source"] == "external_api"

    def test_reconcile_with_external_matching(self):
        """Test reconciling with external facts - exact matches"""
        # Add existing fact
        self.memory.add_fact("Paris is the capital of France", source="memory")

        # External source has exact match
        external_facts = ["Paris is the capital of France"]

        result = self.memory.reconcile_with_external("test query", external_facts, "external_api")

        assert result["facts_added"] == []
        assert result["conflicts_found"] == 0
        assert result["memory_confirmed"] == ["Paris is the capital of France"]
        assert len(self.memory.facts) == 1  # No new facts added

    def test_add_to_learning_queue(self):
        """Test adding fact to learning queue"""
        result = self.memory.add_to_learning_queue(
            "test_keyword",
            "test fact",
            "test_source",
            {"test": "provenance"}
        )

        assert result["status"] == "queued"
        assert "queue_id" in result

    def test_get_learning_queue(self):
        """Test getting learning queue (mock implementation)"""
        # This is a mock implementation, so it returns empty list
        result = self.memory.get_learning_queue()
        assert result == []

    def test_process_queue_item(self):
        """Test processing queue item (mock implementation)"""
        result = self.memory.process_queue_item(123, "processed")
        assert result["status"] == "processed"
        assert result["queue_id"] == 123
        assert result["action"] == "processed"

    def test_update_fact_status(self):
        """Test updating fact status"""
        # Add a fact
        self.memory.add_fact("Test fact", status="not_verified", confidence_score=50)

        # Update its status
        result = self.memory.update_fact_status(0, "true", 85)

        assert result["status"] == "updated"
        assert result["fact_id"] == 0
        assert result["new_status"] == "true"
        assert result["new_confidence_score"] == 85

        # Check the fact was updated
        fact = self.memory.facts[0]
        assert fact["status"] == "true"
        assert fact["confidence_score"] == 85
        assert "updated_at" in fact

    def test_update_fact_status_not_found(self):
        """Test updating status of non-existent fact"""
        result = self.memory.update_fact_status(999, "true", 85)

        assert result["status"] == "not_found"
        assert result["fact_id"] == 999

    def test_get_fact_by_id(self):
        """Test getting fact by ID"""
        self.memory.add_fact("Test fact 1")
        self.memory.add_fact("Test fact 2")

        # Facts are stored most recent first, so index 0 is "Test fact 2"
        fact = self.memory.get_fact_by_id(0)
        assert fact is not None
        assert fact["fact"] == "Test fact 2"

        fact = self.memory.get_fact_by_id(1)
        assert fact is not None
        assert fact["fact"] == "Test fact 1"

        # Test invalid ID
        fact = self.memory.get_fact_by_id(999)
        assert fact is None

    def test_get_all_facts_no_filters(self):
        """Test getting all facts without filters"""
        self.memory.add_fact("Fact 1", category="cat1", status="true")
        self.memory.add_fact("Fact 2", category="cat2", status="false")
        self.memory.add_fact("Fact 3", category="cat1", status="true")

        facts = self.memory.get_all_facts()

        assert len(facts) == 3

    def test_get_all_facts_with_filters(self):
        """Test getting all facts with filters"""
        self.memory.add_fact("Fact 1", category="cat1", status="true")
        self.memory.add_fact("Fact 2", category="cat2", status="false")
        self.memory.add_fact("Fact 3", category="cat1", status="true")

        # Filter by status
        facts = self.memory.get_all_facts(status_filter="true")
        assert len(facts) == 2
        assert all(f["status"] == "true" for f in facts)

        # Filter by category
        facts = self.memory.get_all_facts(category_filter="cat1")
        assert len(facts) == 2
        assert all(f["category"] == "cat1" for f in facts)

        # Filter by both
        facts = self.memory.get_all_facts(status_filter="true", category_filter="cat1")
        assert len(facts) == 2

    def test_get_all_facts_pagination(self):
        """Test getting all facts with pagination"""
        for i in range(10):
            self.memory.add_fact(f"Fact {i}")

        # Get first 5
        facts = self.memory.get_all_facts(limit=5, offset=0)
        assert len(facts) == 5
        assert facts[0]["fact"] == "Fact 9"  # Most recent first

        # Get next 5
        facts = self.memory.get_all_facts(limit=5, offset=5)
        assert len(facts) == 5
        assert facts[0]["fact"] == "Fact 4"

    def test_compute_confidence_score(self):
        """Test computing confidence score"""
        provenance = {
            "source": "wikipedia",
            "channel": "verified_source"
        }
        source_weights = {"wikipedia": 0.9}

        score = self.memory.compute_confidence_score(provenance, source_weights, recency_days=5)

        # Calculation: 50 * 0.9 (source) + 10 (recency) + 15 (verified channel) = 70
        assert score == 70

    def test_mark_false(self):
        """Test marking fact as false"""
        self.memory.add_fact("Test fact", status="true", confidence_score=80)

        result = self.memory.mark_false(0, "test_reviewer", "Test reason")

        assert result["status"] == "updated"
        assert result["new_status"] == "false"
        assert result["new_confidence_score"] == 0

        fact = self.memory.facts[0]
        assert fact["status"] == "false"
        assert fact["confidence_score"] == 0

    def test_rollback_fact(self):
        """Test rolling back fact (mock implementation)"""
        result = self.memory.rollback_fact(123, 456)

        assert result["status"] == "rolled_back"
        assert result["fact_id"] == 123
        assert result["target_log_id"] == 456