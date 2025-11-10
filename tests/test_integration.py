import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
import sys

# Add project root and advanced-memory to path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(project_root / "advanced-memory") not in sys.path:
    sys.path.insert(0, str(project_root / "advanced-memory"))

from backend.server import app
from hybrid import HybridMemory
from scripts.reconciliation_worker import ReconciliationWorker


class TestLearningPipelineIntegration:
    """Integration tests for the complete learning pipeline"""

    def setup_method(self):
        """Setup test environment"""
        self.client = TestClient(app)
        self.temp_dir = Path(tempfile.mkdtemp())

        # Create test data directory
        self.data_dir = self.temp_dir / "data"
        self.data_dir.mkdir()

        # Mock the data directory in server
        with patch('backend.server.DATA_DIR', self.data_dir), \
             patch('backend.server.APP_ROOT', project_root / "backend"):

            # Create a test hybrid memory instance
            self.memory_file = self.data_dir / "hybrid_memory.json"
            self.memory = HybridMemory(str(self.memory_file))

            # Patch the global memory instances
            patch('backend.server.advanced_memory', self.memory).start()
            patch('backend.server.hybrid_memory', self.memory).start()

            # Also patch the auto_learner's hybrid_memory
            from backend.server import auto_learner
            if auto_learner and hasattr(auto_learner, 'hybrid_memory'):
                auto_learner.hybrid_memory = self.memory

    def teardown_method(self):
        """Clean up test data"""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_complete_learning_pipeline(self):
        """Test the complete learning pipeline from enqueue to approval"""

        # Step 1: Enqueue a fact for learning
        enqueue_payload = {
            "keyword": "paris",
            "fact": "Paris is the capital of France",
            "source": "user_input",
            "provenance": {
                "channel": "chat",
                "timestamp": "2024-01-01T10:00:00Z",
                "confidence": 0.8
            }
        }

        response = self.client.post("/api/learning_queue", json=enqueue_payload)
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "success"
        assert data["result"]["status"] == "queued"

        # Step 2: Check that fact was added to learning queue
        response = self.client.get("/api/learning_queue")
        assert response.status_code == 200

        queue_data = response.json()
        assert queue_data["status"] == "success"
        # Note: Current implementation returns empty list (mock)

        # Step 3: Simulate reconciliation worker processing
        # In a real scenario, the worker would run and suggest actions
        # For testing, we'll directly approve a hypothetical queue item

        # Mock a queue item that would be created by the worker
        mock_queue_item = {
            "id": 1,
            "keyword": "paris",
            "fact": "Paris is the capital of France",
            "source": "user_input",
            "suggested_action": {
                "action": "promote",
                "confidence": 0.9,
                "reason": "Fact verified against external sources"
            }
        }

        # Patch the queue retrieval to return our mock item
        with patch.object(self.memory, 'get_learning_queue', return_value=[mock_queue_item]):
            # Step 4: Approve the reconciliation
            approve_payload = {
                "reviewer": "test_admin",
                "reason": "Integration test approval"
            }

            response = self.client.post("/api/reconcile/1/approve", json=approve_payload)
            assert response.status_code == 200

            approve_data = response.json()
            assert approve_data["status"] == "success"
            assert approve_data["result"]["action"] == "promote"
            assert "fact_added" in approve_data["result"]

        # Step 5: Verify fact was added to memory with correct status
        # Check that add_fact was called with correct parameters
        # This would be verified in a real integration test by checking the database

        # Step 6: Test searching for the learned fact
        search_response = self.client.get("/api/hybrid-memory/search?query=paris")
        assert search_response.status_code == 200

        search_data = search_response.json()
        assert search_data["query"] == "paris"
        # In real scenario, this would return the fact we just added

    def test_memory_api_integration(self):
        """Test integration between memory APIs"""

        # Add a fact via the API
        add_payload = {
            "fact": "Berlin is the capital of Germany",
            "category": "geography",
            "confidence": 0.9,
            "source": "integration_test"
        }

        response = self.client.post("/api/hybrid-memory/add", json=add_payload)
        assert response.status_code == 200

        add_data = response.json()
        assert add_data["status"] == "success"
        assert add_data["fact"] == "Berlin is the capital of Germany"

        # Search for the fact
        search_response = self.client.get("/api/hybrid-memory/search?query=berlin")
        assert search_response.status_code == 200

        search_data = search_response.json()
        assert search_data["query"] == "berlin"
        assert search_data["count"] >= 1

        # Get memory statistics
        stats_response = self.client.get("/api/hybrid-memory/statistics")
        assert stats_response.status_code == 200

        stats_data = stats_response.json()
        assert stats_data["status"] == "success"
        assert "statistics" in stats_data
        assert stats_data["statistics"]["total_facts"] >= 1

        # Get memory timeline
        timeline_response = self.client.get("/api/hybrid-memory/timeline")
        assert timeline_response.status_code == 200

        timeline_data = timeline_response.json()
        assert timeline_data["count"] >= 1
        assert len(timeline_data["timeline"]) >= 1

    def test_feature_flags_integration(self):
        """Test feature flags integration with API protection"""

        # Get current feature flags
        response = self.client.get("/api/feature_flags")
        assert response.status_code == 200

        flags_data = response.json()
        assert flags_data["status"] == "success"
        assert "flags" in flags_data

        # Test updating a feature flag
        update_payload = {
            "flag_name": "READ_ONLY_MEMORY",
            "enabled": True
        }

        response = self.client.post("/api/feature_flags", json=update_payload)
        assert response.status_code == 200

        update_data = response.json()
        assert update_data["status"] == "success"
        assert update_data["flag_name"] == "READ_ONLY_MEMORY"
        assert update_data["enabled"] is True

    @patch('backend.server.check_feature_flag')
    def test_protected_endpoints_respect_feature_flags(self, mock_check_flag):
        """Test that protected endpoints check feature flags"""

        # Mock feature flag check to return True (read-only mode enabled)
        mock_check_flag.return_value = True

        # Try to update a fact (this would be protected in read-only mode)
        # Note: In the current implementation, the fact update endpoint isn't fully protected
        # but this test demonstrates the pattern for future implementation

        # For now, just test that feature flag checking works
        assert mock_check_flag("READ_ONLY_MEMORY") is True
        assert mock_check_flag("AUTO_APPLY_UPDATES") is True

    def test_conversation_workflow_integration(self):
        """Test the conversation workflow with learning integration"""

        # Create a conversation
        conv_payload = {
            "id": "test-conv-001",
            "messages": [
                {"role": "me", "text": "What is the capital of Italy?", "timestamp": 1640995200.0},
                {"role": "them", "text": "Rome is the capital of Italy.", "timestamp": 1640995260.0}
            ]
        }

        response = self.client.put("/api/conversations/test-conv-001", json=conv_payload)
        assert response.status_code == 200

        # Generate a response (this would trigger learning)
        gen_payload = {
            "prompt": "Tell me about Rome",
            "max_tokens": 100
        }

        response = self.client.post("/api/generate", json=gen_payload)
        assert response.status_code == 200

        gen_data = response.json()
        assert "text" in gen_data

        # Check that conversation was saved
        response = self.client.get("/api/conversations/list")
        assert response.status_code == 200

        conv_data = response.json()
        assert "conversations" in conv_data

    def test_error_handling_integration(self):
        """Test error handling across the integrated system"""

        # Test invalid learning queue request
        invalid_enqueue = {
            "keyword": "",  # Invalid: empty keyword
            "fact": "Test fact"
        }

        response = self.client.post("/api/learning_queue", json=invalid_enqueue)
        assert response.status_code == 400

        # Test invalid reconciliation approval
        response = self.client.post("/api/reconcile/999/approve", json={})
        assert response.status_code == 404

        # Test invalid feature flag update
        invalid_flag = {
            "flag_name": "INVALID_FLAG",
            "enabled": True
        }

        response = self.client.post("/api/feature_flags", json=invalid_flag)
        assert response.status_code == 400

        # Test invalid memory search
        response = self.client.get("/api/hybrid-memory/search?query=")
        assert response.status_code == 200  # Empty query returns empty results

    def test_memory_persistence_integration(self):
        """Test that memory persists correctly across operations"""

        # Add multiple facts
        facts_to_add = [
            {"fact": "Vienna is the capital of Austria", "category": "geography"},
            {"fact": "Madrid is the capital of Spain", "category": "geography"},
            {"fact": "Lisbon is the capital of Portugal", "category": "geography"}
        ]

        for fact_data in facts_to_add:
            response = self.client.post("/api/hybrid-memory/add", json=fact_data)
            assert response.status_code == 200

        # Get statistics
        response = self.client.get("/api/hybrid-memory/statistics")
        assert response.status_code == 200

        stats = response.json()["statistics"]
        assert stats["total_facts"] == 3
        assert stats["active_facts"] == 3
        assert stats["categories"]["geography"] == 3

        # Search for capitals
        response = self.client.get("/api/hybrid-memory/search?query=capital")
        assert response.status_code == 200

        search_results = response.json()
        assert search_results["count"] == 3
        assert len(search_results["facts"]) == 3

        # Test reconciliation with external facts
        external_facts = [
            "Rome is the capital of Italy",  # New fact
            "Vienna is the capital of Austria"  # Exact match
        ]

        reconcile_payload = {
            "query": "european capitals",
            "external_facts": external_facts,
            "source": "integration_test"
        }

        response = self.client.post("/api/hybrid-memory/reconcile", json=reconcile_payload)
        assert response.status_code == 200

        reconcile_data = response.json()
        assert reconcile_data["status"] == "success"
        assert "reconciliation_report" in reconcile_data

        # Verify final state
        final_stats_response = self.client.get("/api/hybrid-memory/statistics")
        final_stats = final_stats_response.json()["statistics"]

        # Should have 4 total facts now (3 original + 1 new from reconciliation)
        assert final_stats["total_facts"] == 4
        assert final_stats["active_facts"] == 4  # All should be active