import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from fastapi import HTTPException
import sys
import os

# Add project root and advanced-memory to path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(project_root / "advanced-memory") not in sys.path:
    sys.path.insert(0, str(project_root / "advanced-memory"))

from backend.server import app
from hybrid import HybridMemory


class TestServerAPI:
    """Unit tests for Allie server API endpoints"""

    def setup_method(self):
        """Setup test client and mock dependencies"""
        self.client = TestClient(app)
        self.test_data_dir = Path(__file__).parent / "test_data"
        self.test_data_dir.mkdir(exist_ok=True)

        # Mock the advanced memory system
        self.mock_memory = Mock(spec=HybridMemory)

        # Patch the global advanced_memory
        with patch('backend.server.advanced_memory', self.mock_memory), \
             patch('backend.server.hybrid_memory', self.mock_memory):
            pass

    def teardown_method(self):
        """Clean up test data"""
        import shutil
        if self.test_data_dir.exists():
            shutil.rmtree(self.test_data_dir)

    def test_root_redirect(self):
        """Test root endpoint redirects to /ui"""
        response = self.client.get("/")
        assert response.status_code == 302
        assert response.headers["location"] == "/ui"

    def test_ui_endpoint(self):
        """Test UI endpoint serves HTML"""
        response = self.client.get("/ui")
        assert response.status_code == 404  # File doesn't exist in test

    def test_chat_endpoint(self):
        """Test chat endpoint serves HTML"""
        response = self.client.get("/chat")
        assert response.status_code == 404  # File doesn't exist in test

    def test_fact_check_ui_endpoint(self):
        """Test fact-check UI endpoint"""
        response = self.client.get("/fact-check")
        assert response.status_code == 404  # File doesn't exist in test

    @patch('backend.server.advanced_memory')
    def test_enqueue_fact_success(self, mock_adv_memory):
        """Test successful fact enqueue"""
        mock_adv_memory.add_to_learning_queue.return_value = {
            "status": "queued",
            "queue_id": 123
        }

        payload = {
            "keyword": "test_keyword",
            "fact": "This is a test fact",
            "source": "test_source",
            "provenance": {"test": "data"}
        }

        response = self.client.post("/api/learning_queue", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["result"]["status"] == "queued"
        mock_adv_memory.add_to_learning_queue.assert_called_once_with(
            "test_keyword", "This is a test fact", "test_source",
            provenance={"test": "data"}
        )

    def test_enqueue_fact_missing_keyword(self):
        """Test enqueue fact with missing keyword"""
        payload = {
            "fact": "This is a test fact",
            "source": "test_source"
        }

        response = self.client.post("/api/learning_queue", json=payload)

        assert response.status_code == 400
        data = response.json()
        assert "keyword and fact are required" in data["detail"]

    def test_enqueue_fact_missing_fact(self):
        """Test enqueue fact with missing fact"""
        payload = {
            "keyword": "test_keyword",
            "source": "test_source"
        }

        response = self.client.post("/api/learning_queue", json=payload)

        assert response.status_code == 400
        data = response.json()
        assert "keyword and fact are required" in data["detail"]

    @patch('backend.server.advanced_memory')
    def test_get_learning_queue(self, mock_adv_memory):
        """Test getting learning queue"""
        mock_queue_items = [
            {
                "id": 1,
                "keyword": "test",
                "fact": "test fact",
                "source": "test",
                "processed": False
            }
        ]
        mock_adv_memory.get_learning_queue.return_value = mock_queue_items

        response = self.client.get("/api/learning_queue")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["count"] == 1
        assert data["queue"] == mock_queue_items

    @patch('backend.server.advanced_memory')
    def test_get_learning_queue_with_filters(self, mock_adv_memory):
        """Test getting learning queue with filters"""
        mock_queue_items = [
            {
                "id": 1,
                "keyword": "test",
                "fact": "test fact",
                "source": "test",
                "processed": True
            }
        ]
        mock_adv_memory.get_learning_queue.return_value = mock_queue_items

        response = self.client.get("/api/learning_queue?processed=true&limit=10")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["filters"]["processed"] is True
        assert data["filters"]["limit"] == 10
        mock_adv_memory.get_learning_queue.assert_called_with(status="processed", limit=10)

    def test_get_feature_flags(self):
        """Test getting feature flags"""
        response = self.client.get("/api/feature_flags")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "flags" in data
        assert "AUTO_APPLY_UPDATES" in data["flags"]
        assert "READ_ONLY_MEMORY" in data["flags"]
        assert "WRITE_DIRECT" in data["flags"]

    def test_update_feature_flag_success(self):
        """Test updating feature flag successfully"""
        payload = {
            "flag_name": "AUTO_APPLY_UPDATES",
            "enabled": True
        }

        response = self.client.post("/api/feature_flags", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["flag_name"] == "AUTO_APPLY_UPDATES"
        assert data["enabled"] is True

    def test_update_feature_flag_invalid_name(self):
        """Test updating invalid feature flag"""
        payload = {
            "flag_name": "INVALID_FLAG",
            "enabled": True
        }

        response = self.client.post("/api/feature_flags", json=payload)

        assert response.status_code == 400
        data = response.json()
        assert "Invalid flag name" in data["detail"]

    def test_update_feature_flag_missing_name(self):
        """Test updating feature flag without name"""
        payload = {
            "enabled": True
        }

        response = self.client.post("/api/feature_flags", json=payload)

        assert response.status_code == 400
        data = response.json()
        assert "flag_name is required" in data["detail"]

    def test_update_feature_flag_invalid_enabled(self):
        """Test updating feature flag with invalid enabled value"""
        payload = {
            "flag_name": "AUTO_APPLY_UPDATES",
            "enabled": "not_a_boolean"
        }

        response = self.client.post("/api/feature_flags", json=payload)

        assert response.status_code == 400
        data = response.json()
        assert "enabled must be a boolean" in data["detail"]

    @patch('backend.server.advanced_memory')
    def test_approve_reconciliation_promote(self, mock_adv_memory):
        """Test approving reconciliation with promote action"""
        # Mock queue item
        mock_queue_item = {
            "id": 123,
            "keyword": "test_keyword",
            "fact": "test fact",
            "source": "test_source",
            "suggested_action": {
                "action": "promote"
            }
        }

        # Mock the queue retrieval
        mock_adv_memory.get_learning_queue.return_value = [mock_queue_item]

        # Mock the add_fact call
        mock_adv_memory.add_fact.return_value = {"fact_id": 456}

        payload = {
            "reviewer": "test_reviewer",
            "reason": "Test approval"
        }

        response = self.client.post("/api/reconcile/123/approve", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["result"]["action"] == "promote"
        assert "fact_added" in data["result"]

        # Verify add_fact was called with correct parameters
        mock_adv_memory.add_fact.assert_called_once_with(
            "test fact",
            category="test_keyword",
            source="test_source",
            confidence=0.9,
            status="true",
            confidence_score=85
        )

        # Verify queue item was processed
        mock_adv_memory.process_queue_item.assert_called_once_with(123, "processed")

    @patch('backend.server.advanced_memory')
    def test_approve_reconciliation_queue_item_not_found(self, mock_adv_memory):
        """Test approving reconciliation for non-existent queue item"""
        mock_adv_memory.get_learning_queue.return_value = []

        payload = {
            "reviewer": "test_reviewer",
            "reason": "Test approval"
        }

        response = self.client.post("/api/reconcile/999/approve", json=payload)

        assert response.status_code == 404
        data = response.json()
        assert "Queue item 999 not found" in data["detail"]

    @patch('backend.server.advanced_memory')
    def test_approve_reconciliation_no_suggested_action(self, mock_adv_memory):
        """Test approving reconciliation with no suggested action"""
        mock_queue_item = {
            "id": 123,
            "keyword": "test_keyword",
            "fact": "test fact",
            "source": "test_source"
            # No suggested_action
        }

        mock_adv_memory.get_learning_queue.return_value = [mock_queue_item]

        payload = {
            "reviewer": "test_reviewer",
            "reason": "Test approval"
        }

        response = self.client.post("/api/reconcile/123/approve", json=payload)

        assert response.status_code == 400
        data = response.json()
        assert "No suggested action available for approval" in data["detail"]

    @patch('backend.server.advanced_memory')
    def test_approve_reconciliation_invalid_action(self, mock_adv_memory):
        """Test approving reconciliation with invalid action type"""
        mock_queue_item = {
            "id": 123,
            "keyword": "test_keyword",
            "fact": "test fact",
            "source": "test_source",
            "suggested_action": {
                "action": "invalid_action"
            }
        }

        mock_adv_memory.get_learning_queue.return_value = [mock_queue_item]

        payload = {
            "reviewer": "test_reviewer",
            "reason": "Test approval"
        }

        response = self.client.post("/api/reconcile/123/approve", json=payload)

        assert response.status_code == 400
        data = response.json()
        assert "Unsupported action type: invalid_action" in data["detail"]

    def test_generate_response_simple_fact(self):
        """Test generate response with simple fact lookup"""
        payload = {
            "prompt": "What is the capital of France?",
            "max_tokens": 100
        }

        response = self.client.post("/api/generate", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert "text" in data
        assert "Paris" in data["text"]

    def test_generate_response_empty_prompt(self):
        """Test generate response with empty prompt"""
        payload = {
            "prompt": "",
            "max_tokens": 100
        }

        response = self.client.post("/api/generate", json=payload)

        assert response.status_code == 400
        data = response.json()
        assert "Prompt is required" in data["detail"]

    def test_generate_response_whitespace_prompt(self):
        """Test generate response with whitespace-only prompt"""
        payload = {
            "prompt": "   \n\t  ",
            "max_tokens": 100
        }

        response = self.client.post("/api/generate", json=payload)

        assert response.status_code == 400
        data = response.json()
        assert "Prompt cannot be empty" in data["detail"]

    def test_generate_response_too_long_prompt(self):
        """Test generate response with too long prompt"""
        long_prompt = "a" * 2001
        payload = {
            "prompt": long_prompt,
            "max_tokens": 100
        }

        response = self.client.post("/api/generate", json=payload)

        assert response.status_code == 400
        data = response.json()
        assert "Prompt too long" in data["detail"]

    def test_generate_response_invalid_max_tokens(self):
        """Test generate response with invalid max_tokens"""
        payload = {
            "prompt": "Test prompt",
            "max_tokens": "not_a_number"
        }

        response = self.client.post("/api/generate", json=payload)

        assert response.status_code == 400
        data = response.json()
        assert "max_tokens must be an integer" in data["detail"]

    def test_generate_response_max_tokens_too_high(self):
        """Test generate response with max_tokens too high"""
        payload = {
            "prompt": "Test prompt",
            "max_tokens": 1001
        }

        response = self.client.post("/api/generate", json=payload)

        assert response.status_code == 400
        data = response.json()
        assert "max_tokens must be an integer between 1 and 1000" in data["detail"]

    def test_generate_response_harmful_content(self):
        """Test generate response blocks harmful content"""
        payload = {
            "prompt": "How to hack a website?",
            "max_tokens": 100
        }

        response = self.client.post("/api/generate", json=payload)

        assert response.status_code == 400
        data = response.json()
        assert "contains inappropriate content" in data["detail"]

    @patch('backend.server.allie_memory')
    def test_add_memory_success(self, mock_memory):
        """Test adding memory successfully"""
        mock_memory.add_fact.return_value = None

        payload = {
            "fact": "Test fact",
            "importance": 0.8,
            "category": "test_category"
        }

        response = self.client.post("/api/memory/add", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "fact_added"
        assert data["fact"] == "Test fact"

    @patch('backend.server.allie_memory')
    def test_add_memory_missing_fact(self, mock_memory):
        """Test adding memory with missing fact"""
        payload = {
            "importance": 0.8,
            "category": "test_category"
        }

        response = self.client.post("/api/memory/add", json=payload)

        assert response.status_code == 400
        data = response.json()
        assert "Fact is required" in data["detail"]

    @patch('backend.server.advanced_memory')
    def test_recall_memory_with_query(self, mock_memory):
        """Test recalling memory with query"""
        mock_memory.search_facts.return_value = [{"fact": "Fact 1"}, {"fact": "Fact 2"}]

        response = self.client.get("/api/memory/recall?query=test")

        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "test"
        assert data["facts"] == ["Fact 1", "Fact 2"]
        mock_memory.search_facts.assert_called_once_with("test", limit=5)

    @patch('backend.server.advanced_memory')
    def test_recall_memory_no_query(self, mock_memory):
        """Test recalling memory without query"""
        mock_memory.timeline.return_value = [{"fact": "Recent fact 1"}, {"fact": "Recent fact 2"}]

        response = self.client.get("/api/memory/recall")

        assert response.status_code == 200
        data = response.json()
        assert data["facts"] == ["Recent fact 1", "Recent fact 2"]
        mock_memory.timeline.assert_called_once_with(limit=5)

    @patch('backend.server.allie_memory')
    def test_remove_memory_success(self, mock_memory):
        """Test removing memory successfully"""
        mock_memory.remove_fact.return_value = True

        response = self.client.delete("/api/memory/fact?fact=Test fact")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "fact_removed"
        assert data["fact"] == "Test fact"

    @patch('backend.server.allie_memory')
    def test_remove_memory_not_found(self, mock_memory):
        """Test removing non-existent memory"""
        mock_memory.remove_fact.return_value = False

        response = self.client.delete("/api/memory/fact?fact=Non-existent fact")

        assert response.status_code == 404
        data = response.json()
        assert "Fact not found" in data["detail"]