#!/usr/bin/env python3
"""
Integration tests for newly added API endpoints
"""

import pytest
import json
from pathlib import Path
from fastapi.testclient import TestClient
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from backend.server import app


class TestNewAPIEndpoints:
    """Test the newly added API endpoints that were missing"""

    def setup_method(self):
        """Setup test environment"""
        self.client = TestClient(app)

    def test_facts_endpoint_exists(self):
        """Test that /api/facts endpoint exists and returns proper structure"""
        response = self.client.get("/api/facts")
        assert response.status_code == 200
        
        data = response.json()
        assert "facts" in data
        assert "total" in data
        assert "limit" in data
        assert "offset" in data
        assert isinstance(data["facts"], list)

    def test_facts_endpoint_with_pagination(self):
        """Test facts endpoint with pagination parameters"""
        response = self.client.get("/api/facts?limit=10&offset=5")
        assert response.status_code == 200
        
        data = response.json()
        assert data["limit"] == 10
        assert data["offset"] == 5

    def test_learning_status_endpoint_exists(self):
        """Test that /api/learning/status endpoint exists and returns proper structure"""
        response = self.client.get("/api/learning/status")
        assert response.status_code == 200
        
        data = response.json()
        assert "enabled" in data
        assert "is_active" in data
        assert "should_learn" in data
        assert "auto_learning" in data
        assert "reason" in data
        assert isinstance(data["enabled"], bool)

    def test_learning_start_endpoint(self):
        """Test that /api/learning/start endpoint works"""
        response = self.client.post("/api/learning/start")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "episode_id" in data
        assert data["status"] == "started"

    def test_learning_history_endpoint(self):
        """Test that /api/learning/history endpoint works"""
        response = self.client.get("/api/learning/history")
        assert response.status_code == 200
        
        data = response.json()
        assert "history" in data
        assert "total_episodes" in data
        assert isinstance(data["history"], list)

    def test_individual_fact_endpoint(self):
        """Test getting individual facts by ID"""
        # First get the list to see if we have any facts
        response = self.client.get("/api/facts")
        assert response.status_code == 200
        
        data = response.json()
        if data["total"] > 0:
            # Test getting the first fact
            response = self.client.get("/api/facts/1")
            assert response.status_code == 200
            
            fact_data = response.json()
            assert "id" in fact_data
            assert "fact" in fact_data
            assert "category" in fact_data
            assert "confidence" in fact_data
        else:
            # Test getting non-existent fact
            response = self.client.get("/api/facts/999")
            assert response.status_code == 404

    def test_fact_update_endpoint(self):
        """Test updating fact status"""
        # Try to update a non-existent fact
        update_payload = {
            "status": "outdated",
            "confidence": 0.5
        }
        
        response = self.client.put("/api/facts/999", json=update_payload)
        assert response.status_code == 404

    def test_hybrid_memory_stats_still_works(self):
        """Ensure existing hybrid memory stats endpoint still functions"""
        response = self.client.get("/api/hybrid-memory/statistics")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "statistics" in data

    def test_conversations_endpoint_still_works(self):
        """Ensure existing conversations endpoint still functions"""
        response = self.client.get("/api/conversations")
        assert response.status_code == 200
        
        # Should return a list of conversations
        data = response.json()
        assert isinstance(data, list)

    def test_generate_endpoint_still_works(self):
        """Test that the main generate endpoint still works"""
        payload = {
            "prompt": "Hello, how are you?",
            "max_tokens": 50
        }
        
        response = self.client.post("/api/generate", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert "text" in data
        assert isinstance(data["text"], str)
        assert len(data["text"]) > 0

    def test_learning_queue_endpoint_exists(self):
        """Test that learning queue endpoint exists"""
        response = self.client.get("/api/learning_queue")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "queue" in data
        assert "count" in data

    def test_feature_flags_endpoint_exists(self):
        """Test that feature flags endpoint exists"""
        response = self.client.get("/api/feature_flags")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "flags" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])