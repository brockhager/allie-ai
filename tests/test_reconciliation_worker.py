import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from scripts.reconciliation_worker import ReconciliationWorker


class TestReconciliationWorker:
    """Unit tests for ReconciliationWorker"""

    def setup_method(self):
        """Setup test worker instance"""
        db_config = {
            "host": "localhost",
            "database": "allie_memory",
            "user": "allie",
            "password": "StrongPassword123!"
        }
        self.worker = ReconciliationWorker(db_config)

    @patch('scripts.reconciliation_worker.httpx.AsyncClient')
    def test_search_duckduckgo_success(self, mock_client_class):
        """Test successful DuckDuckGo search"""
        # Mock the client and response
        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "Answer": "Test answer",
            "AbstractText": "Test abstract",
            "RelatedTopics": [{"Text": "Related topic"}]
        }
        mock_client.get.return_value = mock_response

        async def run_test():
            result = await self.worker.query_duckduckgo("test query")

            assert result["success"] is True
            assert result["query"] == "test query"
            assert len(result["results"]) > 0
            assert result["results"][0]["source"] in ["duckduckgo_instant", "duckduckgo_abstract"]

        asyncio.run(run_test())

    @patch('scripts.reconciliation_worker.httpx.AsyncClient')
    def test_search_duckduckgo_failure(self, mock_client_class):
        """Test DuckDuckGo search failure"""
        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client

        mock_response = Mock()
        mock_response.status_code = 500
        mock_client.get.return_value = mock_response

        async def run_test():
            result = await self.worker.query_duckduckgo("test query")

            assert result["success"] is False
            assert result["query"] == "test query"
            assert len(result["results"]) == 0
            assert "error" in result

        asyncio.run(run_test())

    @patch('scripts.reconciliation_worker.httpx.AsyncClient')
    def test_search_wikidata_success(self, mock_client_class):
        """Test successful Wikidata search"""
        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client

        # Mock search response
        search_response = Mock()
        search_response.status_code = 200
        search_response.json.return_value = {
            "search": [{
                "id": "Q90",
                "label": "Paris",
                "description": "capital and most populous city of France"
            }]
        }

        # Mock entity response
        entity_response = Mock()
        entity_response.status_code = 200
        entity_response.json.return_value = {
            "entities": {
                "Q90": {
                    "claims": {
                        "P17": [{"mainsnak": {"datavalue": {"value": {"id": "Q142"}}}}]  # France
                    }
                }
            }
        }

        mock_client.get.side_effect = [search_response, entity_response]

        async def run_test():
            result = await self.worker.query_wikidata("test query")

            assert result["success"] is True
            assert result["query"] == "test query"
            assert len(result["results"]) > 0
            assert result["results"][0]["source"] == "wikidata"

        asyncio.run(run_test())

    @patch('scripts.reconciliation_worker.httpx.AsyncClient')
    def test_search_dbpedia_success(self, mock_client_class):
        """Test successful DBpedia search"""
        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client

        # Mock Spotlight response
        spotlight_response = Mock()
        spotlight_response.status_code = 200
        spotlight_response.json.return_value = {
            "Resources": [{
                "@URI": "http://dbpedia.org/resource/Paris",
                "@surfaceForm": "Paris",
                "@similarityScore": "0.9",
                "@support": "100",
                "@types": "City"
            }]
        }

        # Mock SPARQL response
        sparql_response = Mock()
        sparql_response.status_code = 200
        sparql_response.json.return_value = {
            "results": {
                "bindings": [{
                    "property": {"value": "http://dbpedia.org/ontology/country"},
                    "value": {"value": "France"}
                }]
            }
        }

        mock_client.get.side_effect = [spotlight_response, sparql_response]

        async def run_test():
            result = await self.worker.query_dbpedia("test query")

            assert result["success"] is True
            assert result["query"] == "test query"
            assert len(result["results"]) > 0
            assert result["results"][0]["source"] == "dbpedia"

        asyncio.run(run_test())

    def test_calculate_agreement_score_exact_match(self):
        """Test agreement score calculation with exact matches"""
        memory_fact = "Paris is the capital of France"
        external_facts = ["Paris is the capital of France"]

        score = self.worker.calculate_agreement_score(memory_fact, external_facts)

        assert score == 1.0  # Perfect match

    def test_calculate_agreement_score_partial_match(self):
        """Test agreement score with partial matches"""
        memory_fact = "Paris is the capital of France"
        external_facts = ["Paris is the capital city of France", "London is the capital of England"]

        score = self.worker.calculate_agreement_score(memory_fact, external_facts)

        assert score > 0.5  # Should have good agreement

    def test_calculate_agreement_score_no_match(self):
        """Test agreement score with no matches"""
        memory_fact = "Paris is the capital of France"
        external_facts = ["London is the capital of England", "Berlin is the capital of Germany"]

        score = self.worker.calculate_agreement_score(memory_fact, external_facts)

        assert score == 0.0  # No agreement

    def test_apply_policy_rules_high_agreement(self):
        """Test policy rules with high agreement"""
        fact_data = {
            "fact": "Paris is the capital of France",
            "confidence_score": 60,
            "source": "user"
        }

        agreement_score = 0.9
        external_sources = ["wikipedia", "duckduckgo"]

        action = self.worker.apply_policy_engine(agreement_score, 85)

        assert action["action"] == "promote"
        assert action["computed_confidence"] == 85

    def test_apply_policy_rules_low_agreement(self):
        """Test policy rules with low agreement"""
        fact_data = {
            "fact": "Some controversial claim",
            "confidence_score": 30,
            "source": "user"
        }

        agreement_score = 0.2
        external_sources = ["wikipedia"]

        action = self.worker.apply_policy_engine(agreement_score, 45)

        assert action["action"] == "needs_review"
        assert action["computed_confidence"] == 45

    def test_apply_policy_rules_conflicting_sources(self):
        """Test policy rules when sources conflict"""
        fact_data = {
            "fact": "Controversial fact",
            "confidence_score": 50,
            "source": "user"
        }

        agreement_score = 0.0  # Complete disagreement
        external_sources = ["source1", "source2"]

        action = self.worker.apply_policy_engine(agreement_score, 25)

        assert action["action"] == "ignore"
        assert action["computed_confidence"] == 25

    @patch.object(ReconciliationWorker, 'query_duckduckgo', new_callable=AsyncMock)
    @patch.object(ReconciliationWorker, 'query_wikidata', new_callable=AsyncMock)
    @patch.object(ReconciliationWorker, 'query_dbpedia', new_callable=AsyncMock)
    def test_reconcile_fact_success(self, mock_dbpedia, mock_wikidata, mock_duckduckgo):
        """Test successful fact reconciliation"""
        # Mock external search results
        mock_duckduckgo.return_value = {
            "success": True,
            "results": [{"text": "Paris is the capital of France"}]
        }
        mock_wikidata.return_value = {
            "success": True,
            "results": [{"text": "Paris: capital and most populous city of France"}]
        }
        mock_dbpedia.return_value = {
            "success": True,
            "results": [{"text": "Paris"}]
        }

        fact_data = {
            "id": 1,
            "fact": "Paris is the capital of France",
            "confidence_score": 60,
            "source": "user"
        }

        async def run_test():
            result = await self.worker.process_queue_item(fact_data)

            assert "action" in result
            assert result["action"] in ["promote", "needs_review", "auto_update", "ignore"]
            assert "agreement_score" in result
            assert "aggregated_confidence" in result
            assert "sources_queried" in result
            assert "external_results_count" in result

        asyncio.run(run_test())

    @patch('scripts.reconciliation_worker.httpx.AsyncClient')
    def test_batch_process_queue(self, mock_client_class):
        """Test batch processing of queue items"""
        # Mock client for external searches
        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client

        # Mock successful responses for all sources
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"Answer": "Test"}
        mock_client.get.return_value = mock_response

        # Mock queue items
        queue_items = [
            {
                "id": 1,
                "fact": "Fact 1",
                "keyword": "keyword1",
                "source": "user",
                "confidence_score": 50
            },
            {
                "id": 2,
                "fact": "Fact 2",
                "keyword": "keyword2",
                "source": "user",
                "confidence_score": 60
            }
        ]

        # Mock the get_pending_queue_items method
        with patch.object(self.worker, 'get_pending_queue_items', return_value=queue_items):
            async def run_test():
                await self.worker.run_batch()

                # Verify that process_queue_item was called for each item
                # This is a basic test - in real scenario we'd check the results

            asyncio.run(run_test())

    def test_format_reconciliation_report(self):
        """Test formatting of reconciliation report"""
        reconciliation_results = [
            {
                "queue_item": {"id": 1, "fact": "Fact 1"},
                "reconciliation": {
                    "suggested_action": {"action": "promote", "confidence": 0.9},
                    "agreement_score": 0.8,
                    "sources_used": ["wikipedia", "duckduckgo"]
                }
            },
            {
                "queue_item": {"id": 2, "fact": "Fact 2"},
                "reconciliation": {
                    "suggested_action": {"action": "mark_needs_review", "confidence": 0.5},
                    "agreement_score": 0.3,
                    "sources_used": ["wikipedia"]
                }
            }
        ]

        report = self.worker._format_reconciliation_report(reconciliation_results)

        assert "processed" in report
        assert "promoted" in report
        assert "needs_review" in report
        assert "total_processed" in report
        assert report["total_processed"] == 2
        assert report["promoted"] == 1
        assert report["needs_review"] == 1

    def test_get_pending_queue_items(self):
        """Test getting pending queue items"""
        # Mock database connection and cursor
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = [
            {
                "id": 1,
                "keyword": "test",
                "fact": "test fact",
                "source": "test_source",
                "received_at": "2024-01-01T10:00:00Z"
            }
        ]
        
        with patch.object(self.worker, 'connection') as mock_conn:
            mock_conn.cursor.return_value = mock_cursor
            items = self.worker.get_pending_queue_items(limit=5)
            
            assert len(items) == 1
            assert items[0]["keyword"] == "test"
            mock_cursor.execute.assert_called_once()
            mock_cursor.close.assert_called_once()