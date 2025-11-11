"""
Integration tests for Knowledge Base (KB) functionality.

Tests:
- KB CRUD operations via AllieMemoryDB
- Hybrid memory KB preference
- KB status filtering (true/false)
- Confidence scoring
- Audit logging
"""

import unittest
import os
import sys
from datetime import datetime
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from db import AllieMemoryDB
    from hybrid import HybridMemory
    HAS_DB = True
except ImportError:
    HAS_DB = False


class TestKnowledgeBaseIntegration(unittest.TestCase):
    """Test Knowledge Base integration with hybrid memory and DB"""

    @classmethod
    def setUpClass(cls):
        """Check if DB is available before running tests"""
        if not HAS_DB:
            raise unittest.SkipTest("AllieMemoryDB not available - skipping KB integration tests")
        
        # Test DB connection
        try:
            test_db = AllieMemoryDB()
            test_db.connection.close()
        except Exception as e:
            raise unittest.SkipTest(f"Cannot connect to MySQL database: {e}")

    def setUp(self):
        """Initialize test instances before each test"""
        self.db = AllieMemoryDB()
        self.hybrid = HybridMemory()
        
        # Clean up any test KB entries
        self._cleanup_test_data()

    def tearDown(self):
        """Clean up after each test"""
        self._cleanup_test_data()
        if hasattr(self, 'db') and self.db:
            self.db.connection.close()

    def _cleanup_test_data(self):
        """Remove test KB entries"""
        try:
            cursor = self.db.connection.cursor(dictionary=True)
            cursor.execute("DELETE FROM knowledge_base WHERE keyword LIKE 'test_%' OR keyword LIKE 'kb_test_%'")
            self.db.connection.commit()
            cursor.close()
        except Exception as e:
            print(f"Warning: cleanup failed: {e}")

    def test_add_kb_fact(self):
        """Test adding a fact to the Knowledge Base"""
        result = self.db.add_kb_fact(
            keyword="test_capital",
            fact="Paris is the capital of France",
            source="test_suite",
            confidence_score=95,
            provenance="manual entry",
            status="true"
        )
        
        self.assertTrue(result)
        
        # Verify it was added
        kb_fact = self.db.get_kb_fact("test_capital")
        self.assertIsNotNone(kb_fact)
        self.assertEqual(kb_fact['keyword'], 'test_capital')
        self.assertEqual(kb_fact['status'], 'true')
        self.assertEqual(kb_fact['confidence_score'], 95)

    def test_get_kb_fact_not_found(self):
        """Test retrieving non-existent KB fact"""
        result = self.db.get_kb_fact("nonexistent_keyword_xyz")
        self.assertIsNone(result)

    def test_update_kb_fact(self):
        """Test updating an existing KB fact"""
        # Add initial fact
        self.db.add_kb_fact(
            keyword="test_update",
            fact="Initial fact",
            source="test_suite",
            confidence_score=80,
            status="pending"
        )
        
        # Get the ID
        kb_fact = self.db.get_kb_fact("test_update")
        kb_id = kb_fact['id']
        
        # Update it
        result = self.db.update_kb_fact(
            kb_id=kb_id,
            new_fact="Updated fact",
            status="true",
            confidence_score=95,
            reviewer="test_reviewer",
            reason="verified"
        )
        
        self.assertTrue(result)
        
        # Verify update
        kb_fact = self.db.get_kb_fact("test_update")
        self.assertEqual(kb_fact['fact'], "Updated fact")
        self.assertEqual(kb_fact['status'], 'true')
        self.assertEqual(kb_fact['confidence_score'], 95)

    def test_delete_kb_fact(self):
        """Test deleting a KB fact"""
        # Add fact
        self.db.add_kb_fact(
            keyword="test_delete",
            fact="Fact to delete",
            source="test_suite",
            confidence_score=85
        )
        
        # Get the ID
        kb_fact = self.db.get_kb_fact("test_delete")
        kb_id = kb_fact['id']
        
        # Delete it
        result = self.db.delete_kb_fact(
            kb_id=kb_id,
            reviewer="test_reviewer",
            reason="test cleanup"
        )
        
        self.assertTrue(result)
        
        # Verify deletion
        kb_fact = self.db.get_kb_fact("test_delete")
        self.assertIsNone(kb_fact)

    def test_get_all_kb_facts(self):
        """Test listing all KB facts with filters"""
        # Add multiple facts with different statuses
        self.db.add_kb_fact("kb_test_1", "Fact 1", "test", 90, status="true")
        self.db.add_kb_fact("kb_test_2", "Fact 2", "test", 85, status="pending")
        self.db.add_kb_fact("kb_test_3", "Fact 3", "test", 80, status="false")
        
        # Get all test facts
        all_facts = self.db.get_all_kb_facts(limit=100)
        test_facts = [f for f in all_facts if f['keyword'].startswith('kb_test_')]
        self.assertEqual(len(test_facts), 3)
        
        # Filter by status
        true_facts = self.db.get_all_kb_facts(status="true", limit=100)
        true_test_facts = [f for f in true_facts if f['keyword'].startswith('kb_test_')]
        self.assertEqual(len(true_test_facts), 1)
        self.assertEqual(true_test_facts[0]['keyword'], 'kb_test_1')

    def test_hybrid_memory_kb_preference_true(self):
        """Test that hybrid memory prefers KB facts marked as 'true'"""
        # Add a KB fact marked as true
        self.db.add_kb_fact(
            keyword="Paris",
            fact="Paris is the capital and largest city of France",
            source="verified_source",
            confidence_score=98,
            status="true"
        )
        
        # Search via hybrid memory
        results = self.hybrid.search("Paris")
        
        # Should return KB fact immediately
        self.assertIsInstance(results, dict)
        self.assertIn('results', results)
        self.assertEqual(len(results['results']), 1)
        self.assertEqual(results['results'][0]['category'], 'knowledge_base')
        self.assertGreaterEqual(results['results'][0]['confidence'], 0.9)

    def test_hybrid_memory_kb_preference_false(self):
        """Test that hybrid memory excludes KB facts marked as 'false'"""
        # Add a KB fact marked as false
        self.db.add_kb_fact(
            keyword="fake_capital",
            fact="This is incorrect information",
            source="test",
            confidence_score=0,
            status="false"
        )
        
        # Search via hybrid memory
        results = self.hybrid.search("fake_capital")
        
        # Should return no results and include a warning
        self.assertIsInstance(results, dict)
        self.assertEqual(len(results['results']), 0)
        self.assertIn('fact_check_warnings', results)

    def test_kb_audit_logging(self):
        """Test that KB operations are logged to learning_log"""
        # Add a KB fact
        keyword = "test_audit_log"
        self.db.add_kb_fact(
            keyword=keyword,
            fact="Test fact for audit",
            source="test",
            confidence_score=85,
            status="pending"
        )
        
        # Check learning_log for entry (uses change_type not action_type, changed_at not timestamp)
        cursor = self.db.connection.cursor(dictionary=True)
        cursor.execute("""
            SELECT * FROM learning_log 
            WHERE change_type = 'add' 
            AND keyword = %s
            ORDER BY changed_at DESC LIMIT 1
        """, (keyword,))
        log_entry = cursor.fetchone()
        cursor.close()
        
        self.assertIsNotNone(log_entry)
        self.assertEqual(log_entry['change_type'], 'add')
        self.assertEqual(log_entry['keyword'], keyword)

    def test_kb_confidence_scoring(self):
        """Test confidence scoring in KB facts"""
        # Add facts with different confidence scores
        self.db.add_kb_fact("kb_high_conf", "High confidence fact", "test", 95)
        self.db.add_kb_fact("kb_med_conf", "Medium confidence fact", "test", 70)
        self.db.add_kb_fact("kb_low_conf", "Low confidence fact", "test", 40)
        
        high = self.db.get_kb_fact("kb_high_conf")
        med = self.db.get_kb_fact("kb_med_conf")
        low = self.db.get_kb_fact("kb_low_conf")
        
        self.assertEqual(high['confidence_score'], 95)
        self.assertEqual(med['confidence_score'], 70)
        self.assertEqual(low['confidence_score'], 40)
        
        # Verify they're ordered correctly in searches
        results = self.hybrid.search("kb_high_conf")
        if results['results']:
            self.assertGreaterEqual(results['results'][0]['confidence'], 0.9)


class TestKnowledgeBaseWorkerIntegration(unittest.TestCase):
    """Test KB worker integration with learning queue"""

    @classmethod
    def setUpClass(cls):
        if not HAS_DB:
            raise unittest.SkipTest("AllieMemoryDB not available")

    def setUp(self):
        self.db = AllieMemoryDB()
        self._cleanup_test_queue()

    def tearDown(self):
        self._cleanup_test_queue()
        if hasattr(self, 'db') and self.db:
            self.db.connection.close()

    def _cleanup_test_queue(self):
        """Clean up test learning queue entries"""
        try:
            cursor = self.db.connection.cursor()
            cursor.execute("DELETE FROM learning_queue WHERE keyword LIKE 'test_%'")
            self.db.connection.commit()
            cursor.close()
        except Exception:
            pass

    def test_add_to_learning_queue(self):
        """Test adding entries to learning queue for worker processing"""
        result = self.db.add_learning_queue(
            keyword="test_queue_entry",
            fact="Test fact from queue",
            source="test",
            confidence=0.75,
            category='general',
            provenance={'note': 'testing'}
        )
        
        self.assertTrue(result)
        
        # Verify it's in the queue
        cursor = self.db.connection.cursor(dictionary=True)
        cursor.execute("SELECT * FROM learning_queue WHERE keyword = %s", ("test_queue_entry",))
        entry = cursor.fetchone()
        cursor.close()
        
        self.assertIsNotNone(entry)
        self.assertEqual(entry['status'], 'pending')


def run_tests():
    """Run all KB integration tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestKnowledgeBaseIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestKnowledgeBaseWorkerIntegration))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
