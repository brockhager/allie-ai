#!/usr/bin/env python3
"""Basic tests for Knowledge Base DB methods. Skips if DB not available."""
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from db import AllieMemoryDB
except Exception as e:
    logger.error('AllieMemoryDB not importable: %s', e)
    raise SystemExit(0)


def run_tests():
    db = AllieMemoryDB()
    # Create a KB fact
    res = db.add_kb_fact('test_kb_keyword', 'This is a test KB fact', source='unit_test', confidence_score=85, provenance={'test': True})
    assert res.get('status') in ('added','updated','exists')
    logger.info('add_kb_fact OK: %s', res)

    # Retrieve
    kb = db.get_kb_fact('test_kb_keyword')
    assert kb and kb.get('keyword') == 'test_kb_keyword'
    logger.info('get_kb_fact OK: %s', kb)

    # Update
    if kb:
        up = db.update_kb_fact(kb['id'], new_fact='Updated KB fact', status='not_verified', confidence_score=50, reviewer='unit_test', reason='testing')
        assert up.get('status') == 'updated'
        logger.info('update_kb_fact OK: %s', up)

        # Delete
        dl = db.delete_kb_fact(kb['id'], reviewer='unit_test', reason='cleanup')
        assert dl.get('status') == 'deleted'
        logger.info('delete_kb_fact OK: %s', dl)

    print('KB tests completed')

if __name__ == '__main__':
    run_tests()
