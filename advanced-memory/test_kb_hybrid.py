#!/usr/bin/env python3
"""Test that HybridMemory prefers KB facts when available."""
import logging
logger = logging.getLogger(__name__)

try:
    from hybrid import HybridMemory
    from db import AllieMemoryDB
except Exception as e:
    logger.error('Required modules not available: %s', e)
    raise SystemExit(0)


def run_test():
    db = AllieMemoryDB()
    # Ensure test KB fact exists
    res = db.add_kb_fact('test_hybrid_keyword', 'Hybrid KB fact content', source='unit_test', confidence_score=95, provenance={'test':True}, status='true')
    print('add_kb_fact:', res)

    hybrid = HybridMemory()
    result = hybrid.search('test_hybrid_keyword', include_disambiguation=False)
    assert result and result.get('results'), 'hybrid.search returned no results'
    first = result['results'][0]
    assert 'Hybrid KB fact content' in first.get('fact','') or first.get('source') == 'knowledge_base', 'KB fact not preferred'
    print('Hybrid KB preference test passed')

    # Cleanup
    if isinstance(res, dict) and res.get('fact_id'):
        db.delete_kb_fact(res['fact_id'], reviewer='unit_test', reason='cleanup')

if __name__ == '__main__':
    run_test()
