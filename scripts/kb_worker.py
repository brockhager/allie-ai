#!/usr/bin/env python3
"""Simple KB worker that processes learning_queue items and suggests actions

This worker polls the `learning_queue` table and attempts a simple reconciliation:
- If KB has same fact => mark validated
- If KB missing and queue confidence >= 0.8 => suggest promote to KB
- Else mark needs_review

All actions are logged into learning_log table via AllieMemoryDB methods.
"""
import time
import logging
from advanced_memory.db import AllieMemoryDB

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('kb_worker')


def main(poll_interval=5):
    db = AllieMemoryDB()
    logger.info('KB worker started')
    while True:
        try:
            cursor = db.connection.cursor(dictionary=True)
            cursor.execute("SELECT * FROM learning_queue WHERE status = 'pending' ORDER BY created_at ASC LIMIT 5")
            items = cursor.fetchall()
            cursor.close()
            if not items:
                time.sleep(poll_interval)
                continue
            for item in items:
                qid = item['id']
                keyword = item['keyword']
                fact = item['fact']
                conf = item.get('confidence', 0.3)
                logger.info(f'Processing queue item {qid} ({keyword})')

                kb = db.get_kb_fact(keyword)
                if kb and kb.get('fact', '').strip().lower() == fact.strip().lower():
                    # matches KB - mark validated
                    cursor = db.connection.cursor()
                    cursor.execute("UPDATE learning_queue SET status='validated', processed_at=CURRENT_TIMESTAMP WHERE id=%s", (qid,))
                    cursor.execute("INSERT INTO learning_log (keyword, old_fact, new_fact, source, confidence, change_type) VALUES (%s, %s, %s, %s, %s, 'validate')", (keyword, '', fact, item.get('source'), conf))
                    cursor.close()
                    logger.info(f'Queue {qid} validated against KB')
                    continue

                if not kb and conf >= 0.8:
                    # Suggest promote to KB
                    res = db.add_kb_fact(keyword, fact, source=item.get('source', 'auto_worker'), confidence_score=int(conf*100), provenance={'queued_id': qid}, status='not_verified')
                    cursor = db.connection.cursor()
                    cursor.execute("UPDATE learning_queue SET status='processed', processed_at=CURRENT_TIMESTAMP WHERE id=%s", (qid,))
                    cursor.close()
                    logger.info(f'Queue {qid} promoted to KB suggestion: {res}')
                    continue

                # Otherwise mark needs_review
                cursor = db.connection.cursor()
                cursor.execute("UPDATE learning_queue SET status='validated', processed_at=CURRENT_TIMESTAMP WHERE id=%s", (qid,))
                cursor.execute("INSERT INTO learning_log (keyword, old_fact, new_fact, source, confidence, change_type) VALUES (%s, %s, %s, %s, %s, 'needs_review')", (keyword, '', fact, item.get('source'), conf))
                cursor.close()
                logger.info(f'Queue {qid} marked needs_review')
        except Exception as e:
            logger.exception('Worker error: %s', e)
            time.sleep(poll_interval)


if __name__ == '__main__':
    main()
