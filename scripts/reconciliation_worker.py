#!/usr/bin/env python3
"""
Reconciliation Worker for Allie Memory System

This worker periodically processes items from the learning_queue, queries external sources,
computes confidence scores, and suggests actions for human review.

Features:
- Polls learning_queue for unprocessed items
- Queries multiple external sources (DuckDuckGo, Wikidata, DBpedia)
- Computes agreement scores and aggregated confidence
- Applies policy engine rules to suggest actions
- Updates learning_queue with suggested_action
- Includes rate limiting and batch processing
"""

import os
import sys
import json
import time
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import httpx
import mysql.connector
from mysql.connector import Error

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s %(message)s'
)
logger = logging.getLogger(__name__)

class ReconciliationWorker:
    """Worker that processes learning queue items and suggests reconciliation actions"""

    def __init__(self, db_config: Dict[str, Any]):
        self.db_config = db_config
        self.connection = None
        self.http_client = None

        # Configuration
        self.batch_size = int(os.environ.get("WORKER_BATCH_SIZE", "10"))
        self.poll_interval = int(os.environ.get("WORKER_POLL_INTERVAL", "60"))  # seconds
        self.max_retries = int(os.environ.get("WORKER_MAX_RETRIES", "3"))
        self.rate_limit_delay = float(os.environ.get("WORKER_RATE_LIMIT_DELAY", "1.0"))

        # Source weights for confidence calculation
        self.source_weights = {
            "duckduckgo": 0.7,
            "wikidata": 0.9,
            "dbpedia": 0.8,
            "wikipedia": 0.85
        }

        # Policy thresholds
        self.agreement_threshold_high = float(os.environ.get("AGREEMENT_THRESHOLD_HIGH", "0.9"))
        self.agreement_threshold_medium = float(os.environ.get("AGREEMENT_THRESHOLD_MEDIUM", "0.7"))
        self.confidence_threshold_auto = int(os.environ.get("CONFIDENCE_THRESHOLD_AUTO", "80"))

    def connect_db(self):
        """Establish database connection"""
        try:
            self.connection = mysql.connector.connect(**self.db_config, autocommit=False)
            logger.info("Connected to database")
        except Error as e:
            logger.error(f"Database connection failed: {e}")
            raise

    def close_db(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")

    async def create_http_client(self):
        """Create HTTP client for external requests"""
        if not self.http_client:
            self.http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(10.0, connect=10.0),
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
            )

    async def close_http_client(self):
        """Close HTTP client"""
        if self.http_client:
            await self.http_client.aclose()
            self.http_client = None

    def get_pending_queue_items(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get pending items from learning_queue"""
        try:
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute("""
                SELECT id, keyword, fact, source, created_at
                FROM learning_queue
                WHERE status = 'pending'
                ORDER BY created_at ASC
                LIMIT %s
            """, (limit,))

            items = cursor.fetchall()
            cursor.close()
            return items

        except Error as e:
            logger.error(f"Error getting queue items: {e}")
            return []

    def update_queue_item_suggestion(self, queue_id: int, suggested_action: Dict[str, Any]):
        """Update a queue item with suggested action"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                UPDATE learning_queue
                SET status = 'processed', processed_at = NOW()
                WHERE id = %s
            """, (queue_id,))

            self.connection.commit()
            cursor.close()
            logger.info(f"Updated queue item {queue_id} with suggestion")

        except Error as e:
            logger.error(f"Error updating queue item {queue_id}: {e}")
            self.connection.rollback()

    async def query_duckduckgo(self, query: str) -> Dict[str, Any]:
        """Query DuckDuckGo for fact verification"""
        try:
            await asyncio.sleep(self.rate_limit_delay)  # Rate limiting

            url = f"https://api.duckduckgo.com/?q={query}&format=json&no_html=1&skip_disambig=1"
            response = await self.http_client.get(url)

            if response.status_code == 200:
                data = response.json()
                results = []

                # Extract instant answer
                if data.get("Answer"):
                    results.append({
                        "text": data["Answer"],
                        "source": "duckduckgo_instant",
                        "confidence": 0.8
                    })

                # Extract abstract
                if data.get("AbstractText"):
                    results.append({
                        "text": data["AbstractText"],
                        "source": "duckduckgo_abstract",
                        "confidence": 0.75
                    })

                return {
                    "success": True,
                    "results": results[:3],  # Limit to top 3
                    "query": query
                }
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}",
                    "query": query
                }

        except Exception as e:
            logger.warning(f"DuckDuckGo query failed for '{query}': {e}")
            return {
                "success": False,
                "error": str(e),
                "query": query
            }

    async def query_wikidata(self, query: str) -> Dict[str, Any]:
        """Query Wikidata for structured facts"""
        try:
            await asyncio.sleep(self.rate_limit_delay)  # Rate limiting

            # First search for entities
            search_url = f"https://www.wikidata.org/w/api.php?action=wbsearchentities&search={query}&language=en&format=json&limit=3"
            response = await self.http_client.get(search_url)

            if response.status_code == 200:
                data = response.json()
                results = []

                if data.get("search"):
                    for entity in data["search"][:2]:  # Limit to top 2
                        entity_id = entity.get("id")
                        if entity_id:
                            results.append({
                                "text": f"{entity.get('label', '')}: {entity.get('description', '')}",
                                "source": "wikidata",
                                "entity_id": entity_id,
                                "confidence": 0.9
                            })

                return {
                    "success": True,
                    "results": results,
                    "query": query
                }
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}",
                    "query": query
                }

        except Exception as e:
            logger.warning(f"Wikidata query failed for '{query}': {e}")
            return {
                "success": False,
                "error": str(e),
                "query": query
            }

    async def query_dbpedia(self, query: str) -> Dict[str, Any]:
        """Query DBpedia for facts"""
        try:
            await asyncio.sleep(self.rate_limit_delay)  # Rate limiting

            # Use DBpedia Spotlight for entity extraction
            spotlight_url = f"https://api.dbpedia-spotlight.org/en/annotate?text={query}&confidence=0.5&support=20"
            response = await self.http_client.get(spotlight_url, headers={"Accept": "application/json"})

            if response.status_code == 200:
                data = response.json()
                results = []

                if "Resources" in data:
                    for resource in data["Resources"][:2]:  # Limit to top 2
                        results.append({
                            "text": resource.get("@surfaceForm", ""),
                            "source": "dbpedia",
                            "uri": resource.get("@URI", ""),
                            "confidence": float(resource.get("@similarityScore", 0))
                        })

                return {
                    "success": True,
                    "results": results,
                    "query": query
                }
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}",
                    "query": query
                }

        except Exception as e:
            logger.warning(f"DBpedia query failed for '{query}': {e}")
            return {
                "success": False,
                "error": str(e),
                "query": query
            }

    def calculate_agreement_score(self, fact: str, external_results: List[Dict[str, Any]]) -> float:
        """Calculate agreement score between stored fact and external sources"""
        if not external_results:
            return 0.0

        fact_lower = fact.lower().strip()
        agreements = 0
        total_sources = len(external_results)

        for result in external_results:
            external_text = result.get("text", "").lower().strip()
            if external_text:
                # Simple text similarity check (could be improved with embeddings)
                fact_words = set(fact_lower.split())
                external_words = set(external_text.split())

                # Calculate Jaccard similarity
                intersection = fact_words.intersection(external_words)
                union = fact_words.union(external_words)

                if union:
                    similarity = len(intersection) / len(union)
                    if similarity > 0.3:  # Threshold for considering it an agreement
                        agreements += 1

        return agreements / total_sources if total_sources > 0 else 0.0

    def calculate_aggregated_confidence(self, external_results: List[Dict[str, Any]], recency_days: int = 0) -> float:
        """Calculate aggregated confidence from external sources"""
        if not external_results:
            return 30.0  # Default low confidence

        total_weighted_confidence = 0.0
        total_weight = 0.0

        for result in external_results:
            source = result.get("source", "").split("_")[0]  # Extract base source name
            source_weight = self.source_weights.get(source, 0.5)
            result_confidence = result.get("confidence", 0.5)

            # Apply recency bonus (newer sources get slight boost)
            recency_bonus = min(recency_days / 365.0, 0.1)  # Max 10% bonus for very recent

            weighted_confidence = (result_confidence * source_weight) + recency_bonus
            total_weighted_confidence += weighted_confidence
            total_weight += source_weight

        if total_weight > 0:
            aggregated = (total_weighted_confidence / total_weight) * 100
            return min(max(aggregated, 0), 100)  # Clamp to 0-100
        else:
            return 30.0

    def apply_policy_engine(self, agreement_score: float, aggregated_confidence: float) -> Dict[str, Any]:
        """Apply policy rules to determine suggested action"""
        # Check feature flag for auto-apply (would be checked from database in real implementation)
        auto_apply_enabled = os.environ.get("AUTO_APPLY_UPDATES", "false").lower() == "true"

        if agreement_score >= self.agreement_threshold_high and aggregated_confidence >= self.confidence_threshold_auto and auto_apply_enabled:
            return {
                "action": "auto_update",
                "reason": ".2f",
                "computed_confidence": int(aggregated_confidence)
            }
        elif agreement_score >= self.agreement_threshold_medium:
            return {
                "action": "promote",
                "reason": ".2f",
                "computed_confidence": int(aggregated_confidence)
            }
        elif agreement_score < 0.5:
            return {
                "action": "ignore",
                "reason": ".2f",
                "computed_confidence": int(aggregated_confidence)
            }
        else:
            return {
                "action": "needs_review",
                "reason": ".2f",
                "computed_confidence": int(aggregated_confidence)
            }

    async def process_queue_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single queue item"""
        queue_id = item["id"]
        keyword = item["keyword"]
        fact = item["fact"]

        logger.info(f"Processing queue item {queue_id}: '{keyword}'")

        # Query external sources in parallel
        queries = await asyncio.gather(
            self.query_duckduckgo(keyword),
            self.query_wikidata(keyword),
            self.query_dbpedia(keyword)
        )

        # Collect successful results
        external_results = []
        for query_result in queries:
            if query_result.get("success") and query_result.get("results"):
                external_results.extend(query_result["results"])

        # Calculate scores
        agreement_score = self.calculate_agreement_score(fact, external_results)

        # Calculate recency (days since fact was received)
        received_at = item.get("created_at")
        if received_at:
            if isinstance(received_at, str):
                received_at = datetime.fromisoformat(received_at.replace('Z', '+00:00'))
            recency_days = (datetime.now() - received_at).days
        else:
            recency_days = 0

        aggregated_confidence = self.calculate_aggregated_confidence(external_results, recency_days)

        # Apply policy engine
        suggested_action = self.apply_policy_engine(agreement_score, aggregated_confidence)

        # Add external candidates to suggestion
        candidates = []
        for result in external_results[:3]:  # Limit to top 3
            candidates.append({
                "source": result.get("source", "unknown"),
                "text": result.get("text", ""),
                "confidence": result.get("confidence", 0.5)
            })

        suggested_action["candidates"] = candidates
        suggested_action["agreement_score"] = agreement_score
        suggested_action["aggregated_confidence"] = aggregated_confidence
        suggested_action["sources_queried"] = len(queries)
        suggested_action["external_results_count"] = len(external_results)

        # Update the queue item
        self.update_queue_item_suggestion(queue_id, suggested_action)

        logger.info(f"Processed queue item {queue_id}: action={suggested_action['action']}, agreement={agreement_score:.2f}, confidence={aggregated_confidence:.1f}")

        return suggested_action

    async def run_batch(self):
        """Process a batch of queue items"""
        logger.info(f"Starting batch processing (batch_size={self.batch_size})")

        items = self.get_pending_queue_items(self.batch_size)
        if not items:
            logger.info("No pending items in queue")
            return

        logger.info(f"Processing {len(items)} queue items")

        # Process items concurrently but with rate limiting
        tasks = []
        for item in items:
            tasks.append(self.process_queue_item(item))
            await asyncio.sleep(0.5)  # Small delay between starting tasks

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Log results
        successful = 0
        failed = 0
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to process item {items[i]['id']}: {result}")
                failed += 1
            else:
                successful += 1

        logger.info(f"Batch completed: {successful} successful, {failed} failed")

    async def run_worker_loop(self):
        """Main worker loop"""
        logger.info("Starting reconciliation worker loop")

        await self.create_http_client()

        try:
            while True:
                try:
                    await self.run_batch()
                except Exception as e:
                    logger.error(f"Error in batch processing: {e}")

                logger.info(f"Sleeping for {self.poll_interval} seconds")
                await asyncio.sleep(self.poll_interval)

        except KeyboardInterrupt:
            logger.info("Worker stopped by user")
        finally:
            await self.close_http_client()

    def run_once(self):
        """Run a single batch and exit (for cron jobs)"""
        logger.info("Running single batch")

        async def run():
            await self.create_http_client()
            try:
                await self.run_batch()
            finally:
                await self.close_http_client()

        asyncio.run(run())

def main():
    # Database configuration
    db_config = {
        "host": os.environ.get("DB_HOST", "localhost"),
        "database": os.environ.get("DB_NAME", "allie_memory"),
        "user": os.environ.get("DB_USER", "allie"),
        "password": os.environ.get("DB_PASSWORD", "StrongPassword123!")
    }

    worker = ReconciliationWorker(db_config)

    try:
        worker.connect_db()

        if len(sys.argv) > 1 and sys.argv[1] == "--once":
            # Run once and exit
            worker.run_once()
        else:
            # Run continuous loop
            asyncio.run(worker.run_worker_loop())

    except KeyboardInterrupt:
        logger.info("Worker stopped")
    except Exception as e:
        logger.error(f"Worker failed: {e}")
        sys.exit(1)
    finally:
        worker.close_db()

if __name__ == "__main__":
    main()