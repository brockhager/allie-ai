#!/usr/bin/env python3
"""
Multi-Stage Learning Pipeline for Allie

Implements: Ingest → Validate → Compare → Decide → Confirm
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import asyncio

logger = logging.getLogger(__name__)


class LearningPipeline:
    """Multi-stage learning pipeline with confidence scoring and source voting"""
    
    def __init__(self, memory_db, external_sources):
        """
        Initialize learning pipeline
        
        Args:
            memory_db: AllieMemoryDB instance
            external_sources: Dict of external source functions
        """
        self.memory_db = memory_db
        self.external_sources = external_sources
        
        # Source credibility weights
        self.source_weights = {
            'wikipedia': 0.9,
            'wikidata': 0.85,
            'dbpedia': 0.8,
            'duckduckgo': 0.7,
            'user': 1.0,
            'web': 0.6,
            'unknown': 0.5
        }
    
    async def ingest_fact(self, keyword: str, fact: str, source: str, 
                         category: str = 'general') -> Dict:
        """
        Stage 1: Ingest - Accept new fact into learning queue
        
        Args:
            keyword: Fact keyword
            fact: Fact content
            source: Where it came from
            category: Fact category
            
        Returns:
            Dict with ingestion result
        """
        logger.info(f"[INGEST] Processing fact for '{keyword}' from {source}")
        
        # Calculate initial confidence based on source
        initial_confidence = self.source_weights.get(source.lower(), 0.5)
        
        # Add to learning queue
        result = self.memory_db.add_to_learning_queue(
            keyword=keyword,
            fact=fact,
            source=source,
            confidence=initial_confidence,
            category=category
        )
        
        return {
            "stage": "ingest",
            "status": "queued",
            "queue_id": result.get("queue_id"),
            "initial_confidence": initial_confidence
        }
    
    async def validate_fact(self, queue_id: int, external_check: bool = True) -> Dict:
        """
        Stage 2: Validate - Check fact against external sources
        
        Args:
            queue_id: ID of queued fact
            external_check: Whether to check external sources
            
        Returns:
            Dict with validation result
        """
        logger.info(f"[VALIDATE] Checking queue item {queue_id}")
        
        # Get queued fact
        queue = self.memory_db.get_learning_queue('pending', limit=1000)
        fact_item = next((item for item in queue if item['id'] == queue_id), None)
        
        if not fact_item:
            return {"stage": "validate", "status": "not_found"}
        
        validation_results = {
            "sources_checked": [],
            "agreements": 0,
            "disagreements": 0,
            "confidence_scores": []
        }
        
        if external_check:
            # Check against external sources
            keyword = fact_item['keyword']
            queued_fact = fact_item['fact']
            
            for source_name, source_func in self.external_sources.items():
                try:
                    # Query external source
                    external_result = await source_func(keyword)
                    
                    if external_result and external_result.get('success'):
                        validation_results["sources_checked"].append(source_name)
                        
                        # Compare facts (simplified - could use NLP similarity)
                        external_fact = external_result.get('fact', '')
                        if self._facts_agree(queued_fact, external_fact):
                            validation_results["agreements"] += 1
                            confidence = self.source_weights.get(source_name, 0.5)
                            validation_results["confidence_scores"].append(confidence)
                        else:
                            validation_results["disagreements"] += 1
                
                except Exception as e:
                    logger.warning(f"Error checking {source_name}: {e}")
        
        # Calculate validation confidence
        if validation_results["sources_checked"]:
            avg_confidence = sum(validation_results["confidence_scores"]) / len(validation_results["confidence_scores"]) if validation_results["confidence_scores"] else 0.5
            agreement_ratio = validation_results["agreements"] / (validation_results["agreements"] + validation_results["disagreements"]) if (validation_results["agreements"] + validation_results["disagreements"]) > 0 else 0.5
            
            final_confidence = (avg_confidence + agreement_ratio) / 2
        else:
            final_confidence = fact_item['confidence']
        
        # Update queue item with validated confidence
        self.memory_db.process_queue_item(queue_id, 'validate', final_confidence)
        
        return {
            "stage": "validate",
            "status": "validated",
            "queue_id": queue_id,
            "validation_results": validation_results,
            "final_confidence": final_confidence
        }
    
    async def compare_fact(self, queue_id: int) -> Dict:
        """
        Stage 3: Compare - Compare with existing memory
        
        Args:
            queue_id: ID of queued fact
            
        Returns:
            Dict with comparison result
        """
        logger.info(f"[COMPARE] Comparing queue item {queue_id} with existing memory")
        
        # Get queued fact
        queue = self.memory_db.get_learning_queue('validated', limit=1000)
        fact_item = next((item for item in queue if item['id'] == queue_id), None)
        
        if not fact_item:
            # Try pending queue
            queue = self.memory_db.get_learning_queue('pending', limit=1000)
            fact_item = next((item for item in queue if item['id'] == queue_id), None)
        
        if not fact_item:
            return {"stage": "compare", "status": "not_found"}
        
        keyword = fact_item['keyword']
        new_fact = fact_item['fact']
        new_confidence = fact_item['confidence']
        
        # Check if fact exists in memory
        existing_fact = self.memory_db.get_fact(keyword)
        
        comparison = {
            "has_existing": existing_fact is not None,
            "action": None,
            "reason": None
        }
        
        if existing_fact:
            old_fact = existing_fact['fact']
            old_confidence = existing_fact['confidence']
            
            comparison["existing_fact"] = old_fact
            comparison["existing_confidence"] = old_confidence
            comparison["new_fact"] = new_fact
            comparison["new_confidence"] = new_confidence
            
            # Decide if update is needed
            if old_fact == new_fact:
                comparison["action"] = "no_change"
                comparison["reason"] = "Facts are identical"
            elif new_confidence > old_confidence:
                comparison["action"] = "update"
                comparison["reason"] = f"New fact has higher confidence ({new_confidence} > {old_confidence})"
            elif new_confidence > 0.8 and abs(new_confidence - old_confidence) < 0.1:
                comparison["action"] = "update"
                comparison["reason"] = "New fact is recent and confidence is similar"
            else:
                comparison["action"] = "keep_existing"
                comparison["reason"] = f"Existing fact has higher confidence ({old_confidence} >= {new_confidence})"
        else:
            comparison["action"] = "add_new"
            comparison["reason"] = "No existing fact found"
        
        return {
            "stage": "compare",
            "status": "compared",
            "queue_id": queue_id,
            "comparison": comparison
        }
    
    async def decide_action(self, queue_id: int, comparison_result: Dict) -> Dict:
        """
        Stage 4: Decide - Decide what action to take
        
        Args:
            queue_id: ID of queued fact
            comparison_result: Result from compare stage
            
        Returns:
            Dict with decision
        """
        logger.info(f"[DECIDE] Making decision for queue item {queue_id}")
        
        comparison = comparison_result.get("comparison", {})
        action = comparison.get("action")
        
        decision = {
            "queue_id": queue_id,
            "action": action,
            "reason": comparison.get("reason"),
            "will_execute": True
        }
        
        # Add decision logic
        if action == "no_change":
            decision["will_execute"] = False
            decision["final_action"] = "reject"
        elif action == "keep_existing":
            decision["will_execute"] = False
            decision["final_action"] = "reject"
        elif action == "update":
            decision["will_execute"] = True
            decision["final_action"] = "update_memory"
        elif action == "add_new":
            decision["will_execute"] = True
            decision["final_action"] = "add_to_memory"
        else:
            decision["will_execute"] = False
            decision["final_action"] = "reject"
        
        return {
            "stage": "decide",
            "status": "decided",
            "decision": decision
        }
    
    async def confirm_action(self, queue_id: int, decision: Dict) -> Dict:
        """
        Stage 5: Confirm - Execute the decided action
        
        Args:
            queue_id: ID of queued fact
            decision: Decision from decide stage
            
        Returns:
            Dict with confirmation
        """
        logger.info(f"[CONFIRM] Executing action for queue item {queue_id}")
        
        final_action = decision["decision"].get("final_action")
        
        if not decision["decision"].get("will_execute"):
            # Reject the queued fact
            self.memory_db.process_queue_item(queue_id, 'reject')
            return {
                "stage": "confirm",
                "status": "rejected",
                "reason": decision["decision"].get("reason")
            }
        
        # Process the queued fact
        result = self.memory_db.process_queue_item(queue_id, 'process')
        
        return {
            "stage": "confirm",
            "status": "confirmed",
            "action": final_action,
            "result": result
        }
    
    async def process_full_pipeline(self, keyword: str, fact: str, source: str,
                                   category: str = 'general') -> Dict:
        """
        Run the complete learning pipeline
        
        Args:
            keyword: Fact keyword
            fact: Fact content
            source: Fact source
            category: Fact category
            
        Returns:
            Dict with complete pipeline results
        """
        pipeline_start = datetime.now()
        
        results = {
            "keyword": keyword,
            "fact": fact,
            "source": source,
            "stages": {}
        }
        
        # Stage 1: Ingest
        ingest_result = await self.ingest_fact(keyword, fact, source, category)
        results["stages"]["ingest"] = ingest_result
        queue_id = ingest_result.get("queue_id")
        
        if not queue_id:
            results["status"] = "failed"
            results["error"] = "Failed to ingest fact"
            return results
        
        # Stage 2: Validate
        validate_result = await self.validate_fact(queue_id, external_check=True)
        results["stages"]["validate"] = validate_result
        
        # Stage 3: Compare
        compare_result = await self.compare_fact(queue_id)
        results["stages"]["compare"] = compare_result
        
        # Stage 4: Decide
        decide_result = await self.decide_action(queue_id, compare_result)
        results["stages"]["decide"] = decide_result
        
        # Stage 5: Confirm
        confirm_result = await self.confirm_action(queue_id, decide_result)
        results["stages"]["confirm"] = confirm_result
        
        pipeline_end = datetime.now()
        results["duration_ms"] = (pipeline_end - pipeline_start).total_seconds() * 1000
        results["status"] = "completed"
        
        return results
    
    def _facts_agree(self, fact1: str, fact2: str, threshold: float = 0.7) -> bool:
        """
        Simple fact agreement check (could be enhanced with NLP)
        
        Args:
            fact1: First fact
            fact2: Second fact
            threshold: Similarity threshold
            
        Returns:
            True if facts agree
        """
        # Simple keyword overlap check
        words1 = set(fact1.lower().split())
        words2 = set(fact2.lower().split())
        
        # Remove common words
        common_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'to', 'of', 'for'}
        words1 -= common_words
        words2 -= common_words
        
        if not words1 or not words2:
            return False
        
        # Calculate Jaccard similarity
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        similarity = intersection / union if union > 0 else 0
        
        return similarity >= threshold
