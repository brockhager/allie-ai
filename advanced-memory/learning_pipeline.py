#!/usr/bin/env python3
"""
Learning Pipeline for Allie's Advanced Memory System

5-Stage Pipeline: Ingest → Validate → Compare → Decide → Confirm

Handles incoming facts with source credibility weighting, validation,
conflict resolution, and self-correction.
"""

from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class SourceCredibility:
    """Manages credibility scores for different sources"""
    
    # Default credibility scores (0.0 - 1.0)
    CREDIBILITY_SCORES = {
        'user': 0.9,          # Direct user input is highly trusted
        'conversation': 0.85,  # Facts learned from conversations
        'web_search': 0.7,     # Web search results
        'external_api': 0.75,  # External API data
        'inference': 0.6,      # Inferred/derived facts
        'quick_teach': 0.95,   # Bulk teaching mode (user-verified)
        'unknown': 0.5         # Unknown sources
    }
    
    @classmethod
    def get_score(cls, source: str) -> float:
        """
        Get credibility score for a source
        
        Args:
            source: Source identifier
            
        Returns:
            Credibility score (0.0-1.0)
        """
        source_lower = source.lower()
        for key, score in cls.CREDIBILITY_SCORES.items():
            if key in source_lower:
                return score
        return cls.CREDIBILITY_SCORES['unknown']
    
    @classmethod
    def adjust_confidence(cls, base_confidence: float, source: str) -> float:
        """
        Adjust confidence based on source credibility
        
        Args:
            base_confidence: Base confidence score
            source: Source identifier
            
        Returns:
            Adjusted confidence score
        """
        credibility = cls.get_score(source)
        # Weighted average favoring credibility slightly
        return (base_confidence * 0.6) + (credibility * 0.4)


class FactValidator:
    """Validates facts for consistency and quality"""
    
    @staticmethod
    def validate_fact(keyword: str, fact: str) -> Tuple[bool, str]:
        """
        Validate a fact for basic quality checks
        
        Args:
            keyword: Fact keyword
            fact: Fact content
            
        Returns:
            Tuple of (is_valid, reason)
        """
        # Check for empty content
        if not keyword or not keyword.strip():
            return False, "Empty keyword"
        
        if not fact or not fact.strip():
            return False, "Empty fact"
        
        # Check minimum length
        if len(fact.strip()) < 3:
            return False, "Fact too short"
        
        # Check for suspicious patterns
        suspicious_patterns = [
            'i don\'t know',
            'i\'m not sure',
            'maybe',
            'possibly',
            'unclear'
        ]
        
        fact_lower = fact.lower()
        for pattern in suspicious_patterns:
            if pattern in fact_lower:
                return False, f"Contains uncertain language: {pattern}"
        
        return True, "Valid"
    
    @staticmethod
    def extract_keywords(fact: str) -> List[str]:
        """
        Extract potential keywords from a fact
        
        Args:
            fact: Fact content
            
        Returns:
            List of keywords
        """
        # Simple keyword extraction (could be enhanced with NLP)
        words = fact.split()
        # Filter out common words
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'has', 'have', 'had', 'to', 'of', 'in', 'on', 'at', 'for'}
        keywords = [w.strip('.,!?;:').lower() for w in words if len(w) > 3 and w.lower() not in stopwords]
        return keywords[:5]  # Return top 5


class ConflictResolver:
    """Resolves conflicts between existing and new facts"""
    
    @staticmethod
    def detect_conflict(old_fact: str, new_fact: str, threshold: float = 0.3) -> bool:
        """
        Detect if two facts conflict
        
        Args:
            old_fact: Existing fact
            new_fact: New fact
            threshold: Similarity threshold (lower = more different)
            
        Returns:
            True if facts conflict
        """
        # Simple word-overlap based similarity
        old_words = set(old_fact.lower().split())
        new_words = set(new_fact.lower().split())
        
        if not old_words or not new_words:
            return False
        
        intersection = old_words & new_words
        union = old_words | new_words
        
        similarity = len(intersection) / len(union) if union else 0
        
        # If very different content about same keyword, it's a conflict
        return similarity < threshold
    
    @staticmethod
    def resolve(old_fact: Dict, new_fact_data: Dict) -> Dict:
        """
        Resolve conflict between old and new fact
        
        Args:
            old_fact: Existing fact dict with confidence
            new_fact_data: New fact data with confidence
            
        Returns:
            Dict with resolution decision
        """
        old_confidence = old_fact.get('confidence', 0.5)
        new_confidence = new_fact_data.get('confidence', 0.5)
        
        # Resolution strategy
        if new_confidence > old_confidence + 0.2:
            # New fact significantly more confident
            return {
                'action': 'replace',
                'reason': 'Higher confidence in new fact',
                'chosen_fact': new_fact_data['fact'],
                'confidence': new_confidence
            }
        elif old_confidence > new_confidence + 0.2:
            # Old fact significantly more confident
            return {
                'action': 'keep',
                'reason': 'Higher confidence in existing fact',
                'chosen_fact': old_fact['fact'],
                'confidence': old_confidence
            }
        else:
            # Similar confidence - merge or flag for review
            return {
                'action': 'merge',
                'reason': 'Similar confidence - needs review',
                'old_fact': old_fact['fact'],
                'new_fact': new_fact_data['fact'],
                'confidence': max(old_confidence, new_confidence)
            }


class LearningPipeline:
    """
    5-Stage Learning Pipeline for processing new facts
    
    Stages:
    1. Ingest - Receive and validate input
    2. Validate - Check quality and consistency
    3. Compare - Compare with existing knowledge
    4. Decide - Make decision on conflicts
    5. Confirm - Apply changes to memory
    """
    
    def __init__(self, memory_db):
        """
        Initialize pipeline
        
        Args:
            memory_db: AllieMemoryDB instance
        """
        self.memory_db = memory_db
        self.validator = FactValidator()
        self.resolver = ConflictResolver()
    
    def process_fact(self, keyword: str, fact: str, source: str,
                    base_confidence: float = 0.7, category: str = 'general',
                    auto_resolve: bool = True) -> Dict:
        """
        Process a single fact through the complete pipeline
        
        Args:
            keyword: Fact keyword
            fact: Fact content
            source: Fact source
            base_confidence: Base confidence score
            category: Fact category
            auto_resolve: Whether to auto-resolve conflicts
            
        Returns:
            Dict with processing results
        """
        results = {
            'keyword': keyword,
            'stages': {},
            'final_status': None,
            'confidence': base_confidence
        }
        
        # Stage 1: Ingest
        ingest_result = self._stage_ingest(keyword, fact, source, base_confidence)
        results['stages']['ingest'] = ingest_result
        
        if not ingest_result['valid']:
            results['final_status'] = 'rejected_at_ingest'
            return results
        
        # Stage 2: Validate
        validate_result = self._stage_validate(keyword, fact)
        results['stages']['validate'] = validate_result
        
        if not validate_result['valid']:
            results['final_status'] = 'rejected_at_validation'
            return results
        
        # Stage 3: Compare
        compare_result = self._stage_compare(keyword, fact)
        results['stages']['compare'] = compare_result
        results['confidence'] = ingest_result['adjusted_confidence']
        
        # Stage 4: Decide
        decide_result = self._stage_decide(
            keyword, fact, source,
            ingest_result['adjusted_confidence'],
            compare_result,
            auto_resolve
        )
        results['stages']['decide'] = decide_result
        
        if not auto_resolve and decide_result['action'] == 'queue':
            results['final_status'] = 'queued_for_review'
            return results
        
        # Stage 5: Confirm
        confirm_result = self._stage_confirm(
            keyword, fact, source,
            decide_result['confidence'],
            category,
            decide_result['action']
        )
        results['stages']['confirm'] = confirm_result
        results['final_status'] = confirm_result['status']
        
        return results
    
    def _stage_ingest(self, keyword: str, fact: str, source: str,
                     base_confidence: float) -> Dict:
        """Stage 1: Ingest and adjust confidence based on source"""
        adjusted_confidence = SourceCredibility.adjust_confidence(base_confidence, source)
        
        return {
            'stage': 'ingest',
            'valid': True,
            'source': source,
            'source_credibility': SourceCredibility.get_score(source),
            'base_confidence': base_confidence,
            'adjusted_confidence': adjusted_confidence
        }
    
    def _stage_validate(self, keyword: str, fact: str) -> Dict:
        """Stage 2: Validate fact quality"""
        is_valid, reason = self.validator.validate_fact(keyword, fact)
        
        return {
            'stage': 'validate',
            'valid': is_valid,
            'reason': reason,
            'extracted_keywords': self.validator.extract_keywords(fact) if is_valid else []
        }
    
    def _stage_compare(self, keyword: str, fact: str) -> Dict:
        """Stage 3: Compare with existing knowledge"""
        existing_fact = self.memory_db.get_fact(keyword)
        
        if not existing_fact:
            return {
                'stage': 'compare',
                'existing_fact': None,
                'has_conflict': False,
                'is_new': True
            }
        
        has_conflict = self.resolver.detect_conflict(existing_fact['fact'], fact)
        
        return {
            'stage': 'compare',
            'existing_fact': existing_fact,
            'has_conflict': has_conflict,
            'is_new': False
        }
    
    def _stage_decide(self, keyword: str, fact: str, source: str,
                     confidence: float, compare_result: Dict,
                     auto_resolve: bool) -> Dict:
        """Stage 4: Decide on action"""
        
        if compare_result['is_new']:
            # No existing fact, add it
            return {
                'stage': 'decide',
                'action': 'add',
                'reason': 'New fact',
                'confidence': confidence
            }
        
        if not compare_result['has_conflict']:
            # No conflict, update if different
            existing_fact = compare_result['existing_fact']
            if existing_fact['fact'] != fact:
                return {
                    'stage': 'decide',
                    'action': 'update',
                    'reason': 'Refinement of existing fact',
                    'confidence': max(confidence, existing_fact['confidence'])
                }
            else:
                return {
                    'stage': 'decide',
                    'action': 'skip',
                    'reason': 'Identical to existing fact',
                    'confidence': existing_fact['confidence']
                }
        
        # Conflict detected
        if not auto_resolve:
            # Queue for manual review
            queue_result = self.memory_db.add_to_learning_queue(
                keyword, fact, source, confidence
            )
            return {
                'stage': 'decide',
                'action': 'queue',
                'reason': 'Conflict detected, queued for review',
                'queue_id': queue_result.get('queue_id'),
                'confidence': confidence
            }
        
        # Auto-resolve conflict
        resolution = self.resolver.resolve(
            compare_result['existing_fact'],
            {'fact': fact, 'confidence': confidence}
        )
        
        return {
            'stage': 'decide',
            'action': resolution['action'],
            'reason': resolution['reason'],
            'confidence': resolution['confidence'],
            'resolution': resolution
        }
    
    def _stage_confirm(self, keyword: str, fact: str, source: str,
                      confidence: float, category: str, action: str) -> Dict:
        """Stage 5: Confirm and apply changes"""
        
        if action == 'skip':
            return {
                'stage': 'confirm',
                'status': 'skipped',
                'reason': 'No changes needed'
            }
        
        if action == 'add':
            result = self.memory_db.add_fact(keyword, fact, source, confidence, category, 'not_verified')
            return {
                'stage': 'confirm',
                'status': 'added',
                'fact_id': result.get('fact_id'),
                'keyword': keyword
            }
        
        if action in ['update', 'replace']:
            result = self.memory_db.update_fact(keyword, fact, source, confidence)
            return {
                'stage': 'confirm',
                'status': 'updated',
                'keyword': keyword
            }
        
        if action == 'merge':
            # Queue for manual merge
            queue_result = self.memory_db.add_to_learning_queue(
                keyword, fact, source, confidence
            )
            return {
                'stage': 'confirm',
                'status': 'queued_for_merge',
                'queue_id': queue_result.get('queue_id')
            }
        
        return {
            'stage': 'confirm',
            'status': 'unknown_action',
            'action': action
        }
    
    def process_batch(self, facts: List[Dict], auto_resolve: bool = True) -> Dict:
        """
        Process multiple facts in batch
        
        Args:
            facts: List of fact dicts with keys: keyword, fact, source, confidence, category
            auto_resolve: Whether to auto-resolve conflicts
            
        Returns:
            Dict with batch processing results
        """
        results = {
            'total': len(facts),
            'added': 0,
            'updated': 0,
            'skipped': 0,
            'queued': 0,
            'rejected': 0,
            'details': []
        }
        
        for fact_data in facts:
            result = self.process_fact(
                keyword=fact_data.get('keyword', ''),
                fact=fact_data.get('fact', ''),
                source=fact_data.get('source', 'batch_import'),
                base_confidence=fact_data.get('confidence', 0.7),
                category=fact_data.get('category', 'general'),
                auto_resolve=auto_resolve
            )
            
            # Count results
            status = result['final_status']
            if 'rejected' in status:
                results['rejected'] += 1
            elif status == 'queued_for_review':
                results['queued'] += 1
            elif 'added' in result['stages'].get('confirm', {}).get('status', ''):
                results['added'] += 1
            elif 'updated' in result['stages'].get('confirm', {}).get('status', ''):
                results['updated'] += 1
            else:
                results['skipped'] += 1
            
            results['details'].append({
                'keyword': fact_data.get('keyword'),
                'status': status,
                'confidence': result.get('confidence')
            })
        
        return results
    
    def review_queued_item(self, queue_id: int, action: str = 'validate') -> Dict:
        """
        Review and process a queued item
        
        Args:
            queue_id: Queue item ID
            action: 'validate', 'reject', or 'process'
            
        Returns:
            Dict with review results
        """
        return self.memory_db.process_queue_item(queue_id, action)
    
    def get_pipeline_stats(self) -> Dict:
        """Get statistics about pipeline processing"""
        stats = self.memory_db.get_statistics()
        
        # Add pipeline-specific stats
        queue_status = stats.get('queue_status', {})
        
        return {
            'total_facts': stats.get('total_facts', 0),
            'average_confidence': stats.get('average_confidence', 0),
            'pending_review': queue_status.get('pending', 0),
            'validated': queue_status.get('validated', 0),
            'rejected': queue_status.get('rejected', 0),
            'processed': queue_status.get('processed', 0),
            'learning_log_entries': stats.get('learning_log_entries', 0),
            'categories': stats.get('by_category', {}),
            'sources': stats.get('by_source', {})
        }
