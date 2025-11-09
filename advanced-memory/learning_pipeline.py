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
import re

logger = logging.getLogger(__name__)


class SourceCredibility:
    """Manages credibility scores for different sources with enhanced validation"""

    # Enhanced credibility scores with more granular categories
    CREDIBILITY_SCORES = {
        # Primary sources (highest credibility)
        'user_direct': 0.95,          # Direct user input (explicit teaching)
        'verified_database': 0.92,    # Official databases (Wikipedia verified, official APIs)
        'academic_source': 0.90,      # University/academic publications
        'government_official': 0.88,  # Official government sources

        # Secondary sources (high credibility)
        'news_major': 0.82,           # Major news outlets (BBC, NYT, Reuters)
        'encyclopedia': 0.80,         # Encyclopedias and reference works
        'peer_reviewed': 0.78,        # Peer-reviewed journals
        'official_website': 0.75,     # Official organizational websites

        # Tertiary sources (medium credibility)
        'news_regional': 0.65,        # Regional/local news
        'educational_site': 0.62,     # Educational websites (.edu)
        'expert_blog': 0.58,          # Expert-authored blogs
        'technical_docs': 0.55,       # Technical documentation

        # User-generated (lower credibility)
        'conversation': 0.45,         # Facts learned from conversations
        'social_media': 0.35,         # Social media posts
        'forum_post': 0.32,           # Forum discussions
        'user_generated': 0.30,       # General user-generated content

        # Automated/low credibility
        'web_search': 0.25,           # General web search results
        'inferred': 0.20,             # Inferred/derived facts
        'unknown': 0.15               # Unknown or unverified sources
    }

    # Source validation patterns
    TRUSTED_DOMAINS = {
        'wikipedia.org': 'encyclopedia',
        'britannica.com': 'encyclopedia',
        'edu': 'educational_site',
        'gov': 'government_official',
        'ac.uk': 'academic_source',
        'edu.au': 'academic_source',
        'edu.ca': 'academic_source',
        'bbc.com': 'news_major',
        'bbc.co.uk': 'news_major',
        'reuters.com': 'news_major',
        'apnews.com': 'news_major',
        'nytimes.com': 'news_major',
        'washingtonpost.com': 'news_major',
        'theguardian.com': 'news_major'
    }

    @classmethod
    def get_score(cls, source: str, url: str = None) -> float:
        """
        Get credibility score for a source with URL validation

        Args:
            source: Source identifier
            url: Optional URL for domain validation

        Returns:
            Credibility score (0.0-1.0)
        """
        source_lower = source.lower()

        # Check URL domain for trusted sources
        if url:
            domain_score = cls._get_domain_credibility(url)
            if domain_score > 0:
                return domain_score

        # Check source keywords
        for key, score in cls.CREDIBILITY_SCORES.items():
            if key in source_lower:
                return score

        return cls.CREDIBILITY_SCORES['unknown']

    @classmethod
    def _get_domain_credibility(cls, url: str) -> float:
        """Get credibility score based on URL domain"""
        try:
            from urllib.parse import urlparse
            domain = urlparse(url).netloc.lower()

            # Direct domain matches
            for trusted_domain, category in cls.TRUSTED_DOMAINS.items():
                if trusted_domain in domain:
                    return cls.CREDIBILITY_SCORES.get(category, 0.5)

            # Educational domains
            if '.edu' in domain or domain.endswith('.edu'):
                return cls.CREDIBILITY_SCORES['educational_site']

            # Government domains
            if '.gov' in domain or domain.endswith('.gov'):
                return cls.CREDIBILITY_SCORES['government_official']

            # Academic domains
            if any(academic in domain for academic in ['.ac.', '.edu.', 'university', 'college']):
                return cls.CREDIBILITY_SCORES['academic_source']

        except:
            pass

        return 0.0  # No domain credibility found

    @classmethod
    def adjust_confidence(cls, base_confidence: float, source: str, url: str = None, fact_category: str = None) -> float:
        """
        Adjust confidence based on source credibility and fact characteristics

        Args:
            base_confidence: Base confidence score
            source: Source identifier
            url: Optional URL for domain validation
            fact_category: Category of the fact (affects credibility weighting)

        Returns:
            Adjusted confidence score
        """
        credibility = cls.get_score(source, url)

        # Category-specific credibility adjustments
        category_multipliers = {
            'science': 1.1,      # Science facts benefit from credible sources
            'history': 1.05,     # History facts are stable
            'biography': 1.0,    # Biographical facts vary
            'geography': 1.2,    # Geographic facts are highly verifiable
            'technology': 0.95   # Technology facts change rapidly
        }

        multiplier = category_multipliers.get(fact_category, 1.0)

        # Weighted average favoring credibility for important categories
        if fact_category in ['science', 'geography', 'history']:
            # High-stakes facts rely more on source credibility
            return (base_confidence * 0.4) + (credibility * 0.6 * multiplier)
        else:
            # General facts balance base confidence and source credibility
            return (base_confidence * 0.6) + (credibility * 0.4 * multiplier)


class FactValidator:
    """Validates facts for consistency and quality with enhanced checking"""

    # Known factual contradictions to check against
    KNOWN_CONTRADICTIONS = {
        "geography": {
            "capitals": {
                "paris is the capital of germany": False,
                "berlin is the capital of france": False,
                "london is the capital of china": False,
                "washington is the capital of russia": False,
            }
        },
        "history": {
            "world_wars": {
                "world war ii ended in 1945": True,
                "world war ii ended in 1950": False,
                "world war i ended in 1918": True,
                "world war i ended in 1920": False,
            }
        },
        "science": {
            "constants": {
                "water boils at 100 degrees celsius": True,
                "water boils at 50 degrees celsius": False,
                "water freezes at 0 degrees celsius": True,
                "water freezes at 10 degrees celsius": False,
            }
        }
    }

    @staticmethod
    def validate_fact(keyword: str, fact: str, category: str = None) -> Tuple[bool, str]:
        """
        Validate a fact for basic quality checks with category-specific validation

        Args:
            keyword: Fact keyword
            fact: Fact content
            category: Fact category for specialized validation

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
            'unclear',
            'i think',
            'i believe',
            'i feel',
            'in my opinion',
            'perhaps',
            'not sure'
        ]

        fact_lower = fact.lower()
        for pattern in suspicious_patterns:
            if pattern in fact_lower:
                return False, f"Contains uncertain language: {pattern}"

        # Category-specific validation
        if category:
            category_valid, category_reason = FactValidator._validate_category_specific(fact, category)
            if not category_valid:
                return False, category_reason

        # Check against known contradictions
        contradiction_check = FactValidator._check_known_contradictions(fact, category)
        if contradiction_check:
            return False, f"Contradicts known fact: {contradiction_check}"

        return True, "Valid"

    @staticmethod
    def _validate_category_specific(fact: str, category: str) -> Tuple[bool, str]:
        """Category-specific validation"""
        fact_lower = fact.lower()

        if category == "geography":
            # Geographic facts should contain location indicators
            location_indicators = ['capital', 'city', 'country', 'state', 'located', 'in', 'north', 'south', 'east', 'west']
            if not any(indicator in fact_lower for indicator in location_indicators):
                return False, "Geographic fact lacks location indicators"

        elif category == "history":
            # Historical facts should contain time indicators
            time_indicators = ['in', 'during', 'after', 'before', 'century', 'war', 'battle', 'revolution']
            has_time = any(indicator in fact_lower for indicator in time_indicators)
            has_year = bool(re.search(r'\b(1[0-9]{3}|2[0-9]{3})\b', fact))  # 1000-2999
            if not (has_time or has_year):
                return False, "Historical fact lacks time indicators"

        elif category == "science":
            # Scientific facts should be specific and measurable
            if len(fact.split()) < 5:
                return False, "Scientific fact too brief"

        elif category == "biography":
            # Biographical facts should reference people
            person_indicators = ['born', 'died', 'served', 'created', 'invented', 'discovered']
            if not any(indicator in fact_lower for indicator in person_indicators):
                return False, "Biographical fact lacks person indicators"

        return True, "Category validation passed"

    @staticmethod
    def _check_known_contradictions(fact: str, category: str = None) -> str:
        """Check fact against known contradictions"""
        fact_lower = fact.lower().strip()

        # Check category-specific contradictions
        if category and category in FactValidator.KNOWN_CONTRADICTIONS:
            category_contradictions = FactValidator.KNOWN_CONTRADICTIONS[category]
            for contradiction_type, contradictions in category_contradictions.items():
                for contradiction_fact, is_true in contradictions.items():
                    if contradiction_fact in fact_lower and not is_true:
                        return f"Contradicts established fact: {contradiction_fact}"

        # General contradiction checks
        general_contradictions = [
            ("earth is flat", "Earth is an oblate spheroid"),
            ("sun revolves around earth", "Earth revolves around the Sun"),
            ("humans have 3 legs", "Humans have 2 legs"),
        ]

        for contradiction, correction in general_contradictions:
            if contradiction in fact_lower:
                return f"Contradicts scientific consensus: {correction}"

        return ""

    @staticmethod
    def cross_reference_fact(fact: str, category: str, existing_facts: List[Dict]) -> Dict:
        """
        Cross-reference fact against existing knowledge

        Args:
            fact: New fact to validate
            category: Fact category
            existing_facts: List of existing facts in same category

        Returns:
            Dict with validation results
        """
        result = {
            'is_consistent': True,
            'conflicting_facts': [],
            'supporting_facts': [],
            'confidence_boost': 0.0
        }

        fact_lower = fact.lower()

        for existing in existing_facts:
            existing_fact = existing.get('fact', '').lower()
            similarity = FactValidator._calculate_similarity(fact_lower, existing_fact)

            if similarity > 0.8:  # Very similar facts
                if fact_lower == existing_fact:
                    result['supporting_facts'].append(existing)
                    result['confidence_boost'] += 0.1
                elif FactValidator._facts_contradict(fact_lower, existing_fact):
                    result['is_consistent'] = False
                    result['conflicting_facts'].append(existing)
            elif similarity > 0.5:  # Moderately similar - check for contradictions
                # Even if not very similar, check for contradictions
                if FactValidator._facts_contradict(fact_lower, existing_fact):
                    result['is_consistent'] = False
                    result['conflicting_facts'].append(existing)

        return result

    @staticmethod
    def _calculate_similarity(text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        words1 = set(text1.split())
        words2 = set(text2.split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union)

    @staticmethod
    def _facts_contradict(fact1: str, fact2: str) -> bool:
        """Check if two facts contradict each other"""
        # Simple contradiction detection
        contradiction_pairs = [
            ('is', 'is not'),
            ('was', 'was not'),
            ('are', 'are not'),
            ('were', 'were not'),
            ('has', 'does not have'),
            ('have', 'do not have'),
            ('born in', 'died in'),  # Can't be born and die in same place/time context
            ('capital of', 'not the capital of'),
        ]

        fact1_lower = fact1.lower()
        fact2_lower = fact2.lower()

        for pos, neg in contradiction_pairs:
            if pos in fact1_lower and neg in fact2_lower:
                return True
            if pos in fact2_lower and neg in fact1_lower:
                return True

        # Check for same subject with different properties (e.g., "X is A" vs "X is B")
        # Pattern: subject + "is" + property1 vs subject + "is" + property2
        is_pattern = re.compile(r'(.+?)\s+is\s+(.+)')
        match1 = is_pattern.search(fact1_lower)
        match2 = is_pattern.search(fact2_lower)

        if match1 and match2:
            subject1, property1 = match1.groups()
            subject2, property2 = match2.groups()

            # If subjects are similar and properties are different
            subject_similarity = FactValidator._calculate_similarity(subject1, subject2)
            if subject_similarity > 0.8 and property1 != property2:
                # Additional check: properties should be mutually exclusive
                exclusive_categories = [
                    ('capital of', 'capital of'),  # Same subject can't be capital of different places
                    ('president of', 'president of'),
                    ('born in', 'born in'),
                    ('located in', 'located in'),
                    ('founded in', 'founded in'),
                ]

                for cat1, cat2 in exclusive_categories:
                    if (cat1 in property1 and cat2 in property2) or (cat1 in property2 and cat2 in property1):
                        return True

        # Numeric contradictions
        numbers1 = re.findall(r'\b\d+\b', fact1_lower)
        numbers2 = re.findall(r'\b\d+\b', fact2_lower)

        if numbers1 and numbers2:
            # If facts have different numbers in similar contexts, they might contradict
            if len(set(numbers1) & set(numbers2)) == 0:  # No overlapping numbers
                # Check if they're talking about the same thing
                words1 = set(fact1_lower.split())
                words2 = set(fact2_lower.split())
                overlap = len(words1 & words2)
                total_words = len(words1 | words2)

                if overlap / total_words > 0.6:  # High word overlap with different numbers
                    return True

        return False

    @staticmethod
    def extract_keywords(fact: str) -> List[str]:
        """
        Extract potential keywords from a fact with improved filtering

        Args:
            fact: Fact content

        Returns:
            List of keywords
        """
        # Enhanced keyword extraction
        words = fact.split()

        # Filter out common words and short words
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'has', 'have', 'had',
            'to', 'of', 'in', 'on', 'at', 'for', 'by', 'with', 'as', 'and', 'or',
            'but', 'from', 'into', 'during', 'after', 'before', 'since', 'until',
            'about', 'over', 'under', 'above', 'below', 'between', 'among'
        }

        keywords = []
        for word in words:
            clean_word = word.strip('.,!?;:').lower()
            if (len(clean_word) > 3 and
                clean_word not in stopwords and
                not clean_word.isdigit() and
                not re.match(r'^\d+', clean_word)):  # Skip words starting with numbers
                keywords.append(clean_word)

        # Prioritize proper nouns and important terms
        prioritized_keywords = []
        for keyword in keywords:
            if keyword[0].isupper() or keyword in ['president', 'capital', 'university', 'company']:
                prioritized_keywords.insert(0, keyword)
            else:
                prioritized_keywords.append(keyword)

        return prioritized_keywords[:5]  # Return top 5


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
