"""
Hybrid Memory System

Integrates linked list (chronological) and dictionary index (fast lookup)
for efficient fact storage and retrieval.
"""

import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
import sys

from .linked_list_impl import FactLinkedList, FactNode
from .index import KeywordIndex
from .db import MemoryDB

logger = logging.getLogger(__name__)

# Import disambiguation engine from advanced-memory directory
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "advanced-memory"))
try:
    from disambiguation import DisambiguationEngine
except ImportError:
    logger.warning("DisambiguationEngine not available")
    DisambiguationEngine = None


class HybridMemory:
    """
    Hybrid memory system combining chronological storage with indexed retrieval
    
    Features:
    - Chronological fact storage using linked list
    - Fast O(1) keyword lookup using dictionary index
    - Fact versioning and updates
    - Integration with external sources
    - Timeline queries
    - Disk persistence (auto-save)
    """
    
    def __init__(self, storage_file: Optional[Path] = None):
        """
        Initialize the hybrid memory system
        
        Args:
            storage_file: Path to JSON file for persistence (default: data/hybrid_memory.json)
        """
        self.linked_list = FactLinkedList()
        self.index = KeywordIndex()
        
        # Initialize MySQL database connector
        self.db = MemoryDB()
        
        # Initialize disambiguation engine
        if DisambiguationEngine:
            self.disambiguation_engine = DisambiguationEngine()
        else:
            self.disambiguation_engine = None
        
        # Track fact topics for conflict detection
        self.fact_topics: Dict[str, FactNode] = {}  # topic -> latest fact node
        
        # Storage configuration
        if storage_file is None:
            storage_file = Path(__file__).parent.parent.parent / "data" / "hybrid_memory.json"
        self.storage_file = Path(storage_file)
        
        # Load existing facts from disk (legacy support)
        self.load_from_disk()
        
        # Load facts from MySQL into in-memory cache
        self._sync_from_mysql()
    
    def add_fact(
        self,
        fact: str,
        category: str = "general",
        confidence: float = 1.0,
        source: str = "user",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Add a new fact to memory
        
        Args:
            fact: The fact text
            category: Category of the fact
            confidence: Confidence score (0.0 to 1.0)
            source: Source of the fact (user, wikipedia, web, etc.)
            metadata: Additional metadata
        
        Returns:
            Dictionary with status and details
        """
        # Defensive check - ensure fact is actually a string
        if not isinstance(fact, str):
            logger.error(f"add_fact received non-string fact: type={type(fact)}, value={fact}")
            return {
                "status": "error",
                "message": f"Fact must be a string, got {type(fact)}"
            }
        
        # Extract topic for conflict detection and keyword indexing
        topic = self._extract_topic(fact)
        if not topic:
            # Fallback: use first significant word from fact
            import re
            # First try to find proper nouns (capitalized words that aren't "The", "A", etc.)
            words = fact.split()
            proper_nouns = []
            skip_words = {'the', 'a', 'an', 'this', 'that'}
            
            for i, word in enumerate(words):
                cleaned = word.strip('.,!?;:')
                # Look for capitals that aren't common articles
                if cleaned and cleaned[0].isupper() and cleaned.lower() not in skip_words and len(cleaned) > 2:
                    proper_nouns.append(cleaned.lower())
            
            if proper_nouns:
                topic = proper_nouns[0]
            else:
                # Fallback: extract meaningful words (length >= 5)
                words = re.findall(r'\b[a-z]{5,}\b', fact.lower())
                common_words = {'about', 'after', 'before', 'could', 'should', 'would', 'there', 'where', 
                              'their', 'which', 'these', 'those', 'because', 'during', 'through', 'between'}
                words = [w for w in words if w not in common_words]
                topic = words[0] if words else "general_fact"
        
        # Store in MySQL (authoritative source)
        db_result = self.db.add_fact(
            keyword=topic,
            fact=fact,
            source=source,
            category=category,
            confidence=confidence,
            metadata=metadata
        )
        
        # Check if MySQL operation succeeded
        if db_result["status"] == "error":
            logger.error(f"MySQL error: {db_result['message']}")
            return db_result
        
        # Also add to in-memory cache for fast access
        existing_node = self.linked_list.find_by_fact(fact)
        if existing_node and not existing_node.is_outdated:
            # Already in cache
            return {
                "status": "duplicate",
                "message": f"Fact already exists from {existing_node.timestamp}",
                "node": existing_node,
                "db_status": db_result["status"],
                "fact": existing_node.fact
            }
        
        # Check for conflicting facts
        old_node = None
        if topic and topic in self.fact_topics:
            old_node = self.fact_topics[topic]
            if old_node.fact != fact:
                logger.info(f"Updating fact about '{topic}': '{old_node.fact}' -> '{fact}'")
        
        # Add to linked list cache
        new_node = self.linked_list.append(
            fact=fact,
            category=category,
            confidence=confidence,
            source=source,
            metadata=metadata
        )
        
        # Add to keyword index
        self.index.add_node(new_node)
        
        # Update topic tracking
        if topic:
            if old_node:
                old_node.mark_outdated(new_node)
                self.index.update_node(old_node, new_node)
            self.fact_topics[topic] = new_node
        
        result = {
            "status": db_result["status"],  # "added" or "updated"
            "message": db_result["message"],
            "node": new_node,
            "fact": fact,
            "keyword": topic
        }
        
        if old_node:
            result["updated"] = True
            result["old_fact"] = old_node.fact
        
        # Auto-save to disk (legacy support - optional now that MySQL is primary)
        # self.save_to_disk()  # Disabled: MySQL is now the authoritative source

        return result
    
    def _extract_topic(self, fact: str) -> Optional[str]:
        """
        Extract the main topic from a fact for conflict detection
        
        Examples:
        - "Paris is the capital of France" -> "capital of france"
        - "Phoenix is in Arizona" -> "phoenix location"
        """
        import re
        
        fact_lower = fact.lower()
        
        # Capital patterns
        capital_match = re.search(r'capital (?:of|city of) ([a-z\s]+)', fact_lower)
        if capital_match:
            return f"capital of {capital_match.group(1).strip()}"
        
        # Location patterns
        location_match = re.search(r'([a-z]+) is (?:in|located in) ([a-z\s]+)', fact_lower)
        if location_match:
            return f"{location_match.group(1).strip()} location"
        
        # Math facts
        math_match = re.search(r'(?:square root|cube root) of ([\d.]+)', fact_lower)
        if math_match:
            return f"square root of {math_match.group(1)}"
        
        # Generic topic extraction (first significant noun phrase, skip articles)
        words = re.findall(r'\b[a-z]{3,}\b', fact_lower)
        skip_words = {'the', 'and', 'for', 'are', 'this', 'that', 'with', 'from', 'have', 'been'}
        words = [w for w in words if w not in skip_words]
        if words:
            return words[0]
        
        return None
    
    def search(self, query: str, limit: int = 5, include_disambiguation: bool = False) -> Dict[str, Any]:
        """
        Search for facts matching the query
        
        Args:
            query: The search query
            limit: Maximum number of results
            include_disambiguation: Whether to check for ambiguous queries
        
        Returns:
            Dictionary with:
            - results: List of fact dictionaries
            - disambiguation: Disambiguation info (if include_disambiguation=True)
            - fact_check_warnings: List of fact-check warnings
        """
        # First, try MySQL search (authoritative source)
        db_results = self.db.search_facts(query, limit=limit)
        
        results = []
        if db_results:
            # Convert MySQL results to expected format
            for db_fact in db_results:
                results.append({
                    "fact": db_fact["fact"],
                    "category": db_fact["category"] or "general",
                    "confidence": db_fact["confidence"] or 0.8,
                    "source": db_fact["source"] or "unknown",
                    "timestamp": db_fact["updated_at"].isoformat() if db_fact["updated_at"] else None,
                    "keywords_matched": 1,  # MySQL relevance
                    # DB-backed facts are authoritative and not marked outdated here
                    "is_outdated": False,
                    "keyword": db_fact["keyword"]
                })
        else:
            # Fallback to in-memory index search (legacy support)
            matching_nodes = self.index.search(query, include_outdated=False)
            
            # Convert to dictionaries and limit results
            for node in matching_nodes[:limit]:
                result = node.to_dict()
                result["keywords_matched"] = len(
                    self.index.extract_keywords(query) & 
                    self.index.extract_keywords(node.fact)
                )
                results.append(result)
        
        # Prepare response
        response = {
            "results": results,
            "disambiguation": None,
            "fact_check_warnings": []
        }
        
        # Check for disambiguation if requested
        if include_disambiguation and self.disambiguation_engine:
            disambiguation = self.disambiguation_engine.detect_ambiguity(query, results)
            response["disambiguation"] = disambiguation
        
        return response
    
    def get_timeline(self, include_outdated: bool = False) -> List[Dict[str, Any]]:
        """
        Get all facts in chronological order
        
        Args:
            include_outdated: Whether to include outdated facts
        
        Returns:
            List of fact dictionaries in chronological order
        """
        # Ensure we always return serializable dicts (tests expect dicts with string fields)
        return self.linked_list.get_timeline(include_outdated)
    
    def update_fact(self, old_fact: str, new_fact: str, source: str = "correction") -> Dict[str, Any]:
        """
        Update an existing fact with corrected information
        
        Args:
            old_fact: The outdated fact text
            new_fact: The corrected fact text
            source: Source of the correction
        
        Returns:
            Dictionary with update status
        """
        old_node = self.linked_list.find_by_fact(old_fact)
        
        if not old_node:
            return {
                "status": "not_found",
                "message": f"Fact '{old_fact}' not found in memory"
            }
        
        # Add the new fact
        result = self.add_fact(
            fact=new_fact,
            category=old_node.category,
            confidence=0.9,  # Slightly lower confidence for corrections
            source=source,
            metadata={"corrects": old_fact}
        )
        
        return result
    
    def remove_fact(self, fact: str) -> bool:
        """
        Remove a fact from memory (marks as outdated)
        
        Args:
            fact: The fact text to remove
        
        Returns:
            True if fact was found and removed, False otherwise
        """
        node = self.linked_list.find_by_fact(fact)
        
        if node and not node.is_outdated:
            node.is_outdated = True
            self.index.remove_node(node)
            
            # Remove from topic tracking
            for topic, tracked_node in list(self.fact_topics.items()):
                if tracked_node == node:
                    del self.fact_topics[topic]
            
            logger.info(f"Removed fact: {fact}")
            return True
        
        return False
    
    def reconcile_with_external(
        self,
        query: str,
        external_facts: List[str],
        source: str = "external"
    ) -> Dict[str, Any]:
        """
        Reconcile memory with external sources
        
        Args:
            query: The query that triggered the external search
            external_facts: Facts from external sources
            source: Name of the external source
        
        Returns:
            Dictionary with reconciliation results
        """
        results = {
            "conflicts_found": 0,
            "facts_updated": [],
            "facts_added": [],
            "memory_confirmed": []
        }
        
        # Get current memory facts for this query
        memory_facts = self.search(query, limit=10)
        
        # Process each external fact
        for ext_fact in external_facts:
            topic = self._extract_topic(ext_fact)
            
            if topic and topic in self.fact_topics:
                # We have a fact about this topic
                existing_node = self.fact_topics[topic]
                
                if existing_node.fact != ext_fact:
                    # Conflict detected
                    results["conflicts_found"] += 1
                    
                    # Prefer external source (fresher data)
                    logger.info(f"Conflict: '{existing_node.fact}' vs '{ext_fact}' (preferring external)")
                    
                    update_result = self.update_fact(
                        old_fact=existing_node.fact,
                        new_fact=ext_fact,
                        source=source
                    )
                    
                    results["facts_updated"].append({
                        "old": existing_node.fact,
                        "new": ext_fact,
                        "source": source
                    })
                else:
                    # External source confirms our memory
                    results["memory_confirmed"].append(existing_node.fact)
            else:
                # New fact from external source
                add_result = self.add_fact(
                    fact=ext_fact,
                    confidence=0.9,
                    source=source
                )
                
                if add_result["status"] == "added":
                    results["facts_added"].append(ext_fact)
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the memory system
        
        Returns:
            Dictionary with memory statistics
        """
        # Get statistics from MySQL (authoritative source)
        db_stats = self.db.get_statistics()

        # Only return DB stats if they include the expected keys used by tests
        if db_stats and isinstance(db_stats, dict) and "active_facts" in db_stats:
            return db_stats

        # Fallback to in-memory statistics (legacy)
        all_facts = list(self.linked_list.traverse(include_outdated=True))
        active_facts = [f for f in all_facts if not f.is_outdated]

        # Category breakdown
        categories = {}
        for node in active_facts:
            categories[node.category] = categories.get(node.category, 0) + 1

        # Source breakdown
        sources = {}
        for node in active_facts:
            sources[node.source] = sources.get(node.source, 0) + 1

        stats = {
            "total_facts": len(all_facts),
            "active_facts": len(active_facts),
            "outdated_facts": len(all_facts) - len(active_facts),
            "indexed_keywords": self.index.size(),
            "categories": categories,
            "sources": sources,
            "oldest_fact": all_facts[0].timestamp.isoformat() if all_facts else None,
            "newest_fact": all_facts[-1].timestamp.isoformat() if all_facts else None
        }

        # If DB returned partial stats, merge but prefer in-memory keys when tests expect them
        if db_stats and isinstance(db_stats, dict):
            merged = db_stats.copy()
            # ensure the keys tests expect exist
            for k, v in stats.items():
                merged.setdefault(k, v)
            # also ensure total_facts consistent
            merged.setdefault("total_facts", stats["total_facts"])
            return merged

        return stats
    def clear(self):
        """Clear all facts from memory"""
        self.linked_list.clear()
        self.index.clear()
        self.fact_topics.clear()
        logger.info("Memory cleared")
    
    def save_to_disk(self):
        """Save all facts to disk as JSON"""
        try:
            # Ensure directory exists
            self.storage_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Get all facts (including outdated for versioning history)
            all_facts = list(self.linked_list.traverse(include_outdated=True))
            
            # Serialize to JSON-compatible format
            facts_data = []
            for node in all_facts:
                fact_dict = node.to_dict()
                
                # Ensure fact is a string (not a FactNode)
                if not isinstance(fact_dict["fact"], str):
                    logger.error(f"Skipping non-string fact: {type(fact_dict['fact'])}")
                    continue
                
                # timestamp is already converted to ISO string by to_dict()
                # Clean up any non-serializable objects
                if fact_dict.get("updated_by"):
                    fact_dict["updated_by"] = None
                if fact_dict.get("metadata") and isinstance(fact_dict["metadata"], dict):
                    # Remove any FactNode references from metadata
                    fact_dict["metadata"] = {
                        k: str(v) if not isinstance(v, (str, int, float, bool, type(None))) else v
                        for k, v in fact_dict["metadata"].items()
                    }
                facts_data.append(fact_dict)
            
            # Write to file
            with open(self.storage_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "version": "1.0",
                    "saved_at": datetime.now().isoformat(),
                    "fact_count": len(facts_data),
                    "facts": facts_data
                }, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {len(facts_data)} facts to {self.storage_file}")
            
        except Exception as e:
            logger.error(f"Failed to save memory to disk: {e}")
            # Debug: print problematic fact
            if facts_data:
                logger.error(f"Last fact being saved: {facts_data[-1]}")
                for key, value in facts_data[-1].items():
                    logger.error(f"  {key}: {type(value)}")
    
    def load_from_disk(self):
        """Load facts from disk"""
        try:
            if not self.storage_file.exists():
                logger.info(f"No existing memory file at {self.storage_file}")
                return
            
            with open(self.storage_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            facts = data.get("facts", [])

            # Restore facts in chronological order, preserving timestamps
            for fact_dict in facts:
                # Convert ISO string back to datetime
                timestamp = None
                try:
                    timestamp = datetime.fromisoformat(fact_dict["timestamp"])
                except Exception:
                    timestamp = None

                # Append into linked list preserving timestamp
                appended_node = self.linked_list.append(
                    fact=fact_dict["fact"],
                    timestamp=timestamp,
                    category=fact_dict.get("category", "general"),
                    confidence=fact_dict.get("confidence", 1.0),
                    source=fact_dict.get("source", "user"),
                    metadata=fact_dict.get("metadata")
                )

                # Set outdated flag if present
                appended_node.is_outdated = fact_dict.get("is_outdated", False)

                # Add to index (only if not outdated)
                if not appended_node.is_outdated:
                    self.index.add_node(appended_node)

                    # Track topics
                    topic = self._extract_topic(appended_node.fact)
                    if topic:
                        self.fact_topics[topic] = appended_node
            
            logger.info(f"Loaded {len(facts)} facts from {self.storage_file}")
            
        except Exception as e:
            logger.error(f"Failed to load memory from disk: {e}")
    
    def _sync_from_mysql(self):
        """Load facts from MySQL into in-memory cache on startup"""
        try:
            # Get all facts from MySQL timeline
            db_facts = self.db.timeline(limit=1000)  # Load recent 1000 facts
            
            if not db_facts:
                logger.info("No facts in MySQL database")
                return
            
            logger.info(f"Syncing {len(db_facts)} facts from MySQL to memory cache...")
            
            for db_fact in db_facts:
                # Add to linked list using proper parameters
                node = self.linked_list.append(
                    fact=db_fact["fact"],  # Pass string, not FactNode
                    category=db_fact["category"] or "general",
                    confidence=db_fact["confidence"] or 0.8,
                    source=db_fact["source"] or "unknown",
                    metadata=db_fact.get("metadata")
                )
                
                # Add to index
                self.index.add_node(node)
                
                # Track topics
                topic = db_fact["keyword"]
                if topic:
                    self.fact_topics[topic] = node
            
            logger.info(f"Successfully synced {len(db_facts)} facts from MySQL")
            
        except Exception as e:
            logger.error(f"Failed to sync from MySQL: {e}")
    
    def __repr__(self) -> str:
        return f"HybridMemory(facts={self.linked_list.size()}, keywords={self.index.size()})"
