"""
Hybrid Memory System

Integrates linked list (chronological) and dictionary index (fast lookup)
for efficient fact storage and retrieval.
"""

import logging
from typing import List, Dict, Any, Optional, Set
from datetime import datetime

from .linked_list import FactLinkedList, FactNode
from .index import KeywordIndex

logger = logging.getLogger(__name__)


class HybridMemory:
    """
    Hybrid memory system combining chronological storage with indexed retrieval
    
    Features:
    - Chronological fact storage using linked list
    - Fast O(1) keyword lookup using dictionary index
    - Fact versioning and updates
    - Integration with external sources
    - Timeline queries
    """
    
    def __init__(self):
        """Initialize the hybrid memory system"""
        self.linked_list = FactLinkedList()
        self.index = KeywordIndex()
        
        # Track fact topics for conflict detection
        self.fact_topics: Dict[str, FactNode] = {}  # topic -> latest fact node
    
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
        # Check if this fact already exists
        existing_node = self.linked_list.find_by_fact(fact)
        if existing_node and not existing_node.is_outdated:
            return {
                "status": "duplicate",
                "message": f"Fact already exists from {existing_node.timestamp}",
                "node": existing_node
            }
        
        # Extract topic for conflict detection
        topic = self._extract_topic(fact)
        
        # Check if we have a conflicting fact about the same topic
        old_node = None
        if topic and topic in self.fact_topics:
            old_node = self.fact_topics[topic]
            if old_node.fact != fact:
                # This is an update/correction
                logger.info(f"Updating fact about '{topic}': '{old_node.fact}' -> '{fact}'")
        
        # Append to linked list
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
                # Mark old fact as outdated
                old_node.mark_outdated(new_node)
                self.index.update_node(old_node, new_node)
                
            self.fact_topics[topic] = new_node
        
        result = {
            "status": "added",
            "message": f"Added fact to {category}",
            "node": new_node,
            "fact": fact
        }
        
        if old_node:
            result["updated"] = True
            result["old_fact"] = old_node.fact
        
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
        
        # Generic topic extraction (first significant noun phrase)
        words = re.findall(r'\b[a-z]{3,}\b', fact_lower)
        if words:
            return words[0]
        
        return None
    
    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for facts matching the query
        
        Args:
            query: The search query
            limit: Maximum number of results
        
        Returns:
            List of fact dictionaries
        """
        # Use index for O(1) lookup
        matching_nodes = self.index.search(query, include_outdated=False)
        
        # Convert to dictionaries and limit results
        results = []
        for node in matching_nodes[:limit]:
            result = node.to_dict()
            result["keywords_matched"] = len(
                self.index.extract_keywords(query) & 
                self.index.extract_keywords(node.fact)
            )
            results.append(result)
        
        return results
    
    def get_timeline(self, include_outdated: bool = False) -> List[Dict[str, Any]]:
        """
        Get all facts in chronological order
        
        Args:
            include_outdated: Whether to include outdated facts
        
        Returns:
            List of fact dictionaries in chronological order
        """
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
        
        return {
            "total_facts": len(all_facts),
            "active_facts": len(active_facts),
            "outdated_facts": len(all_facts) - len(active_facts),
            "indexed_keywords": self.index.size(),
            "categories": categories,
            "sources": sources,
            "oldest_fact": all_facts[0].timestamp.isoformat() if all_facts else None,
            "newest_fact": all_facts[-1].timestamp.isoformat() if all_facts else None
        }
    
    def clear(self):
        """Clear all facts from memory"""
        self.linked_list.clear()
        self.index.clear()
        self.fact_topics.clear()
        logger.info("Memory cleared")
    
    def __repr__(self) -> str:
        return f"HybridMemory(facts={self.linked_list.size()}, keywords={self.index.size()})"
