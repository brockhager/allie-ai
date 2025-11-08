"""
Dictionary Index for Fast Fact Retrieval

Maps keywords to linked list nodes for O(1) lookup.
Supports multiple nodes per keyword (different facts containing the same keyword).
"""

import re
from typing import List, Set, Dict, Optional
from collections import defaultdict


class KeywordIndex:
    """Dictionary-based index for fast keyword lookup"""
    
    def __init__(self):
        """Initialize the keyword index"""
        # Maps keyword -> list of FactNodes containing that keyword
        self.index: Dict[str, List] = defaultdict(list)
        
        # Common stop words to ignore during indexing
        self.stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'this', 'these', 'those', 'their',
            'there', 'they', 'we', 'you', 'what', 'when', 'where', 'who',
            'which', 'how', 'about', 'also', 'can', 'could', 'would', 'should'
        }
    
    def extract_keywords(self, text: str) -> Set[str]:
        """
        Extract keywords from text
        
        Args:
            text: The text to extract keywords from
        
        Returns:
            Set of lowercase keywords (excluding stop words)
        """
        # Convert to lowercase and extract words
        words = re.findall(r'\b[a-z]+\b', text.lower())
        
        # Filter out stop words and very short words
        keywords = {
            word for word in words 
            if word not in self.stop_words and len(word) > 2
        }
        
        return keywords
    
    def add_node(self, node, fact_text: Optional[str] = None):
        """
        Add a node to the index
        
        Args:
            node: The FactNode to index
            fact_text: Optional explicit fact text (uses node.fact if not provided)
        """
        text = fact_text or node.fact
        keywords = self.extract_keywords(text)
        
        for keyword in keywords:
            # Avoid duplicate entries
            if node not in self.index[keyword]:
                self.index[keyword].append(node)
    
    def remove_node(self, node, fact_text: Optional[str] = None):
        """
        Remove a node from the index
        
        Args:
            node: The FactNode to remove
            fact_text: Optional explicit fact text (uses node.fact if not provided)
        """
        text = fact_text or node.fact
        keywords = self.extract_keywords(text)
        
        for keyword in keywords:
            if keyword in self.index and node in self.index[keyword]:
                self.index[keyword].remove(node)
                # Clean up empty lists
                if not self.index[keyword]:
                    del self.index[keyword]
    
    def update_node(self, old_node, new_node):
        """
        Update index when a fact is superseded
        
        Args:
            old_node: The outdated FactNode
            new_node: The new FactNode
        """
        # Remove old node from index
        self.remove_node(old_node)
        
        # Add new node to index
        self.add_node(new_node)
    
    def search(self, query: str, include_outdated: bool = False) -> List:
        """
        Search for nodes matching query keywords
        
        Args:
            query: The search query
            include_outdated: Whether to include outdated facts
        
        Returns:
            List of matching FactNodes, sorted by relevance
        """
        query_keywords = self.extract_keywords(query)
        
        if not query_keywords:
            return []
        
        # Score nodes by number of matching keywords
        node_scores: Dict = defaultdict(int)
        
        for keyword in query_keywords:
            if keyword in self.index:
                for node in self.index[keyword]:
                    # Skip outdated facts unless explicitly requested
                    if not include_outdated and node.is_outdated:
                        continue
                    node_scores[node] += 1
        
        # Sort by score (descending) and then by timestamp (newest first)
        sorted_nodes = sorted(
            node_scores.items(),
            key=lambda x: (x[1], x[0].timestamp),
            reverse=True
        )
        
        return [node for node, score in sorted_nodes]
    
    def get_all_keywords(self) -> Set[str]:
        """Get all indexed keywords"""
        return set(self.index.keys())
    
    def get_nodes_for_keyword(self, keyword: str, include_outdated: bool = False) -> List:
        """
        Get all nodes containing a specific keyword
        
        Args:
            keyword: The keyword to search for
            include_outdated: Whether to include outdated facts
        
        Returns:
            List of FactNodes containing the keyword
        """
        keyword_lower = keyword.lower()
        nodes = self.index.get(keyword_lower, [])
        
        if not include_outdated:
            nodes = [node for node in nodes if not node.is_outdated]
        
        return nodes
    
    def clear(self):
        """Clear the entire index"""
        self.index.clear()
    
    def size(self) -> int:
        """Get the number of unique keywords in the index"""
        return len(self.index)
    
    def __repr__(self) -> str:
        return f"KeywordIndex(keywords={len(self.index)}, total_entries={sum(len(nodes) for nodes in self.index.values())})"
