"""
Linked List Implementation for Chronological Fact Storage

Each node contains:
- Fact text
- Timestamp
- Metadata (category, confidence, source)
- Pointer to next node
"""

from datetime import datetime
from typing import Optional, Any, Dict


class FactNode:
    """Node in the chronological fact linked list"""
    
    def __init__(
        self, 
        fact: str, 
        timestamp: Optional[datetime] = None,
        category: str = "general",
        confidence: float = 1.0,
        source: str = "user",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a fact node
        
        Args:
            fact: The fact text
            timestamp: When the fact was learned (defaults to now)
            category: Category of the fact (geography, science, etc.)
            confidence: Confidence score (0.0 to 1.0)
            source: Where the fact came from (user, wikipedia, web, etc.)
            metadata: Additional metadata about the fact
        """
        self.fact = fact
        self.timestamp = timestamp or datetime.now()
        self.category = category
        self.confidence = confidence
        self.source = source
        self.metadata = metadata or {}
        self.next: Optional['FactNode'] = None
        
        # Track if this fact has been superseded by a newer version
        self.is_outdated = False
        self.updated_by: Optional['FactNode'] = None
    
    def mark_outdated(self, updated_node: 'FactNode'):
        """Mark this fact as outdated and link to the updated version"""
        self.is_outdated = True
        self.updated_by = updated_node
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary for serialization"""
        return {
            "fact": self.fact,
            "timestamp": self.timestamp.isoformat(),
            "category": self.category,
            "confidence": self.confidence,
            "source": self.source,
            "metadata": self.metadata,
            "is_outdated": self.is_outdated
        }
    
    def __repr__(self) -> str:
        status = "[OUTDATED]" if self.is_outdated else ""
        return f"FactNode({status} {self.fact[:50]}... [{self.category}] @ {self.timestamp})"


class FactLinkedList:
    """Chronological linked list of facts"""
    
    def __init__(self):
        """Initialize empty linked list"""
        self.head: Optional[FactNode] = None
        self.tail: Optional[FactNode] = None
        self._size = 0
    
    def append(
        self, 
        fact: str,
        category: str = "general",
        confidence: float = 1.0,
        source: str = "user",
        metadata: Optional[Dict[str, Any]] = None
    ) -> FactNode:
        """
        Append a new fact to the end of the list
        
        Args:
            fact: The fact text
            category: Category of the fact
            confidence: Confidence score
            source: Source of the fact
            metadata: Additional metadata
        
        Returns:
            The newly created FactNode
        """
        new_node = FactNode(
            fact=fact,
            category=category,
            confidence=confidence,
            source=source,
            metadata=metadata
        )
        
        if self.head is None:
            # First node
            self.head = new_node
            self.tail = new_node
        else:
            # Append to tail
            self.tail.next = new_node
            self.tail = new_node
        
        self._size += 1
        return new_node
    
    def traverse(self, include_outdated: bool = False):
        """
        Traverse the linked list from head to tail
        
        Args:
            include_outdated: Whether to include outdated facts
        
        Yields:
            FactNode objects in chronological order
        """
        current = self.head
        while current is not None:
            if include_outdated or not current.is_outdated:
                yield current
            current = current.next
    
    def get_timeline(self, include_outdated: bool = False) -> list:
        """
        Get all facts in chronological order
        
        Args:
            include_outdated: Whether to include outdated facts
        
        Returns:
            List of fact dictionaries in chronological order
        """
        return [node.to_dict() for node in self.traverse(include_outdated)]
    
    def find_by_fact(self, fact_text: str) -> Optional[FactNode]:
        """
        Find a node by exact fact text match
        
        Args:
            fact_text: The fact text to search for
        
        Returns:
            The FactNode if found, None otherwise
        """
        for node in self.traverse(include_outdated=True):
            if node.fact == fact_text:
                return node
        return None
    
    def size(self) -> int:
        """Get the number of facts in the list"""
        return self._size
    
    def is_empty(self) -> bool:
        """Check if the list is empty"""
        return self.head is None
    
    def clear(self):
        """Clear all facts from the list"""
        self.head = None
        self.tail = None
        self._size = 0
    
    def __len__(self) -> int:
        return self._size
    
    def __repr__(self) -> str:
        return f"FactLinkedList(size={self._size})"
