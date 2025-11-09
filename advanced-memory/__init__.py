"""
Advanced Memory System for Allie

This module provides persistent, self-correcting memory with:
- MySQL-based storage with confidence scoring
- 5-stage learning pipeline with conflict resolution
- Learning queue for uncertain facts
- Fact clustering and organization
- Complete audit trail

Main Components:
- db.py: AllieMemoryDB - MySQL connector with full CRUD operations
- learning_pipeline.py: Intelligent fact processing with validation
- hybrid.py: Legacy hybrid memory interface (for backward compatibility)
- linked_list.py: Legacy linked-list memory structure
- index.py: Memory indexing utilities
"""

from .db import AllieMemoryDB
from .learning_pipeline import LearningPipeline

__version__ = "2.0.0"
__all__ = ["AllieMemoryDB", "LearningPipeline"]
