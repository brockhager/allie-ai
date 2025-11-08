#!/usr/bin/env python3
"""
Experience Replay Buffer for Allie Learning System

Manages historical conversation data for replay-based continual learning.
"""

import json
import random
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime, timedelta
import hashlib

logger = logging.getLogger(__name__)

class ExperienceReplayBuffer:
    """Manages historical training data for experience replay"""

    def __init__(self, buffer_path: str, max_size: int = 5000, quality_threshold: float = 0.7):
        self.buffer_path = Path(buffer_path)
        self.max_size = max_size
        self.quality_threshold = quality_threshold
        self.buffer: List[Dict[str, Any]] = []
        self._load_buffer()

    def _load_buffer(self):
        """Load existing replay buffer from disk"""
        if self.buffer_path.exists():
            try:
                with open(self.buffer_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.buffer = data.get('conversations', [])
                    logger.info(f"Loaded {len(self.buffer)} conversations from replay buffer")
            except Exception as e:
                logger.error(f"Failed to load replay buffer: {e}")
                self.buffer = []
        else:
            self.buffer = []
            logger.info("No existing replay buffer found, starting fresh")

    def _save_buffer(self):
        """Save replay buffer to disk"""
        try:
            data = {
                'metadata': {
                    'created': datetime.now().isoformat(),
                    'size': len(self.buffer),
                    'max_size': self.max_size,
                    'quality_threshold': self.quality_threshold
                },
                'conversations': self.buffer
            }

            # Create backup of old buffer
            if self.buffer_path.exists():
                backup_path = self.buffer_path.with_suffix('.backup')
                self.buffer_path.rename(backup_path)

            # Save new buffer
            with open(self.buffer_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved {len(self.buffer)} conversations to replay buffer")

        except Exception as e:
            logger.error(f"Failed to save replay buffer: {e}")
            # Restore backup if save failed
            backup_path = self.buffer_path.with_suffix('.backup')
            if backup_path.exists():
                backup_path.rename(self.buffer_path)

    def add_conversation(self, conversation: Dict[str, Any], quality_score: float):
        """
        Add a conversation to the replay buffer

        Args:
            conversation: Dict with 'prompt' and 'response' keys
            quality_score: Quality score from 0-1
        """
        if quality_score < self.quality_threshold:
            logger.debug(f"Conversation rejected: quality {quality_score:.2f} below threshold {self.quality_threshold}")
            return

        # Add metadata
        entry = {
            'id': self._generate_id(conversation),
            'conversation': conversation,
            'quality_score': quality_score,
            'added_at': datetime.now().isoformat(),
            'usage_count': 0
        }

        # Check for duplicates
        existing_ids = {item['id'] for item in self.buffer}
        if entry['id'] in existing_ids:
            logger.debug("Conversation already in buffer, skipping")
            return

        # Add to buffer
        self.buffer.append(entry)

        # Maintain size limit (remove oldest, lowest quality items first)
        if len(self.buffer) > self.max_size:
            self._trim_buffer()

        logger.debug(f"Added conversation to replay buffer (quality: {quality_score:.2f})")

    def sample_replay_data(self, size: int, exclude_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Sample conversations from replay buffer for training

        Args:
            size: Number of conversations to sample
            exclude_ids: IDs to exclude from sampling

        Returns:
            List of conversation dicts with 'prompt' and 'response' keys
        """
        if not self.buffer:
            logger.warning("Replay buffer is empty")
            return []

        # Filter out excluded conversations
        available = self.buffer
        if exclude_ids:
            available = [item for item in self.buffer if item['id'] not in exclude_ids]

        if len(available) == 0:
            logger.warning("No available conversations after filtering")
            return []

        # Sample with quality-weighted probability
        # Higher quality conversations more likely to be selected
        weights = [item['quality_score'] for item in available]
        sample_size = min(size, len(available))

        try:
            sampled_items = random.choices(available, weights=weights, k=sample_size)

            # Update usage counts
            for item in sampled_items:
                item['usage_count'] += 1
                item['last_used'] = datetime.now().isoformat()

            # Extract conversations
            conversations = [item['conversation'] for item in sampled_items]

            logger.info(f"Sampled {len(conversations)} conversations from replay buffer")
            return conversations

        except Exception as e:
            logger.error(f"Failed to sample replay data: {e}")
            # Fallback to random sampling without weights
            sampled_items = random.sample(available, min(size, len(available)))
            return [item['conversation'] for item in sampled_items]

    def get_buffer_stats(self) -> Dict[str, Any]:
        """Get statistics about the replay buffer"""
        if not self.buffer:
            return {'size': 0, 'avg_quality': 0, 'topics': {}}

        qualities = [item['quality_score'] for item in self.buffer]
        usage_counts = [item['usage_count'] for item in self.buffer]

        # Simple topic extraction (could be more sophisticated)
        topics = {}
        for item in self.buffer:
            prompt = item['conversation'].get('prompt', '').lower()
            # Very basic topic detection
            if any(word in prompt for word in ['what', 'how', 'why', 'explain']):
                topic = 'questions'
            elif any(word in prompt for word in ['tell', 'story', 'describe']):
                topic = 'narrative'
            else:
                topic = 'general'

            topics[topic] = topics.get(topic, 0) + 1

        return {
            'size': len(self.buffer),
            'max_size': self.max_size,
            'avg_quality': sum(qualities) / len(qualities),
            'quality_distribution': {
                'excellent': len([q for q in qualities if q >= 0.9]),
                'good': len([q for q in qualities if 0.7 <= q < 0.9]),
                'fair': len([q for q in qualities if 0.5 <= q < 0.7]),
                'poor': len([q for q in qualities if q < 0.5])
            },
            'avg_usage': sum(usage_counts) / len(usage_counts),
            'topics': topics,
            'oldest_entry': min(item['added_at'] for item in self.buffer),
            'newest_entry': max(item['added_at'] for item in self.buffer)
        }

    def _trim_buffer(self):
        """Trim buffer to maintain size limit"""
        if len(self.buffer) <= self.max_size:
            return

        # Sort by quality (descending) and then by usage (ascending)
        # Keep higher quality, less-used items
        self.buffer.sort(key=lambda x: (x['quality_score'], -x['usage_count']), reverse=True)

        # Remove excess items
        excess = len(self.buffer) - self.max_size
        removed = self.buffer[-excess:]
        self.buffer = self.buffer[:-excess]

        logger.info(f"Trimmed {len(removed)} low-quality conversations from replay buffer")

    def _generate_id(self, conversation: Dict[str, Any]) -> str:
        """Generate unique ID for conversation"""
        content = f"{conversation.get('prompt', '')}|{conversation.get('response', '')}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()[:16]

    def cleanup_old_entries(self, max_age_days: int = 90):
        """Remove entries older than specified days"""
        cutoff = datetime.now() - timedelta(days=max_age_days)
        original_size = len(self.buffer)

        self.buffer = [
            item for item in self.buffer
            if datetime.fromisoformat(item['added_at']) > cutoff
        ]

        removed = original_size - len(self.buffer)
        if removed > 0:
            logger.info(f"Cleaned up {removed} old entries from replay buffer")
            self._save_buffer()

    def export_for_analysis(self, output_path: str):
        """Export buffer data for analysis"""
        stats = self.get_buffer_stats()
        stats['conversations'] = self.buffer

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        logger.info(f"Exported replay buffer analysis to {output_path}")

def main():
    """Example usage"""
    # Initialize buffer
    buffer = ExperienceReplayBuffer("data/replay_buffer.json", max_size=100)

    # Add some example conversations
    conversations = [
        {"prompt": "What is the capital of France?", "response": "Paris"},
        {"prompt": "How does photosynthesis work?", "response": "Plants use sunlight..."},
        {"prompt": "Tell me a joke", "response": "Why don't scientists..."},
    ]

    # Add with quality scores
    for conv in conversations:
        quality = 0.8 + random.random() * 0.2  # Random quality 0.8-1.0
        buffer.add_conversation(conv, quality)

    # Sample replay data
    replay_data = buffer.sample_replay_data(5)
    print(f"Sampled {len(replay_data)} conversations for replay")

    # Get stats
    stats = buffer.get_buffer_stats()
    print(f"Buffer stats: {stats}")

    # Save buffer
    buffer._save_buffer()

if __name__ == "__main__":
    main()