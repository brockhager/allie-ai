#!/usr/bin/env python3
"""
Safety and Quality Controls for Allie Learning System

This module provides comprehensive filtering and validation for training data
to ensure safe, high-quality learning episodes.
"""

import re
import json
from typing import Dict, List, Any, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class SafetyFilter:
    """Multi-layer safety filtering for training data"""

    def __init__(self):
        # Harmful content patterns
        self.harmful_patterns = [
            r'\b(kill|murder|suicide|self-harm)\b',
            r'\b(hate|racist|sexist|homophobic)\b',
            r'\b(drug|illegal|criminal)\b',
            r'\b(violence|abuse|assault)\b',
            r'\b(exploit|hack|phish)\b'
        ]

        # PII patterns (simplified)
        self.pii_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b\d{16}\b',  # Credit card
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'\b\d{10}\b'  # Phone
        ]

    def check_conversation(self, conversation: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Check if conversation passes safety filters

        Returns:
            (is_safe: bool, reason: str)
        """
        text = f"{conversation.get('prompt', '')} {conversation.get('response', '')}"

        # Check for harmful content
        for pattern in self.harmful_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return False, f"Harmful content detected: {pattern}"

        # Check for PII
        for pattern in self.pii_patterns:
            if re.search(pattern, text):
                return False, f"PII detected: {pattern}"

        # Check for excessive caps (potential shouting/abuse)
        if len(re.findall(r'[A-Z]{3,}', text)) > len(text) * 0.1:
            return False, "Excessive caps detected"

        return True, "Safe"

class QualityFilter:
    """Quality assessment for training data"""

    def __init__(self):
        self.min_length = 10  # Minimum characters
        self.max_length = 2000  # Maximum characters
        self.min_words = 3  # Minimum words in response

    def assess_quality(self, conversation: Dict[str, Any]) -> Dict[str, float]:
        """
        Assess conversation quality with multiple metrics

        Returns dict with quality scores:
        - coherence: 0-1 (sentence structure, grammar)
        - relevance: 0-1 (response relevance to prompt)
        - informativeness: 0-1 (information content)
        - engagement: 0-1 (conversation flow)
        - overall: 0-1 (weighted average)
        """
        prompt = conversation.get('prompt', '')
        response = conversation.get('response', '')

        scores = {
            'coherence': self._check_coherence(response),
            'relevance': self._check_relevance(prompt, response),
            'informativeness': self._check_informativeness(response),
            'engagement': self._check_engagement(prompt, response),
        }

        # Weighted overall score
        weights = {'coherence': 0.3, 'relevance': 0.3, 'informativeness': 0.25, 'engagement': 0.15}
        scores['overall'] = sum(scores[k] * weights[k] for k in scores.keys())

        return scores

    def _check_coherence(self, text: str) -> float:
        """Check text coherence and grammar"""
        if len(text) < self.min_length:
            return 0.0

        # Basic heuristics
        score = 1.0

        # Check for complete sentences
        sentences = re.split(r'[.!?]+', text)
        complete_sentences = sum(1 for s in sentences if len(s.strip()) > 5)
        if complete_sentences == 0:
            return 0.0

        # Penalize excessive punctuation
        punctuation_ratio = len(re.findall(r'[.!?]', text)) / len(text)
        if punctuation_ratio > 0.1:
            score *= 0.8

        # Penalize repetitive words
        words = text.lower().split()
        if len(words) > 0:
            unique_ratio = len(set(words)) / len(words)
            score *= min(1.0, unique_ratio * 2)  # Boost uniqueness

        return min(1.0, score)

    def _check_relevance(self, prompt: str, response: str) -> float:
        """Check if response is relevant to prompt"""
        prompt_words = set(prompt.lower().split())
        response_words = set(response.lower().split())

        if len(prompt_words) == 0:
            return 0.5  # Neutral if no prompt words

        # Simple word overlap
        overlap = len(prompt_words.intersection(response_words))
        overlap_ratio = overlap / len(prompt_words)

        # Boost for question-answer patterns
        if '?' in prompt and len(response.strip()) > 0:
            overlap_ratio *= 1.2

        return min(1.0, overlap_ratio * 2)

    def _check_informativeness(self, text: str) -> float:
        """Check information content"""
        words = text.split()
        if len(words) < self.min_words:
            return 0.0

        # Favor longer, more detailed responses
        length_score = min(1.0, len(words) / 20)  # Boost up to 20 words

        # Check for question marks (indicates information seeking)
        question_ratio = text.count('?') / max(1, len(text) / 100)
        info_score = min(1.0, question_ratio)

        # Check for numbers/dates (factual content)
        fact_indicators = len(re.findall(r'\d+', text)) + text.count('/') + text.count('-')
        fact_score = min(1.0, fact_indicators / 5)

        return (length_score + info_score + fact_score) / 3

    def _check_engagement(self, prompt: str, response: str) -> float:
        """Check conversation engagement"""
        # Favor back-and-forth conversations
        if len(prompt.strip()) > 0 and len(response.strip()) > 0:
            base_score = 0.8
        else:
            base_score = 0.4

        # Boost for follow-up questions
        if '?' in response:
            base_score *= 1.2

        # Boost for personalized responses
        personal_words = ['you', 'your', 'I', 'me', 'we', 'us']
        personal_count = sum(1 for word in personal_words if word in response.lower())
        personal_score = min(1.0, personal_count / 3)

        return min(1.0, (base_score + personal_score) / 2)

class BiasDetector:
    """Detect and mitigate bias in training data"""

    def __init__(self):
        # Simplified bias patterns (would be more sophisticated in production)
        self.bias_patterns = {
            'gender': [r'\b(he|him|his)\b.*\b(she|her|hers)\b', r'\b(man|men)\b.*\b(woman|women)\b'],
            'race': [r'\b(white|black|asian|hispanic)\b'],
            'age': [r'\b(old|young|elderly)\b'],
            'religion': [r'\b(christian|muslim|jewish|hindu|buddhist|atheist)\b']
        }

    def detect_bias(self, conversation: Dict[str, Any]) -> Dict[str, float]:
        """Detect potential bias in conversation"""
        text = f"{conversation.get('prompt', '')} {conversation.get('response', '')}"

        bias_scores = {}
        for bias_type, patterns in self.bias_patterns.items():
            matches = 0
            for pattern in patterns:
                matches += len(re.findall(pattern, text, re.IGNORECASE))
            bias_scores[bias_type] = min(1.0, matches / 10)  # Normalize

        # Overall bias score
        bias_scores['overall'] = sum(bias_scores.values()) / len(bias_scores)

        return bias_scores

class LearningDataValidator:
    """Main validation orchestrator"""

    def __init__(self):
        self.safety_filter = SafetyFilter()
        self.quality_filter = QualityFilter()
        self.bias_detector = BiasDetector()

        # Quality thresholds
        self.min_quality_score = 0.6
        self.max_bias_score = 0.3

    def validate_conversation(self, conversation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive validation of a conversation for training

        Returns validation result with scores and decision
        """
        result = {
            'conversation': conversation,
            'approved': False,
            'reason': '',
            'scores': {}
        }

        # Safety check first
        is_safe, safety_reason = self.safety_filter.check_conversation(conversation)
        if not is_safe:
            result['reason'] = f"Safety violation: {safety_reason}"
            return result

        # Quality assessment
        quality_scores = self.quality_filter.assess_quality(conversation)
        result['scores']['quality'] = quality_scores

        if quality_scores['overall'] < self.min_quality_score:
            result['reason'] = f"Quality too low: {quality_scores['overall']:.2f}"
            return result

        # Bias detection
        bias_scores = self.bias_detector.detect_bias(conversation)
        result['scores']['bias'] = bias_scores

        if bias_scores['overall'] > self.max_bias_score:
            result['reason'] = f"Bias detected: {bias_scores['overall']:.2f}"
            return result

        # All checks passed
        result['approved'] = True
        result['reason'] = "Approved for training"

        return result

    def validate_dataset(self, conversations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate an entire dataset"""
        results = []
        approved_count = 0

        for conv in conversations:
            validation = self.validate_conversation(conv)
            results.append(validation)
            if validation['approved']:
                approved_count += 1

        summary = {
            'total_conversations': len(conversations),
            'approved_count': approved_count,
            'approval_rate': approved_count / len(conversations) if conversations else 0,
            'results': results,
            'quality_stats': self._compute_stats([r for r in results if r['approved']], 'quality'),
            'bias_stats': self._compute_stats([r for r in results if r['approved']], 'bias')
        }

        return summary

    def _compute_stats(self, results: List[Dict], score_type: str) -> Dict[str, float]:
        """Compute statistics for approved conversations"""
        if not results:
            return {'mean': 0, 'min': 0, 'max': 0}

        scores = [r['scores'][score_type]['overall'] for r in results]

        return {
            'mean': sum(scores) / len(scores),
            'min': min(scores),
            'max': max(scores)
        }

def main():
    """Example usage"""
    validator = LearningDataValidator()

    # Example conversations
    test_conversations = [
        {
            "prompt": "What is the capital of France?",
            "response": "The capital of France is Paris."
        },
        {
            "prompt": "How do I hack a website?",
            "response": "I'm sorry, but I can't help with illegal activities."
        },
        {
            "prompt": "Hi",
            "response": "Hello! How can I help you today?"
        }
    ]

    results = validator.validate_dataset(test_conversations)

    print("Validation Summary:")
    print(f"Total: {results['total_conversations']}")
    print(f"Approved: {results['approved_count']} ({results['approval_rate']:.1%})")

    for i, result in enumerate(results['results']):
        status = "✅" if result['approved'] else "❌"
        print(f"{status} Conversation {i+1}: {result['reason']}")

if __name__ == "__main__":
    main()