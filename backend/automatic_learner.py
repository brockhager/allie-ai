#!/usr/bin/env python3
"""
Automatic Learning System for Allie

Extracts, categorizes, and stores factual information from conversations.
Provides continuous learning capabilities without explicit commands.
"""

import re
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class AutomaticLearner:
    """Handles automatic extraction and storage of factual information"""

    def __init__(self, memory_system, hybrid_memory=None, learning_queue=None):
        self.memory_system = memory_system
        self.hybrid_memory = hybrid_memory  # Optional hybrid memory system
        self.learning_queue = learning_queue  # Optional learning queue system

        # Knowledge expansion rules
        self.expansion_rules = {
            "geography": self._expand_geography,
            "biography": self._expand_biography,
            "history": self._expand_history,
            "science": self._expand_science,
            "technology": self._expand_technology
        }

        # Fact patterns for different categories - capture complete fact sentences
        self.fact_patterns = {
            "geography": [
                r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)* is the capital (?:of|city of) [A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
                r"((?:the )?capital (?:of|city of) [A-Z][a-z]+(?:\s+[A-Z][a-z]+)* is [A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
                r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)* is (?:in|located in) [A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
                r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)* is the (?:largest|biggest|smallest) city in [A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
                r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)* has (?:about|approximately|around) [\d,]+(?:\.\d+)? million (?:people|residents|inhabitants))",
                r"(Mount [A-Z][a-z]+ is the (?:highest|tallest) mountain (?:in the world|on Earth))",
                r"([A-Z][a-z]+ is (?:north|south|east|west) of [A-Z][a-z]+)",
                r"(The [A-Z][a-z]+ River (?:flows through|runs through) [A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            ],
            "biography": [
                r"([A-Z][a-z]+ [A-Z][a-z]+ was born (?:on )?(?:\d{1,2} \w+ \d{4}|in \d{4}) in [A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
                r"([A-Z][a-z]+ [A-Z][a-z]+ (?:developed|created|invented|discovered) (?:the )?[\w\s]+ in \d{4})",
                r"((?:the )?[\w\s]+ was (?:invented|created|developed|discovered) by [A-Z][a-z]+ [A-Z][a-z]+ in \d{4})",
                r"([A-Z][a-z]+ [A-Z][a-z]+ served as (?:president|prime minister|governor|mayor) (?:of [A-Z][a-z]+(?:\s+[A-Z][a-z]+)* )?from \d{4} to \d{4})",
                r"([A-Z][a-z]+ [A-Z][a-z]+ died (?:on )?(?:\d{1,2} \w+ \d{4}|in \d{4}) in [A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            ],
            "history": [
                r"([\w\s]+ (?:ended|began|started|occurred) (?:on )?(?:\d{1,2} \w+ \d{4}|in \d{4}))",
                r"([\w\s]+ (?:lasted|continued) from \d{4} to \d{4})",
                r"(World War (?:I|II|One|Two) (?:ended|began|started) in \d{4})",
                r"(The [A-Z][a-z]+ Revolution (?:occurred|began|ended) in \d{4})",
            ],
            "science": [
                r"([\w\s]+ is (?:the process (?:by which|of)|a (?:chemical|physical|biological) process)[\w\s]+)",
                r"(the (?:square root|cube root) of \d+(?:\.\d+)? is (?:approximately |about )?\d+(?:\.\d+)?)",
                r"(\d+(?:\.\d+)? (?:\+|\-|\*|×|÷|/) \d+(?:\.\d+)? (?:equals|is|=) \d+(?:\.\d+)?)",
                r"(Water (?:boils|freezes) at \d+(?:\.\d+)? degrees (?:Celsius|Fahrenheit))",
                r"(The speed of light is \d+(?:\.\d+)? (?:million |billion |trillion )?(?:meters|kilometers|miles) per second)",
                r"(DNA stands for deoxyribonucleic acid)",
            ],
            "technology": [
                r"(the [A-Z][\w\s]+ was (?:released|launched|introduced) (?:by )?[A-Z][\w\s]+ in \d{4})",
                r"([A-Z][\w]+ is a (?:programming language|software|operating system|platform)[\w\s]*)",
                r"(Python was created by Guido van Rossum in \d{4})",
                r"(JavaScript was (?:created|developed) by Brendan Eich (?:in|during) \d{4})",
                r"(Linux was created by Linus Torvalds in \d{4})",
            ]
        }

        # Quality filters to reject low-quality facts
        self.quality_filters = {
            "min_length": 10,  # Minimum fact length
            "max_length": 500,  # Maximum fact length
            "reject_patterns": [
                r'\b(i|we|you|they) (don\'t|do not) know\b',
                r'\b(not sure|uncertain|maybe|perhaps|possibly)\b',
                r'\b(i think|i believe|i feel)\b',
                r'\b(according to|based on|from what i)\b',
                r'\b(let me|can you|would you)\b',
                r'\b(example|for instance|such as)\b',
                r'\b(many|some|few|several)\b',
                r'\b(approximately|about|around|roughly)\b',
            ],
            "require_verbs": [
                'is', 'was', 'are', 'were', 'has', 'have', 'had', 'developed', 'created',
                'invented', 'discovered', 'served', 'died', 'born', 'located', 'flows',
                'boils', 'freezes', 'stands'
            ]
        }

    def process_message(self, message: str, speaker: str = "user") -> Dict[str, Any]:
        """
        Process a message for factual information extraction and learning

        Args:
            message: The message text to analyze
            speaker: Who said the message ("user" or "assistant")

        Returns:
            Dict containing extracted facts, categories, and learning actions
        """
        extracted_facts = []
        learning_actions = []

        # Extract facts from the message
        facts = self._extract_facts(message)
        extracted_facts.extend(facts)

        # Process each fact
        for fact_data in facts:
            fact_text = fact_data["fact"]
            category = fact_data["category"]
            confidence = fact_data["confidence"]

            # Only store high-confidence facts
            if confidence >= 0.6:
                # Store the fact in legacy memory
                self.memory_system.add_fact(fact_text, importance=confidence, category=category)
                
                # Also store in hybrid memory if available
                if self.hybrid_memory:
                    self.hybrid_memory.add_fact(
                        fact=fact_text,
                        category=category,
                        confidence=confidence,
                        source="automatic_learning"
                    )

                # Add to learning queue for reconciliation if available and confidence is high enough
                if self.learning_queue and confidence >= 0.7:
                    try:
                        # Extract keyword from fact (first few significant words)
                        words = fact_text.split()[:3]  # Use first 3 words as keyword
                        keyword = ' '.join(words).strip('.,!?;:')
                        
                        queue_result = self.learning_queue.add_to_learning_queue(
                            keyword=keyword,
                            fact=fact_text,
                            source="conversation_learning",
                            confidence=confidence,
                            category=category
                        )
                        
                        if queue_result.get("status") == "queued":
                            learning_actions.append({
                                "action": "queued_for_reconciliation",
                                "fact": fact_text,
                                "category": category,
                                "confidence": confidence,
                                "queue_id": queue_result.get("queue_id")
                            })
                    except Exception as e:
                        logger.warning(f"Failed to add fact to learning queue: {e}")

                # Generate related information
                related_facts = self._expand_knowledge(fact_text, category)
                for related_fact in related_facts:
                    self.memory_system.add_fact(related_fact, importance=0.7, category=category)
                    if self.hybrid_memory:
                        self.hybrid_memory.add_fact(
                            fact=related_fact,
                            category=category,
                            confidence=0.7,
                            source="automatic_learning_expansion"
                        )
                    extracted_facts.append({
                        "fact": related_fact,
                        "category": category,
                        "confidence": 0.7,
                        "type": "expanded"
                    })

                learning_actions.append({
                    "action": "stored_fact",
                    "fact": fact_text,
                    "category": category,
                    "confidence": confidence
                })

        return {
            "extracted_facts": extracted_facts,
            "learning_actions": learning_actions,
            "total_facts_learned": len([f for f in extracted_facts if f.get("type") != "expanded"])
        }

    def _extract_facts(self, message: str) -> List[Dict[str, Any]]:
        """Extract factual information from a message with improved quality control"""
        facts = []

        # Skip messages that are clearly not factual
        message_lower = message.lower()
        if any(phrase in message_lower for phrase in [
            "i think", "i believe", "i feel", "in my opinion", "i'm not sure",
            "maybe", "perhaps", "possibly", "i don't know", "not sure",
            "let me check", "i'll look it up", "can you tell me"
        ]):
            return facts

        # Split message into sentences for better processing
        sentences = re.split(r'[.!?]+', message)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 5]

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Apply quality filters first
            if not self._passes_quality_filters(sentence):
                continue

            # Try to match against fact patterns
            for category, patterns in self.fact_patterns.items():
                for pattern in patterns:
                    matches = re.findall(pattern, sentence, re.IGNORECASE)
                    for match in matches:
                        fact_text = match.strip()
                        if fact_text and self._passes_quality_filters(fact_text):
                            # Calculate confidence based on pattern specificity and fact completeness
                            confidence = self._calculate_fact_confidence(fact_text, category, pattern)

                            if confidence >= 0.7:  # Only accept high-confidence facts
                                facts.append({
                                    "fact": fact_text,
                                    "category": category,
                                    "confidence": confidence,
                                    "source_sentence": sentence
                                })
                                break  # Take the first good match for this sentence

        return facts

    def _passes_quality_filters(self, text: str) -> bool:
        """Check if text passes quality filters"""
        # Length checks
        if len(text) < self.quality_filters["min_length"] or len(text) > self.quality_filters["max_length"]:
            return False

        # Reject patterns
        text_lower = text.lower()
        for pattern in self.quality_filters["reject_patterns"]:
            if re.search(pattern, text_lower):
                return False

        # Must contain at least one factual verb
        has_factual_verb = any(verb in text_lower for verb in self.quality_filters["require_verbs"])
        if not has_factual_verb:
            return False

        # Must have some structure (subject-verb-object like)
        words = text.split()
        if len(words) < 4:  # Too short to be a proper fact
            return False

        # Check for proper nouns (indicating specific entities)
        proper_nouns = [w for w in words if w[0].isupper() and w[0].isalpha()]
        if len(proper_nouns) < 1:  # Facts should reference specific things
            return False

        return True

    def _calculate_fact_confidence(self, fact: str, category: str, pattern: str) -> float:
        """Calculate confidence score for a fact"""
        confidence = 0.5  # Base confidence

        # Pattern specificity bonus
        if 'born' in pattern or 'died' in pattern:
            confidence += 0.2  # Biographical facts with dates are reliable
        elif 'capital' in pattern or 'located' in pattern:
            confidence += 0.15  # Geographic facts are usually stable
        elif any(word in pattern for word in ['invented', 'created', 'discovered']):
            confidence += 0.1  # Attribution facts are important

        # Fact completeness bonus
        words = fact.split()
        if len(words) > 6:
            confidence += 0.1  # Longer facts tend to be more complete

        # Contains numbers/dates (more specific)
        if re.search(r'\d{4}', fact):  # Year
            confidence += 0.1
        if re.search(r'\d{1,2} \w+ \d{4}', fact):  # Full date
            confidence += 0.1

        # Multiple proper nouns (more specific)
        proper_nouns = [w for w in words if w[0].isupper() and w[0].isalpha()]
        if len(proper_nouns) >= 2:
            confidence += 0.1

        # Category-specific adjustments
        if category == "science":
            confidence += 0.05  # Science facts are generally reliable
        elif category == "history":
            confidence += 0.05  # Historical facts are generally reliable

        return min(confidence, 1.0)  # Cap at 1.0

    def _expand_knowledge(self, fact: str, category: str) -> List[str]:
        """Expand knowledge by adding related facts"""
        if category in self.expansion_rules:
            return self.expansion_rules[category](fact)
        return []

    def _expand_geography(self, fact: str) -> List[str]:
        """Expand geographical knowledge"""
        expansions = []
        fact_lower = fact.lower()

        # City-state relationships
        city_state_map = {
            "phoenix": "Arizona",
            "tucson": "Arizona",
            "mesa": "Arizona",
            "los angeles": "California",
            "san diego": "California",
            "san francisco": "California",
            "denver": "Colorado",
            "colorado springs": "Colorado"
        }

        for city, state in city_state_map.items():
            if city in fact_lower and state not in fact_lower:
                expansions.append(f"{city.title()} is located in {state}")

        # State-country relationships
        if "arizona" in fact_lower and "united states" not in fact_lower:
            expansions.append("Arizona is a state in the United States")
        if "california" in fact_lower and "united states" not in fact_lower:
            expansions.append("California is a state in the United States")
        if "colorado" in fact_lower and "united states" not in fact_lower:
            expansions.append("Colorado is a state in the United States")

        return expansions

    def _expand_biography(self, fact: str) -> List[str]:
        """Expand biographical knowledge"""
        expansions = []
        fact_lower = fact.lower()

        # Add context about professions, time periods, etc.
        if "born" in fact_lower:
            # Could add birth year ranges, famous contemporaries, etc.
            pass

        return expansions

    def _expand_history(self, fact: str) -> List[str]:
        """Expand historical knowledge"""
        expansions = []
        fact_lower = fact.lower()

        # Add context about time periods, related events, etc.
        if "world war" in fact_lower:
            expansions.append("World War II lasted from 1939 to 1945")

        return expansions

    def _expand_science(self, fact: str) -> List[str]:
        """Expand scientific knowledge"""
        expansions = []
        fact_lower = fact.lower()

        # Add related scientific principles, discoveries, etc.
        if "oxygen" in fact_lower:
            expansions.append("Oxygen is essential for human respiration")

        return expansions

    def _expand_technology(self, fact: str) -> List[str]:
        """Expand technological knowledge"""
        expansions = []
        fact_lower = fact.lower()

        # Add related technologies, companies, etc.
        if "python" in fact_lower:
            expansions.append("Python is a high-level programming language")

        return expansions

    def generate_learning_response(self, learning_actions: List[Dict]) -> str:
        """Generate a response confirming what was learned"""
        if not learning_actions:
            return ""

        responses = []
        for action in learning_actions:
            if action["action"] == "stored_fact":
                fact = action["fact"]
                category = action["category"]
                confidence = action["confidence"]

                if confidence >= 0.8:
                    responses.append(f"✓ Learned: {fact} (stored in {category})")
                elif confidence >= 0.6:
                    responses.append(f"📝 Noted: {fact} (stored in {category})")

        if responses:
            return "\n\n" + "\n".join(responses)
        return ""

    def remove_fact(self, fact_text: str) -> bool:
        """Remove a fact from memory by exact text match"""
        return self.memory_system.remove_fact(fact_text)