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

    def __init__(self, memory_system, hybrid_memory=None):
        self.memory_system = memory_system
        self.hybrid_memory = hybrid_memory  # Optional hybrid memory system

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
                r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)* has (?:about|approximately|around) [\d,]+(?:\.\d+)? million (?:people|residents))",
                r"(Mount [A-Z][a-z]+ is the (?:highest|tallest) mountain (?:in the world|on Earth))",
            ],
            "biography": [
                r"([A-Z][a-z]+ [A-Z][a-z]+ was born in [A-Z][a-z]+)",
                r"([A-Z][a-z]+ [A-Z][a-z]+ (?:developed|created|invented|discovered) (?:the )?[\w\s]+)",
                r"((?:the )?[\w\s]+ was (?:invented|created|developed|discovered) by [A-Z][a-z]+ [A-Z][a-z]+)",
            ],
            "history": [
                r"([\w\s]+ (?:ended|began|started|occurred) in \d{4})",
                r"([\w\s]+ (?:lasted|continued) from \d{4} to \d{4})",
            ],
            "science": [
                r"([\w\s]+ is (?:the process (?:by which|of)|a (?:chemical|physical|biological) process)[\w\s]+)",
                r"(the (?:square root|cube root) of \d+(?:\.\d+)? is (?:approximately |about )?\d+(?:\.\d+)?)",
                r"(\d+(?:\.\d+)? (?:\+|\-|\*|×|÷|/) \d+(?:\.\d+)? (?:equals|is|=) \d+(?:\.\d+)?)",
            ],
            "technology": [
                r"(the [A-Z][\w\s]+ was (?:released|launched|introduced) (?:by )?[A-Z][\w\s]+ in \d{4})",
                r"([A-Z][\w]+ is a (?:programming language|software|operating system|platform)[\w\s]*)",
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
                    self.hybrid_memory.add_fact(fact_text, category=category, confidence=confidence, source="automatic_learning")

                # Generate related information
                related_facts = self._expand_knowledge(fact_text, category)
                for related_fact in related_facts:
                    self.memory_system.add_fact(related_fact, importance=0.7, category=category)
                    if self.hybrid_memory:
                        self.hybrid_memory.add_fact(related_fact, category=category, confidence=0.7, source="automatic_learning_expansion")
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
        """Extract factual information from a message"""
        facts = []

        # Split message into clauses to handle compound sentences
        clauses = re.split(r'\s+(and|but|or)\s+', message)
        
        # Process each clause separately, carrying subject context forward
        current_subject = None
        for i, clause in enumerate(clauses):
            clause = clause.strip()
            if not clause or clause.lower() in ['and', 'but', 'or']:
                continue
            
            # Capitalize first letter if needed
            if clause and not clause[0].isupper():
                clause = clause[0].upper() + clause[1:]
            
            # Try to extract subject from this clause - look for proper nouns, not just capitalized words
            subject_match = re.search(r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)(?:\s+(is|was|has|have|developed|invented|pumps|released|are|were)\b)', clause)
            if subject_match:
                current_subject = subject_match.group(1)
            
            # If this clause doesn't have a subject but we have one from previous clause,
            # and the clause seems to be continuing a fact about the same subject
            if not subject_match and current_subject and not clause.startswith(('The ', 'A ', 'An ')):
                # Prepend the subject to make it a complete clause
                test_clause = f"{current_subject} {clause.lower()}"
                # Check if this makes sense by looking for verb patterns
                if re.search(r'\b(is|was|has|have|developed|invented|pumps|released|are|were)\b', test_clause, re.IGNORECASE):
                    # Prepend subject and fix capitalization
                    clause = f"{current_subject} {clause[0].lower() + clause[1:]}"
            
            # Check each category's patterns against this clause
            for category, patterns in self.fact_patterns.items():
                for pattern in patterns:
                    matches = re.findall(pattern, clause, re.IGNORECASE)
                    for match in matches:
                        if isinstance(match, tuple) and len(match) >= 1:
                            # For patterns with capture groups, the first group should be the complete fact
                            fact_text = match[0].strip()
                        else:
                            # Handle single matches (complete fact)
                            fact_text = match.strip()

                        if fact_text and len(fact_text) > 10:  # Minimum length check
                            confidence = self._calculate_confidence(fact_text, category, clause)
                            facts.append({
                                "fact": fact_text,
                                "category": category,
                                "confidence": confidence,
                                "source": "pattern_match"
                            })

        # Remove duplicates and sort by confidence
        unique_facts = []
        seen = set()
        for fact in sorted(facts, key=lambda x: x["confidence"], reverse=True):
            fact_key = fact["fact"].lower()
            if fact_key not in seen:
                unique_facts.append(fact)
                seen.add(fact_key)

        return unique_facts[:5]  # Limit to top 5 facts per message

    def _format_fact(self, match_groups: Tuple, category: str) -> str:
        """Format extracted match groups into a coherent fact"""
        # Since we're now capturing complete facts, just join the groups
        return " ".join(match_groups).strip()

    def _calculate_confidence(self, fact: str, category: str, original_message: str) -> float:
        """Calculate confidence score for a fact"""
        confidence = 0.5  # Base confidence

        # Length bonus
        if len(fact) > 20:
            confidence += 0.1

        # Category-specific bonuses
        if category == "geography" and any(word in fact.lower() for word in ["city", "state", "country", "capital"]):
            confidence += 0.2

        # Question context penalty (facts in questions are less reliable)
        if "?" in original_message:
            confidence -= 0.1

        # Certainty indicators
        certainty_words = ["is", "are", "was", "were", "has", "have", "located", "born"]
        if any(word in fact.lower() for word in certainty_words):
            confidence += 0.1

        return min(confidence, 1.0)

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