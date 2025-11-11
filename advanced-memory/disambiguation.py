#!/usr/bin/env python3
"""
Disambiguation System for Allie AI

Handles ambiguous topics by detecting multiple meanings and returning
up to 3 interpretations with confidence scores and sources.
"""

import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import re

logger = logging.getLogger(__name__)

class DisambiguationEngine:
    """Engine for detecting and resolving ambiguous topics"""

    def __init__(self, data_path: Optional[str] = None):
        if data_path:
            self.data_file = Path(data_path)
        else:
            self.data_file = Path(__file__).parent.parent.joinpath("data", "disambiguation_cache.json")

        self.cache = {}
        self._load_cache()

        # Source credibility weights (higher = more credible)
        self.source_weights = {
            'wikipedia': 0.95,
            'wikidata': 0.90,
            'dbpedia': 0.85,
            'duckduckgo': 0.75,
            'openlibrary': 0.80,
            'musicbrainz': 0.75,
            'restcountries': 0.85,
            'arxiv': 0.80,
            'pubmed': 0.85,
            'user': 0.60,
            'correction': 0.70,
            'automatic_learning': 0.65,
            'bulk_learn': 0.55
        }

    def _load_cache(self):
        """Load disambiguation cache from file"""
        if self.data_file.exists():
            try:
                with open(self.data_file, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
            except Exception:
                self.cache = {}
        else:
            self.cache = {}

    def _save_cache(self):
        """Save disambiguation cache to file"""
        try:
            self.data_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.data_file, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to save disambiguation cache: {e}")

    def detect_ambiguity(self, query: str, search_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Detect if a query has multiple meanings based on search results

        Returns:
        {
            "is_ambiguous": bool,
            "interpretations": List[Dict],  # Up to 3 interpretations
            "confidence": float,  # Overall confidence in disambiguation
            "needs_clarification": bool
        }
        """
        query_lower = query.lower().strip()

        # Check cache first
        cache_key = query_lower
        if cache_key in self.cache:
            cached = self.cache[cache_key]
            # Check if cache is still valid (24 hours)
            if datetime.now().isoformat() < cached.get("expires", ""):
                return cached["result"]

        # Analyze search results for different meanings
        interpretations = self._analyze_search_results(query, search_results)

        # Determine if ambiguous
        is_ambiguous = len(interpretations) > 1

        # Calculate overall confidence
        if is_ambiguous:
            # Average confidence across interpretations, weighted by their individual confidence
            total_weighted_confidence = sum(interpretation["confidence_score"] * interpretation["confidence_score"] for interpretation in interpretations)
            total_weight = sum(interpretation["confidence_score"] for interpretation in interpretations)
            overall_confidence = total_weighted_confidence / total_weight if total_weight > 0 else 0.5
        else:
            overall_confidence = interpretations[0]["confidence_score"] if interpretations else 0.5

        # Determine if clarification is needed
        needs_clarification = (
            is_ambiguous and
            (overall_confidence <= 0.4 or self._interpretations_tied(interpretations))
        )

        result = {
            "is_ambiguous": is_ambiguous,
            "interpretations": interpretations[:3],  # Limit to 3
            "confidence": overall_confidence,
            "needs_clarification": needs_clarification
        }

        # Cache result for 24 hours
        self.cache[cache_key] = {
            "result": result,
            "expires": (datetime.now() + timedelta(hours=24)).isoformat()
        }
        self._save_cache()

        return result

    def _analyze_search_results(self, query: str, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyze search results to extract different interpretations

        Each interpretation should have:
        - meaning_label: str (e.g., "Company", "Person", "Concept")
        - summary: str (brief description)
        - sources_consulted: List[str]
        - confidence_score: float (0-1)
        - status: str ("true", "not_verified", "false")
        """
        interpretations = []

        # Group results by semantic meaning
        meaning_groups = self._group_by_meaning(query, search_results)

        for meaning_label, results in meaning_groups.items():
            if not results:
                continue

            # Extract summary from highest-confidence result
            best_result = max(results, key=lambda r: r.get("confidence", 0.5))
            summary = self._extract_summary(best_result, query)

            # Calculate confidence score
            confidence_score = self._calculate_interpretation_confidence(results, meaning_label)

            # Determine status
            status = self._determine_interpretation_status(results)

            # Collect sources
            sources_consulted = list(set(r.get("source", "unknown") for r in results))

            interpretation = {
                "meaning_label": meaning_label,
                "summary": summary,
                "sources_consulted": sources_consulted,
                "confidence_score": confidence_score,
                "status": status
            }

            interpretations.append(interpretation)

        # Sort by confidence
        interpretations.sort(key=lambda x: x["confidence_score"], reverse=True)

        return interpretations

    def _group_by_meaning(self, query: str, search_results: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group search results by semantic meaning, allowing credible sources to corroborate"""
        groups = {}

        # First pass: group by strict semantic meaning
        for result in search_results:
            text = result.get("fact", "").lower()
            source = result.get("source", "unknown")

            # Simple keyword-based categorization
            if any(word in text for word in ["company", "corporation", "inc", "ltd", "business", "industry"]):
                meaning = "Company/Organization"
            elif any(word in text for word in ["born", "person", "author", "actor", "scientist", "president", "artist"]):
                meaning = "Person"
            elif any(word in text for word in ["country", "city", "location", "geography", "capital", "population"]):
                meaning = "Place/Geography"
            elif any(word in text for word in ["concept", "theory", "principle", "idea", "definition"]):
                meaning = "Concept/Idea"
            elif any(word in text for word in ["book", "novel", "author", "literature", "isbn"]):
                meaning = "Book/Literature"
            elif any(word in text for word in ["music", "album", "song", "artist", "band", "composer"]):
                meaning = "Music/Artist"
            elif any(word in text for word in ["science", "research", "study", "paper", "theory", "genus", "species", "biology", "animal", "snake", "snakes", "venomous", "non-venomous"]):
                meaning = "Science/Research"
            elif any(word in text for word in ["politics", "government", "election", "policy", "political"]):
                meaning = "Politics/Government"
            else:
                # Try to infer from source
                if source == "wikipedia":
                    meaning = "General Knowledge"
                elif source == "wikidata":
                    meaning = "Structured Data"
                else:
                    meaning = "General"

            if meaning not in groups:
                groups[meaning] = []
            groups[meaning].append(result)

        # Second pass: merge groups that represent the same topic based on content similarity
        merged_groups = {}
        processed_groups = set()

        # For each group, check if it should be merged with other groups
        for meaning, results in groups.items():
            if meaning in processed_groups:
                continue

            # Start with this group
            merged_results = results.copy()
            processed_groups.add(meaning)

            # Check other groups for merging
            for other_meaning, other_results in groups.items():
                if other_meaning in processed_groups or other_meaning == meaning:
                    continue

                # Merge if they have overlapping key terms and credible sources
                if self._groups_should_merge(results, other_results):
                    merged_results.extend(other_results)
                    processed_groups.add(other_meaning)

            merged_groups[meaning] = merged_results

        return merged_groups

    def _groups_should_merge(self, group1: List[Dict[str, Any]], group2: List[Dict[str, Any]]) -> bool:
        """Determine if two groups should be merged based on content and source credibility"""
        # Check if both groups have credible sources
        credible1 = any(self.source_weights.get(r.get("source", "unknown"), 0) >= 0.8 for r in group1)
        credible2 = any(self.source_weights.get(r.get("source", "unknown"), 0) >= 0.8 for r in group2)

        if not (credible1 and credible2):
            return False

        # For merging, require that the facts are actually about the same specific topic
        # This is more restrictive than just overlapping keywords
        text1 = " ".join(r.get("fact", "").lower() for r in group1)
        text2 = " ".join(r.get("fact", "").lower() for r in group2)

        # Extract key terms (words longer than 3 chars, excluding only the most generic words)
        common_words = {"python", "used", "widely", "data", "software", "development"}
        words1 = set(w for w in text1.split() if len(w) > 3 and w not in common_words)
        words2 = set(w for w in text2.split() if len(w) > 3 and w not in common_words)

        # Require significant overlap (programming language facts should merge)
        overlap = words1.intersection(words2)
        return len(overlap) >= 1  # At least 1 specific overlapping term

    def _extract_summary(self, result: Dict[str, Any], query: str) -> str:
        """Extract a concise summary from a search result"""
        text = result.get("fact", "")

        # Remove the query from the beginning if present
        query_lower = query.lower()
        if text.lower().startswith(query_lower):
            text = text[len(query_lower):].strip(" -,.")

        # Limit to first sentence or reasonable length
        sentences = re.split(r'[.!?]', text)
        summary = sentences[0].strip()

        if len(summary) > 200:
            summary = summary[:197] + "..."

        return summary or text[:200] + "..."

    def _calculate_interpretation_confidence(self, results: List[Dict[str, Any]], meaning_label: str) -> float:
        """
        Calculate confidence score for an interpretation using the new formula:
        base_source_weight + agreement_bonus + recency_bonus − ambiguity_penalty − provenance_penalty
        """
        if not results:
            return 0.0

        # Base source weight (average of source weights)
        source_weights = [self.source_weights.get(r.get("source", "unknown"), 0.5) for r in results]
        base_weight = sum(source_weights) / len(source_weights)

        # Agreement bonus: +0.2 if 2+ sources agree, +0.1 if 3+ sources
        unique_sources = len(set(r.get("source", "unknown") for r in results))
        agreement_bonus = 0.0
        if unique_sources >= 3:
            agreement_bonus = 0.2
        elif unique_sources >= 2:
            agreement_bonus = 0.1

        # Recency bonus: +0.1 if result is recent (within 30 days)
        recency_bonus = 0.0
        now = datetime.now()
        for result in results:
            timestamp = result.get("timestamp", "")
            if timestamp:
                try:
                    result_date = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    if (now - result_date).days <= 30:
                        recency_bonus = 0.1
                        break
                except:
                    pass

        # Ambiguity penalty: -0.1 if this is one of multiple interpretations
        ambiguity_penalty = 0.0  # This will be calculated at a higher level

        # Provenance penalty: -0.5 if sources contain false facts, -0.1 for other issues
        provenance_penalty = 0.0
        if any(r.get("status") == "false" for r in results):
            provenance_penalty = 0.5  # Strong penalty for false facts
        elif any(r.get("status") == "not_verified" for r in results):
            provenance_penalty = 0.1  # Mild penalty for unverified facts

        confidence = base_weight + agreement_bonus + recency_bonus - ambiguity_penalty - provenance_penalty

        # Cap at 50% unless 2+ independent credible sources agree
        credible_sources = [r for r in results if self.source_weights.get(r.get("source", "unknown"), 0) >= 0.8]
        if len(credible_sources) < 2 and confidence > 0.5:
            confidence = 0.5

        return max(0.0, min(1.0, confidence))

    def _determine_interpretation_status(self, results: List[Dict[str, Any]]) -> str:
        """Determine the status of an interpretation based on Fact-Check integration"""
        # Priority: true > not_verified > false
        statuses = [r.get("status", "not_verified") for r in results]

        if "true" in statuses:
            return "true"
        elif "not_verified" in statuses:
            return "not_verified"
        elif "false" in statuses:
            return "false"
        else:
            return "not_verified"

    def _interpretations_tied(self, interpretations: List[Dict[str, Any]]) -> bool:
        """Check if top interpretations have very similar confidence scores"""
        if len(interpretations) < 2:
            return False

        top_two = sorted(interpretations, key=lambda x: x["confidence_score"], reverse=True)[:2]
        diff = abs(top_two[0]["confidence_score"] - top_two[1]["confidence_score"])
        return diff < 0.1  # Less than 10% difference

    def generate_clarifying_question(self, interpretations: List[Dict[str, Any]]) -> str:
        """Generate a clarifying question when confidence is low or interpretations are tied"""
        if not interpretations:
            return "Could you provide more context about what you're asking about?"

        # Create question based on the different meanings
        meanings = [interp["meaning_label"] for interp in interpretations[:3]]

        if len(meanings) == 2:
            return f"Are you asking about {meanings[0]} or {meanings[1]}?"
        elif len(meanings) == 3:
            return f"Are you asking about {meanings[0]}, {meanings[1]}, or {meanings[2]}?"
        else:
            return f"I found multiple meanings for your query. Could you specify which one you mean?"

    def log_disambiguation_event(self, query: str, result: Dict[str, Any], user_choice: Optional[str] = None):
        """Log disambiguation events for analysis and improvement"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "is_ambiguous": result["is_ambiguous"],
            "num_interpretations": len(result["interpretations"]),
            "confidence": result["confidence"],
            "needs_clarification": result["needs_clarification"],
            "user_choice": user_choice,
            "interpretations": [
                {
                    "meaning": interp["meaning_label"],
                    "confidence": interp["confidence_score"],
                    "status": interp["status"],
                    "sources": interp["sources_consulted"]
                }
                for interp in result["interpretations"]
            ]
        }

        # In a real implementation, this would be stored in a database
        # For now, we'll log it
        logger.info(f"Disambiguation event: {json.dumps(log_entry, default=str)}")

        # TODO: Store in learning_log table when database integration is complete