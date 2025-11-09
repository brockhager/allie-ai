import re
from collections import Counter
from typing import List, Any
import logging

logger = logging.getLogger("allie.context_utils")


def enhance_query_with_context(prompt: str, conversation_context: List[Any]) -> str:
    """Simple pronoun resolution / query enhancement.

    If the prompt contains pronouns (they, them, it, he, she, their, etc.),
    scan recent conversation context for likely referent phrases (proper nouns
    or multi-word capitalized entities) and augment the query so memory search
    includes that referent. This is a lightweight heuristic to reduce cases
    where follow-up questions get resolved against the wrong entity.

    Only applies resolution when the question type is compatible with the referent.
    """
    try:
        pronoun_pattern = r"\b(they|them|it|he|she|their|its|those)\b"
        if not re.search(pronoun_pattern, (prompt or "").lower()):
            return prompt

        # Define question types and what referents they make sense for
        question_patterns = {
            'height': [r'\b(how tall|height|high)\b', ['mountains', 'buildings', 'people', 'animals']],
            'distance': [r'\b(how (far|long|wide|big)|distance|miles|kilometers|across|from.*to)\b', ['countries', 'states', 'cities', 'continents', 'rivers', 'roads']],
            'population': [r'\b(how many people|population|inhabitants|live)\b', ['countries', 'cities', 'states']],
            'capital': [r'\b(capital|capital city)\b', ['countries', 'states']],
            'age': [r'\b(how old|age|when|year)\b', ['people', 'buildings', 'organizations', 'events']],
            'location': [r'\b(where|location|located)\b', ['cities', 'countries', 'states', 'mountains']],
        }

        # Determine what type of question this is
        prompt_lower = prompt.lower()
        question_type = None
        compatible_categories = []

        for qtype, (pattern, categories) in question_patterns.items():
            if re.search(pattern, prompt_lower):
                question_type = qtype
                compatible_categories = categories
                break

        # If we can't determine question type, be conservative and don't resolve
        if not question_type:
            return prompt

        candidates = []

        # conversation_context may be a list of message dicts or list of strings
        if isinstance(conversation_context, list):
            # Walk most recent messages first
            for msg in reversed(conversation_context[-12:]):
                text = ""
                if isinstance(msg, dict):
                    text = msg.get("text", "") or ""
                else:
                    text = str(msg)

                # Look for multi-word capitalized sequences first (e.g., 'Rocky Mountains')
                multi = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", text)
                if multi:
                    # Check if this entity type is compatible with the question
                    for entity in multi:
                        if _entity_compatible_with_question(entity, question_type, compatible_categories):
                            candidates.append(entity)

                # Fall back to single capitalized words that look like proper nouns
                single = re.findall(r"\b([A-Z][a-z]{2,})\b", text)
                if single:
                    # filter out pronoun-like capitalized tokens (They, He, She, It)
                    pronoun_excludes = {"they", "them", "it", "he", "she", "their", "those", "its", "we", "us", "i", "you"}
                    filtered = [s for s in single if s.lower() not in pronoun_excludes]
                    for entity in filtered:
                        if _entity_compatible_with_question(entity, question_type, compatible_categories):
                            candidates.append(entity)

        # If we found candidates, pick the most common / recent
        if candidates:
            most = Counter(candidates).most_common(1)[0][0]
            # Augment the prompt rather than attempt brittle substitution
            return f"{prompt} (referring to {most})"

        # Optional fallback: try spaCy + neuralcoref if available to resolve pronouns
        try:
            # neuralcoref is optional and heavy; only use if installed and works
            import spacy
            try:
                import neuralcoref  # type: ignore
            except Exception:
                neuralcoref = None

            if neuralcoref:
                nlp = spacy.load("en_core_web_sm")
                neuralcoref.add_to_pipe(nlp)
                # build a short combined text from recent context and the prompt
                texts = []
                for msg in conversation_context[-12:]:
                    if isinstance(msg, dict):
                        texts.append(msg.get("text", "") or "")
                    else:
                        texts.append(str(msg))
                combined = " ".join(texts + [prompt])
                doc = nlp(combined)
                clusters = getattr(doc._, "coref_clusters", None)
                if clusters:
                    rep = clusters[0].main.text
                    # Also check if the resolved entity is compatible
                    if _entity_compatible_with_question(rep, question_type, compatible_categories):
                        return f"{prompt} (referring to {rep})"
        except Exception:
            # ignore any failures from optional coref path
            pass

    except Exception as e:
        logger.debug(f"enhance_query_with_context failed: {e}")

    return prompt


def _entity_compatible_with_question(entity: str, question_type: str, compatible_categories: List[str]) -> bool:
    """Check if an entity is compatible with a question type."""
    entity_lower = entity.lower()

    # Check against compatible categories
    for category in compatible_categories:
        if category in entity_lower:
            return True

    # Additional heuristics based on entity characteristics
    if question_type == 'distance':
        # Distance questions work with geographical entities, but be specific
        # Countries, states, cities, continents are good; mountains and rivers are not typically measured "across"
        geo_indicators = ['state', 'country', 'city', 'continent', 'america', 'united states', 'usa', 'canada', 'mexico']
        if any(indicator in entity_lower for indicator in geo_indicators):
            return True
        # Exclude entities that don't make sense for "across" questions
        exclude_indicators = ['mountain', 'river', 'lake', 'ocean', 'sea']
        if any(indicator in entity_lower for indicator in exclude_indicators):
            return False

    elif question_type == 'height':
        # Height questions work with mountains, buildings, etc.
        height_indicators = ['mountain', 'peak', 'building', 'tower', 'bridge']
        if any(indicator in entity_lower for indicator in height_indicators):
            return True

    elif question_type == 'population':
        # Population questions work with places that have people
        population_indicators = ['city', 'country', 'state', 'county', 'town']
        if any(indicator in entity_lower for indicator in population_indicators):
            return True

    # Be conservative - if we're not sure, don't resolve
    return False
