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
    """
    try:
        pronoun_pattern = r"\b(they|them|it|he|she|their|its|those)\b"
        if not re.search(pronoun_pattern, (prompt or "").lower()):
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
                    candidates.extend(multi)
                    continue

                # Fall back to single capitalized words that look like proper nouns
                single = re.findall(r"\b([A-Z][a-z]{2,})\b", text)
                if single:
                    # filter out pronoun-like capitalized tokens (They, He, She, It)
                    pronoun_excludes = {"they", "them", "it", "he", "she", "their", "those", "its", "we", "us", "i", "you"}
                    filtered = [s for s in single if s.lower() not in pronoun_excludes]
                    if filtered:
                        candidates.extend(filtered)

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
                    return f"{prompt} (referring to {rep})"
        except Exception:
            # ignore any failures from optional coref path
            pass

    except Exception as e:
        logger.debug(f"enhance_query_with_context failed: {e}")

    return prompt
