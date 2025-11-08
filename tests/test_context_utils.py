import pytest

from backend.context_utils import enhance_query_with_context


def test_no_pronoun_returns_same():
    prompt = "How tall is Mount Everest?"
    ctx = [
        {"text": "We talked about mountains earlier."},
    ]
    out = enhance_query_with_context(prompt, ctx)
    assert out == prompt


def test_pronoun_with_recent_multiword_entity():
    prompt = "how tall are they"
    ctx = [
        {"text": "I was reading about the Rocky Mountains recently."},
        {"text": "They are huge and span many states."},
    ]
    out = enhance_query_with_context(prompt, ctx)
    assert "Rocky Mountains" in out
    assert out.startswith(prompt)


def test_pronoun_prefers_most_common_candidate():
    prompt = "how tall are they"
    ctx = [
        {"text": "Rocky Mountains are amazing."},
        {"text": "Rocky Mountains have many peaks."},
        {"text": "Eiffel Tower was mentioned earlier."},
    ]
    out = enhance_query_with_context(prompt, ctx)
    # most common should be Rocky Mountains
    assert "Rocky Mountains" in out


def test_lowercase_referent_no_match():
    # If referent is lowercase (no capitalization), heuristic shouldn't invent a referent
    prompt = "how tall are they"
    ctx = [
        {"text": "we talked about the rocky mountains earlier"},
    ]
    out = enhance_query_with_context(prompt, ctx)
    assert out == prompt
