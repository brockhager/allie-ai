"""
External Knowledge Sources

Modular retrieval system for multiple knowledge bases:
- DuckDuckGo: Web search
- Wikidata: Structured factual data
- DBpedia: Semantic encyclopedia
- OpenLibrary: Cultural and bibliographic data
"""

from .duckduckgo import search_duckduckgo
from .wikidata import search_wikidata
from .dbpedia import search_dbpedia
from .openlibrary import search_openlibrary

__all__ = [
    'search_duckduckgo',
    'search_wikidata',
    'search_dbpedia',
    'search_openlibrary'
]
