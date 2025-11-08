"""
Knowledge Retrieval Orchestrator

Coordinates searches across multiple sources and synthesizes results.
Implements the workflow: Memory → DuckDuckGo → Wikidata → DBpedia → OpenLibrary
"""

import logging
from typing import Dict, Any, List, Optional
import asyncio

from sources.duckduckgo import search_duckduckgo
from sources.wikidata import search_wikidata
from sources.dbpedia import search_dbpedia
from sources.openlibrary import search_openlibrary

logger = logging.getLogger("allie.retrieval")


async def search_all_sources(
    query: str,
    memory_results: Optional[List[str]] = None,
    max_results_per_source: int = 3
) -> Dict[str, Any]:
    """
    Search all external sources and synthesize results
    
    Args:
        query: Search query
        memory_results: Facts already found in memory
        max_results_per_source: Max results per source
        
    Returns:
        Synthesized results from all sources
    """
    logger.info(f"Starting multi-source search for: '{query}'")
    
    # Clean query - remove question words for better search results
    clean_query = query.lower()
    for word in ["what", "how", "where", "when", "who", "why", "is", "are", "the"]:
        clean_query = clean_query.replace(f"{word} ", "")
    clean_query = clean_query.strip()
    
    if not clean_query:
        clean_query = query  # Fallback to original if everything was removed
    
    logger.info(f"Cleaned query: '{clean_query}'")
    
    # Search all sources in parallel
    tasks = [
        search_duckduckgo(clean_query, max_results_per_source),
        search_wikidata(clean_query, max_results_per_source),
        search_dbpedia(clean_query, max_results_per_source),
        search_openlibrary(clean_query, max_results_per_source)
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    duckduckgo_result = results[0] if not isinstance(results[0], Exception) else {"success": False, "error": str(results[0])}
    wikidata_result = results[1] if not isinstance(results[1], Exception) else {"success": False, "error": str(results[1])}
    dbpedia_result = results[2] if not isinstance(results[2], Exception) else {"success": False, "error": str(results[2])}
    openlibrary_result = results[3] if not isinstance(results[3], Exception) else {"success": False, "error": str(results[3])}
    
    # Synthesize all results
    synthesis = synthesize_results(
        query=query,
        memory_results=memory_results or [],
        duckduckgo=duckduckgo_result,
        wikidata=wikidata_result,
        dbpedia=dbpedia_result,
        openlibrary=openlibrary_result
    )
    
    return synthesis


def synthesize_results(
    query: str,
    memory_results: List[str],
    duckduckgo: Dict[str, Any],
    wikidata: Dict[str, Any],
    dbpedia: Dict[str, Any],
    openlibrary: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Synthesize results from all sources into a coherent response
    
    Returns:
        {
            "success": bool,
            "query": str,
            "memory_used": bool,
            "sources_used": [str],
            "synthesized_text": str,
            "facts_to_store": [dict],
            "all_results": {source: result}
        }
    """
    sources_used = []
    facts_to_store = []
    text_parts = []
    
    # 1. Memory results
    if memory_results:
        sources_used.append("memory")
        text_parts.append(f"From memory: {' '.join(memory_results[:2])}")
    
    # 2. DuckDuckGo results
    if duckduckgo.get("success") and duckduckgo.get("results"):
        sources_used.append("duckduckgo")
        
        # Add instant answer if available
        if duckduckgo.get("instant_answer"):
            text_parts.append(duckduckgo["instant_answer"])
            facts_to_store.append({
                "fact": duckduckgo["instant_answer"],
                "source": "duckduckgo",
                "category": categorize_query(query)
            })
        
        # Add top results
        for result in duckduckgo["results"][:2]:
            text = result.get("text", "")
            if text and text not in [f.get("fact") for f in facts_to_store]:
                text_parts.append(text)
                facts_to_store.append({
                    "fact": text,
                    "source": "duckduckgo",
                    "category": categorize_query(query)
                })
    
    # 3. Wikidata results
    if wikidata.get("success") and wikidata.get("results"):
        sources_used.append("wikidata")
        
        for result in wikidata["results"][:1]:  # Top result only
            facts = result.get("facts", {})
            if facts:
                # Extract key facts
                for key, value in list(facts.items())[:3]:
                    if key != "label" and key != "description":
                        fact_text = f"{result.get('title', 'Entity')}: {key} = {value}"
                        facts_to_store.append({
                            "fact": fact_text,
                            "source": "wikidata",
                            "category": categorize_query(query),
                            "confidence": 0.95
                        })
    
    # 4. DBpedia results
    if dbpedia.get("success") and dbpedia.get("results"):
        sources_used.append("dbpedia")
        
        for result in dbpedia["results"][:1]:  # Top result only
            description = result.get("text", "")
            if description:
                # Truncate long descriptions
                if len(description) > 200:
                    description = description[:200] + "..."
                text_parts.append(description)
                
                facts_to_store.append({
                    "fact": description,
                    "source": "dbpedia",
                    "category": categorize_query(query),
                    "confidence": 0.9
                })
    
    # 5. OpenLibrary results
    if openlibrary.get("success") and openlibrary.get("results"):
        sources_used.append("openlibrary")
        
        for result in openlibrary["results"][:1]:  # Top result only
            title = result.get("title", "")
            text = result.get("text", "")
            if title and text:
                book_info = f"{title}: {text}"
                text_parts.append(book_info)
                
                facts_to_store.append({
                    "fact": book_info,
                    "source": "openlibrary",
                    "category": "cultural",
                    "confidence": 0.85
                })
    
    # Build synthesized response
    if text_parts:
        synthesized_text = " ".join(text_parts)
        success = True
    else:
        synthesized_text = f"I couldn't find specific information about '{query}' in my knowledge sources. You might want to try rephrasing your question or being more specific."
        success = False
    
    return {
        "success": success,
        "query": query,
        "memory_used": len(memory_results) > 0,
        "sources_used": sources_used,
        "synthesized_text": synthesized_text,
        "facts_to_store": facts_to_store,
        "all_results": {
            "memory": memory_results,
            "duckduckgo": duckduckgo,
            "wikidata": wikidata,
            "dbpedia": dbpedia,
            "openlibrary": openlibrary
        }
    }


def categorize_query(query: str) -> str:
    """
    Categorize a query to determine the appropriate fact category
    
    Args:
        query: The search query
        
    Returns:
        Category string
    """
    query_lower = query.lower()
    
    # Geography keywords
    if any(word in query_lower for word in ['capital', 'city', 'country', 'state', 'location', 'where', 'population', 'area']):
        return "geography"
    
    # History keywords
    if any(word in query_lower for word in ['when', 'history', 'war', 'ancient', 'century', 'year', 'founded', 'discovered']):
        return "history"
    
    # Science keywords
    if any(word in query_lower for word in ['planet', 'element', 'formula', 'theory', 'scientific', 'chemical', 'physics']):
        return "science"
    
    # Technology keywords
    if any(word in query_lower for word in ['computer', 'software', 'program', 'algorithm', 'code', 'technology', 'internet']):
        return "technology"
    
    # Biography keywords
    if any(word in query_lower for word in ['who is', 'who was', 'born', 'died', 'lived', 'person', 'people']):
        return "biography"
    
    # Cultural keywords
    if any(word in query_lower for word in ['book', 'author', 'artist', 'music', 'movie', 'film', 'art', 'culture']):
        return "cultural"
    
    # Mathematics keywords
    if any(word in query_lower for word in ['calculate', 'equation', 'math', 'number', 'plus', 'minus', 'times', 'divided']):
        return "mathematics"
    
    return "general"


async def search_with_memory_first(
    query: str,
    memory_search_func: callable,
    max_results: int = 5
) -> Dict[str, Any]:
    """
    Complete search workflow: Check memory first, then external sources
    
    Args:
        query: Search query
        memory_search_func: Function to search hybrid memory
        max_results: Maximum results to return
        
    Returns:
        Complete search results with memory and external sources
    """
    # 1. Check memory first
    memory_results = memory_search_func(query, limit=max_results)
    memory_facts = [result["fact"] for result in memory_results] if memory_results else []
    
    logger.info(f"Memory search for '{query}': found {len(memory_facts)} results")
    
    # 2. If memory has good results, use those
    if len(memory_facts) >= 2:
        logger.info(f"Using memory results for '{query}'")
        return {
            "success": True,
            "query": query,
            "memory_used": True,
            "sources_used": ["memory"],
            "synthesized_text": " ".join(memory_facts[:2]),
            "facts_to_store": [],  # Nothing new to store
            "memory_results": memory_results
        }
    
    # 3. Search external sources
    external_results = await search_all_sources(
        query=query,
        memory_results=memory_facts,
        max_results_per_source=3
    )
    
    return external_results
