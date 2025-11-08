"""
DuckDuckGo Search Integration

Provides web search capabilities using DuckDuckGo's instant answer API
and HTML search (no API key required).
"""

import logging
from typing import Dict, Any, List
import httpx
from urllib.parse import quote_plus

logger = logging.getLogger("allie.sources.duckduckgo")


async def search_duckduckgo(query: str, max_results: int = 5) -> Dict[str, Any]:
    """
    Search DuckDuckGo for web results
    
    Args:
        query: Search query
        max_results: Maximum number of results to return
        
    Returns:
        Dictionary with search results:
        {
            "success": bool,
            "source": "duckduckgo",
            "results": [
                {
                    "title": str,
                    "text": str,
                    "url": str (optional)
                }
            ],
            "instant_answer": str (if available)
        }
    """
    try:
        # Try instant answer API first
        url = f"https://api.duckduckgo.com/?q={quote_plus(query)}&format=json&no_html=1&skip_disambig=1"
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url)
            
            if response.status_code == 200:
                data = response.json()
                results = []
                instant_answer = None
                
                # Extract instant answer
                if data.get("Answer"):
                    instant_answer = data["Answer"]
                    results.append({
                        "title": "Instant Answer",
                        "text": data["Answer"],
                        "source": "duckduckgo_instant"
                    })
                
                # Extract abstract
                if data.get("AbstractText"):
                    results.append({
                        "title": data.get("Heading", "Abstract"),
                        "text": data["AbstractText"],
                        "url": data.get("AbstractURL"),
                        "source": "duckduckgo_abstract"
                    })
                
                # Extract definition
                if data.get("Definition"):
                    results.append({
                        "title": data.get("DefinitionSource", "Definition"),
                        "text": data["Definition"],
                        "url": data.get("DefinitionURL"),
                        "source": "duckduckgo_definition"
                    })
                
                # Extract related topics
                for topic in data.get("RelatedTopics", [])[:max_results]:
                    if isinstance(topic, dict) and topic.get("Text"):
                        results.append({
                            "title": topic.get("FirstURL", "").split("/")[-1].replace("_", " "),
                            "text": topic["Text"],
                            "url": topic.get("FirstURL"),
                            "source": "duckduckgo_related"
                        })
                
                if results:
                    logger.info(f"DuckDuckGo: Found {len(results)} results for '{query}'")
                    return {
                        "success": True,
                        "source": "duckduckgo",
                        "results": results[:max_results],
                        "instant_answer": instant_answer,
                        "query": query
                    }
                else:
                    logger.info(f"DuckDuckGo: No results for '{query}'")
                    return {
                        "success": False,
                        "source": "duckduckgo",
                        "results": [],
                        "message": "No results found",
                        "query": query
                    }
            else:
                logger.warning(f"DuckDuckGo API returned status {response.status_code}")
                return {
                    "success": False,
                    "source": "duckduckgo",
                    "results": [],
                    "error": f"HTTP {response.status_code}",
                    "query": query
                }
                
    except Exception as e:
        logger.error(f"DuckDuckGo search error: {e}")
        return {
            "success": False,
            "source": "duckduckgo",
            "results": [],
            "error": str(e),
            "query": query
        }


async def search_duckduckgo_html(query: str, max_results: int = 5) -> Dict[str, Any]:
    """
    Fallback HTML search for DuckDuckGo
    
    Note: This is a basic implementation. For production use,
    consider using a proper HTML parser and respecting robots.txt
    """
    try:
        url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
        
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            response = await client.get(url)
            
            if response.status_code == 200:
                # Basic extraction (would need proper HTML parsing for production)
                logger.info(f"DuckDuckGo HTML: Retrieved results for '{query}'")
                return {
                    "success": True,
                    "source": "duckduckgo_html",
                    "results": [],
                    "message": "HTML parsing not implemented",
                    "query": query
                }
            else:
                return {
                    "success": False,
                    "source": "duckduckgo_html",
                    "results": [],
                    "error": f"HTTP {response.status_code}",
                    "query": query
                }
                
    except Exception as e:
        logger.error(f"DuckDuckGo HTML search error: {e}")
        return {
            "success": False,
            "source": "duckduckgo_html",
            "results": [],
            "error": str(e),
            "query": query
        }
