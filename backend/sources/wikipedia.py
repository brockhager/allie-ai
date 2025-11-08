"""
Wikipedia Search Integration

Provides encyclopedia data from Wikipedia API.
"""

import logging
from typing import Dict, Any, List
import httpx

logger = logging.getLogger("allie.sources.wikipedia")

WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"


async def search_wikipedia(query: str, max_results: int = 3) -> Dict[str, Any]:
    """
    Search Wikipedia for encyclopedia articles
    
    Args:
        query: Search query
        max_results: Maximum number of results
        
    Returns:
        Dictionary with search results
    """
    try:
        # Add headers to appear as a legitimate client
        headers = {
            "User-Agent": "AllieAI/1.0 (Educational AI Assistant; https://github.com/brockhager/allie-ai)",
            "Accept": "application/json"
        }
        
        # First, search for relevant pages
        search_params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": max_results,
            "format": "json",
            "utf8": 1
        }
        
        async with httpx.AsyncClient(timeout=15.0, headers=headers) as client:
            search_response = await client.get(WIKIPEDIA_API_URL, params=search_params)
            
            if search_response.status_code != 200:
                logger.warning(f"Wikipedia search returned status {search_response.status_code}")
                return {
                    "success": False,
                    "source": "wikipedia",
                    "results": [],
                    "message": f"API returned status {search_response.status_code}",
                    "query": query
                }
            
            search_data = search_response.json()
            search_results = search_data.get("query", {}).get("search", [])
            
            if not search_results:
                logger.info(f"Wikipedia: No results for '{query}'")
                return {
                    "success": False,
                    "source": "wikipedia",
                    "results": [],
                    "message": "No articles found",
                    "query": query
                }
            
            # Get extracts for each page
            page_ids = [str(result["pageid"]) for result in search_results]
            extract_params = {
                "action": "query",
                "prop": "extracts|info",
                "pageids": "|".join(page_ids),
                "exintro": 1,
                "explaintext": 1,
                "exsentences": 3,
                "inprop": "url",
                "format": "json",
                "utf8": 1
            }
            
            extract_response = await client.get(WIKIPEDIA_API_URL, params=extract_params)
            
            if extract_response.status_code != 200:
                logger.warning(f"Wikipedia extract returned status {extract_response.status_code}")
                return {
                    "success": False,
                    "source": "wikipedia",
                    "results": [],
                    "message": "Failed to get article extracts",
                    "query": query
                }
            
            extract_data = extract_response.json()
            pages = extract_data.get("query", {}).get("pages", {})
            
            results = []
            for page_id, page_data in pages.items():
                if "extract" in page_data:
                    results.append({
                        "title": page_data.get("title", ""),
                        "text": page_data.get("extract", ""),
                        "url": page_data.get("fullurl", ""),
                        "source": "wikipedia"
                    })
            
            if results:
                logger.info(f"Wikipedia: Found {len(results)} results for '{query}'")
                return {
                    "success": True,
                    "source": "wikipedia",
                    "results": results,
                    "query": query
                }
            else:
                logger.info(f"Wikipedia: No extracts for '{query}'")
                return {
                    "success": False,
                    "source": "wikipedia",
                    "results": [],
                    "message": "No article content found",
                    "query": query
                }
                
    except Exception as e:
        logger.error(f"Wikipedia search error: {e}")
        return {
            "success": False,
            "source": "wikipedia",
            "results": [],
            "error": str(e),
            "query": query
        }
