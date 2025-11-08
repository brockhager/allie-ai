"""
Freebase Knowledge Graph Integration

Note: Freebase was shut down but data is available via Wikidata mappings
and archived endpoints. This uses the Freebase Easy API.
"""

import logging
from typing import Dict, Any, List
import httpx

logger = logging.getLogger("allie.sources.freebase")

# Freebase Easy - community maintained endpoint
FREEBASE_API_URL = "https://freebase-easy.herokuapp.com/search"


async def search_freebase(query: str, max_results: int = 3) -> Dict[str, Any]:
    """
    Search Freebase knowledge graph (archived/community endpoint)
    
    Note: This may have limited availability as Freebase was discontinued.
    Keeping for historical/compatibility reasons.
    
    Args:
        query: Search query
        max_results: Maximum number of results
        
    Returns:
        Dictionary with search results
    """
    try:
        params = {
            "query": query,
            "limit": max_results
        }
        
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(FREEBASE_API_URL, params=params)
            
            if response.status_code != 200:
                logger.info(f"Freebase: API unavailable (status {response.status_code})")
                return {
                    "success": False,
                    "source": "freebase",
                    "results": [],
                    "message": "Freebase API unavailable (discontinued service)",
                    "query": query
                }
            
            data = response.json()
            items = data.get("result", [])
            
            if not items:
                logger.info(f"Freebase: No results for '{query}'")
                return {
                    "success": False,
                    "source": "freebase",
                    "results": [],
                    "message": "No entities found",
                    "query": query
                }
            
            results = []
            for item in items:
                name = item.get("name", "")
                mid = item.get("mid", "")
                notable = item.get("notable", {})
                notable_name = notable.get("name", "") if isinstance(notable, dict) else ""
                
                text = f"{name} - {notable_name}" if notable_name else name
                
                results.append({
                    "title": name,
                    "text": text,
                    "mid": mid,
                    "notable": notable_name,
                    "source": "freebase"
                })
            
            logger.info(f"Freebase: Found {len(results)} results for '{query}'")
            return {
                "success": True,
                "source": "freebase",
                "results": results,
                "query": query
            }
                
    except Exception as e:
        logger.info(f"Freebase unavailable: {e}")
        return {
            "success": False,
            "source": "freebase",
            "results": [],
            "error": "Service unavailable",
            "query": query
        }
