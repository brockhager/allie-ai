"""
Google Knowledge Graph API Integration

Requires API key from Google Cloud Console.
"""

import logging
import os
from typing import Dict, Any, List
import httpx

logger = logging.getLogger("allie.sources.google_kg")

GOOGLE_KG_API_URL = "https://kgsearch.googleapis.com/v1/entities:search"
GOOGLE_API_KEY = os.getenv("GOOGLE_KG_API_KEY")


async def search_google_kg(query: str, max_results: int = 3) -> Dict[str, Any]:
    """
    Search Google Knowledge Graph
    
    Args:
        query: Search query
        max_results: Maximum number of results
        
    Returns:
        Dictionary with search results
    """
    if not GOOGLE_API_KEY:
        logger.info("Google KG API key not configured")
        return {
            "success": False,
            "source": "google_kg",
            "results": [],
            "message": "API key not configured (set GOOGLE_KG_API_KEY env var)",
            "query": query
        }
    
    try:
        params = {
            "query": query,
            "limit": max_results,
            "indent": True,
            "key": GOOGLE_API_KEY
        }
        
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(GOOGLE_KG_API_URL, params=params)
            
            if response.status_code != 200:
                logger.warning(f"Google KG returned status {response.status_code}")
                return {
                    "success": False,
                    "source": "google_kg",
                    "results": [],
                    "message": f"API returned status {response.status_code}",
                    "query": query
                }
            
            data = response.json()
            items = data.get("itemListElement", [])
            
            if not items:
                logger.info(f"Google KG: No results for '{query}'")
                return {
                    "success": False,
                    "source": "google_kg",
                    "results": [],
                    "message": "No entities found",
                    "query": query
                }
            
            results = []
            for item in items:
                entity = item.get("result", {})
                name = entity.get("name", "")
                description = entity.get("description", "")
                detailed_desc = entity.get("detailedDescription", {})
                article_body = detailed_desc.get("articleBody", "")
                url = detailed_desc.get("url", "")
                
                # Combine description and article body
                text = f"{description}. {article_body}" if article_body else description
                
                results.append({
                    "title": name,
                    "text": text,
                    "url": url,
                    "types": entity.get("@type", []),
                    "score": item.get("resultScore", 0),
                    "source": "google_kg"
                })
            
            logger.info(f"Google KG: Found {len(results)} results for '{query}'")
            return {
                "success": True,
                "source": "google_kg",
                "results": results,
                "query": query
            }
                
    except Exception as e:
        logger.error(f"Google KG search error: {e}")
        return {
            "success": False,
            "source": "google_kg",
            "results": [],
            "error": str(e),
            "query": query
        }
