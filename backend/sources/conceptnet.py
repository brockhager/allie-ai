"""
ConceptNet Integration

ConceptNet is a semantic network that describes general human knowledge
in the form of relations between concepts.
"""

import logging
from typing import Dict, Any, List
import httpx
from urllib.parse import quote

logger = logging.getLogger("allie.sources.conceptnet")

CONCEPTNET_API_URL = "https://api.conceptnet.io"


async def search_conceptnet(query: str, max_results: int = 5) -> Dict[str, Any]:
    """
    Search ConceptNet for concept relationships
    
    Args:
        query: Search query (concept)
        max_results: Maximum number of edges to return
        
    Returns:
        Dictionary with search results
    """
    try:
        # Format query for ConceptNet (replace spaces with underscores)
        concept = query.replace(" ", "_")
        url = f"{CONCEPTNET_API_URL}/c/en/{quote(concept)}"
        
        params = {
            "limit": max_results * 2  # Get more edges to filter
        }
        
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(url, params=params)
            
            if response.status_code == 404:
                logger.info(f"ConceptNet: No concept found for '{query}'")
                return {
                    "success": False,
                    "source": "conceptnet",
                    "results": [],
                    "message": "Concept not found",
                    "query": query
                }
            
            if response.status_code != 200:
                logger.warning(f"ConceptNet returned status {response.status_code}")
                return {
                    "success": False,
                    "source": "conceptnet",
                    "results": [],
                    "message": f"API returned status {response.status_code}",
                    "query": query
                }
            
            data = response.json()
            edges = data.get("edges", [])
            
            if not edges:
                logger.info(f"ConceptNet: No relationships for '{query}'")
                return {
                    "success": False,
                    "source": "conceptnet",
                    "results": [],
                    "message": "No relationships found",
                    "query": query
                }
            
            # Extract meaningful relationships
            results = []
            seen_facts = set()
            
            for edge in edges[:max_results * 3]:
                rel = edge.get("rel", {}).get("label", "")
                start = edge.get("start", {}).get("label", "")
                end = edge.get("end", {}).get("label", "")
                weight = edge.get("weight", 0)
                
                # Filter for high-quality relationships
                if weight < 1.0:
                    continue
                
                # Create human-readable fact
                fact = f"{start} {rel} {end}"
                
                if fact not in seen_facts and len(fact) < 200:
                    seen_facts.add(fact)
                    results.append({
                        "text": fact,
                        "relation": rel,
                        "start": start,
                        "end": end,
                        "weight": weight,
                        "source": "conceptnet"
                    })
                
                if len(results) >= max_results:
                    break
            
            if results:
                logger.info(f"ConceptNet: Found {len(results)} relationships for '{query}'")
                return {
                    "success": True,
                    "source": "conceptnet",
                    "results": results,
                    "query": query
                }
            else:
                return {
                    "success": False,
                    "source": "conceptnet",
                    "results": [],
                    "message": "No high-quality relationships found",
                    "query": query
                }
                
    except Exception as e:
        logger.error(f"ConceptNet search error: {e}")
        return {
            "success": False,
            "source": "conceptnet",
            "results": [],
            "error": str(e),
            "query": query
        }
