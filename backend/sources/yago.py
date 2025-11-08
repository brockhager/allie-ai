"""
YAGO Knowledge Graph Integration

YAGO is a large semantic knowledge base derived from Wikipedia, WordNet and GeoNames.
"""

import logging
from typing import Dict, Any, List
import httpx

logger = logging.getLogger("allie.sources.yago")

YAGO_SPARQL_ENDPOINT = "https://yago-knowledge.org/sparql"


async def search_yago(query: str, max_results: int = 3) -> Dict[str, Any]:
    """
    Search YAGO knowledge graph
    
    Args:
        query: Search query
        max_results: Maximum number of results
        
    Returns:
        Dictionary with search results
    """
    try:
        # SPARQL query to search for entities
        sparql_query = f"""
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX yago: <http://yago-knowledge.org/resource/>
        PREFIX schema: <http://schema.org/>
        
        SELECT DISTINCT ?entity ?label ?description WHERE {{
            ?entity rdfs:label ?label .
            OPTIONAL {{ ?entity schema:description ?description }}
            FILTER(CONTAINS(LCASE(?label), LCASE("{query}")))
        }}
        LIMIT {max_results}
        """
        
        params = {
            "query": sparql_query,
            "format": "json"
        }
        
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(YAGO_SPARQL_ENDPOINT, params=params)
            
            if response.status_code != 200:
                logger.warning(f"YAGO SPARQL returned status {response.status_code}")
                return {
                    "success": False,
                    "source": "yago",
                    "results": [],
                    "message": f"API returned status {response.status_code}",
                    "query": query
                }
            
            data = response.json()
            bindings = data.get("results", {}).get("bindings", [])
            
            if not bindings:
                logger.info(f"YAGO: No results for '{query}'")
                return {
                    "success": False,
                    "source": "yago",
                    "results": [],
                    "message": "No entities found",
                    "query": query
                }
            
            results = []
            for binding in bindings:
                entity = binding.get("entity", {}).get("value", "")
                label = binding.get("label", {}).get("value", "")
                description = binding.get("description", {}).get("value", "")
                
                if label:
                    results.append({
                        "uri": entity,
                        "title": label,
                        "text": description,
                        "source": "yago"
                    })
            
            if results:
                logger.info(f"YAGO: Found {len(results)} results for '{query}'")
                return {
                    "success": True,
                    "source": "yago",
                    "results": results,
                    "query": query
                }
            else:
                return {
                    "success": False,
                    "source": "yago",
                    "results": [],
                    "message": "No valid entities found",
                    "query": query
                }
                
    except Exception as e:
        logger.error(f"YAGO search error: {e}")
        return {
            "success": False,
            "source": "yago",
            "results": [],
            "error": str(e),
            "query": query
        }
