"""
DBpedia Search Integration

Provides semantic encyclopedia data from DBpedia using SPARQL endpoint.
"""

import logging
from typing import Dict, Any, List
import httpx
from urllib.parse import quote_plus

logger = logging.getLogger("allie.sources.dbpedia")

DBPEDIA_SPARQL_URL = "https://dbpedia.org/sparql"
DBPEDIA_LOOKUP_URL = "https://lookup.dbpedia.org/api/search"


async def search_dbpedia(query: str, max_results: int = 5) -> Dict[str, Any]:
    """
    Search DBpedia for semantic encyclopedia data
    
    Args:
        query: Search query
        max_results: Maximum number of results
        
    Returns:
        Dictionary with search results
    """
    try:
        # Use DBpedia Lookup service
        params = {
            "query": query,
            "maxResults": max_results,
            "format": "json"
        }
        
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(DBPEDIA_LOOKUP_URL, params=params)
            
            if response.status_code == 200:
                data = response.json()
                results = []
                
                for doc in data.get("docs", []):
                    resource_uri = doc.get("resource", [""])[0]
                    label = doc.get("label", [""])[0]
                    description = doc.get("comment", [""])[0] if doc.get("comment") else ""
                    categories = doc.get("category", [])
                    
                    # Get additional facts from resource
                    facts = await get_resource_facts(client, resource_uri)
                    
                    results.append({
                        "uri": resource_uri,
                        "title": label,
                        "text": description,
                        "categories": categories,
                        "facts": facts,
                        "source": "dbpedia"
                    })
                
                if results:
                    logger.info(f"DBpedia: Found {len(results)} results for '{query}'")
                    return {
                        "success": True,
                        "source": "dbpedia",
                        "results": results,
                        "query": query
                    }
                else:
                    logger.info(f"DBpedia: No results for '{query}'")
                    return {
                        "success": False,
                        "source": "dbpedia",
                        "results": [],
                        "message": "No resources found",
                        "query": query
                    }
            else:
                logger.warning(f"DBpedia Lookup returned status {response.status_code}")
                return {
                    "success": False,
                    "source": "dbpedia",
                    "results": [],
                    "error": f"HTTP {response.status_code}",
                    "query": query
                }
                
    except Exception as e:
        logger.error(f"DBpedia search error: {e}")
        return {
            "success": False,
            "source": "dbpedia",
            "results": [],
            "error": str(e),
            "query": query
        }


async def get_resource_facts(client: httpx.AsyncClient, resource_uri: str) -> Dict[str, Any]:
    """
    Get facts about a DBpedia resource using SPARQL
    
    Args:
        client: HTTP client
        resource_uri: DBpedia resource URI
        
    Returns:
        Dictionary of facts
    """
    try:
        # SPARQL query to get basic facts
        sparql_query = f"""
        SELECT ?property ?value WHERE {{
            <{resource_uri}> ?property ?value .
            FILTER (
                ?property = dbo:abstract ||
                ?property = dbo:populationTotal ||
                ?property = dbo:areaTotal ||
                ?property = dbp:population ||
                ?property = dbo:capital ||
                ?property = dbo:largestCity ||
                ?property = dbo:foundingDate ||
                ?property = dbo:country
            )
            FILTER (lang(?value) = 'en' || !isLiteral(?value))
        }} LIMIT 10
        """
        
        params = {
            "query": sparql_query,
            "format": "json"
        }
        
        response = await client.get(DBPEDIA_SPARQL_URL, params=params, timeout=10.0)
        
        if response.status_code == 200:
            data = response.json()
            facts = {}
            
            for binding in data.get("results", {}).get("bindings", []):
                prop = binding.get("property", {}).get("value", "")
                val = binding.get("value", {}).get("value", "")
                
                # Extract property name from URI
                prop_name = prop.split("/")[-1].split("#")[-1]
                
                if prop_name and val:
                    # Limit abstract length
                    if prop_name == "abstract":
                        val = val[:500] + "..." if len(val) > 500 else val
                    
                    facts[prop_name] = val
            
            return facts
        else:
            return {}
            
    except Exception as e:
        logger.debug(f"Error getting resource facts: {e}")
        return {}


async def sparql_query_dbpedia(query: str) -> Dict[str, Any]:
    """
    Execute a SPARQL query against DBpedia
    
    Args:
        query: SPARQL query string
        
    Returns:
        Query results
    """
    try:
        params = {
            "query": query,
            "format": "json"
        }
        
        headers = {
            "Accept": "application/sparql-results+json",
            "User-Agent": "AllieAI/1.0"
        }
        
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.get(DBPEDIA_SPARQL_URL, params=params, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "success": True,
                    "results": data.get("results", {}).get("bindings", [])
                }
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}"
                }
                
    except Exception as e:
        logger.error(f"DBpedia SPARQL query error: {e}")
        return {
            "success": False,
            "error": str(e)
        }
