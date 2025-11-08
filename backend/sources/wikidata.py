"""
Wikidata Search Integration

Provides structured factual data from Wikidata using the MediaWiki API
and SPARQL endpoint.
"""

import logging
from typing import Dict, Any, List, Optional
import httpx
from urllib.parse import quote_plus

logger = logging.getLogger("allie.sources.wikidata")

WIKIDATA_API_URL = "https://www.wikidata.org/w/api.php"
WIKIDATA_SPARQL_URL = "https://query.wikidata.org/sparql"


async def search_wikidata(query: str, max_results: int = 5) -> Dict[str, Any]:
    """
    Search Wikidata for structured facts
    
    Args:
        query: Search query
        max_results: Maximum number of results
        
    Returns:
        Dictionary with search results including entities and facts
    """
    try:
        # Add headers to appear as a legitimate client
        headers = {
            "User-Agent": "AllieAI/1.0 (Educational AI Assistant; https://github.com/brockhager/allie-ai)",
            "Accept": "application/json"
        }
        
        # Search for entities
        params = {
            "action": "wbsearchentities",
            "format": "json",
            "language": "en",
            "search": query,
            "limit": max_results
        }
        
        async with httpx.AsyncClient(timeout=15.0, headers=headers) as client:
            response = await client.get(WIKIDATA_API_URL, params=params)
            
            if response.status_code == 200:
                data = response.json()
                results = []
                
                for item in data.get("search", []):
                    entity_id = item.get("id")
                    label = item.get("label", "")
                    description = item.get("description", "")
                    
                    # Get detailed facts about this entity
                    entity_data = await get_entity_facts(client, entity_id)
                    
                    results.append({
                        "entity_id": entity_id,
                        "title": label,
                        "text": description,
                        "url": f"https://www.wikidata.org/wiki/{entity_id}",
                        "facts": entity_data,
                        "source": "wikidata"
                    })
                
                if results:
                    logger.info(f"Wikidata: Found {len(results)} entities for '{query}'")
                    return {
                        "success": True,
                        "source": "wikidata",
                        "results": results,
                        "query": query
                    }
                else:
                    logger.info(f"Wikidata: No results for '{query}'")
                    return {
                        "success": False,
                        "source": "wikidata",
                        "results": [],
                        "message": "No entities found",
                        "query": query
                    }
            else:
                logger.warning(f"Wikidata API returned status {response.status_code}")
                return {
                    "success": False,
                    "source": "wikidata",
                    "results": [],
                    "error": f"HTTP {response.status_code}",
                    "query": query
                }
                
    except Exception as e:
        logger.error(f"Wikidata search error: {e}")
        return {
            "success": False,
            "source": "wikidata",
            "results": [],
            "error": str(e),
            "query": query
        }


async def get_entity_facts(client: httpx.AsyncClient, entity_id: str) -> Dict[str, Any]:
    """
    Get structured facts about a Wikidata entity
    
    Args:
        client: HTTP client
        entity_id: Wikidata entity ID (e.g., Q90)
        
    Returns:
        Dictionary of facts about the entity
    """
    try:
        params = {
            "action": "wbgetentities",
            "format": "json",
            "ids": entity_id,
            "languages": "en"
        }
        
        response = await client.get(WIKIDATA_API_URL, params=params)
        
        if response.status_code == 200:
            data = response.json()
            entity = data.get("entities", {}).get(entity_id, {})
            
            facts = {}
            
            # Extract label and description
            labels = entity.get("labels", {})
            if "en" in labels:
                facts["label"] = labels["en"]["value"]
            
            descriptions = entity.get("descriptions", {})
            if "en" in descriptions:
                facts["description"] = descriptions["en"]["value"]
            
            # Extract key claims/properties
            claims = entity.get("claims", {})
            
            # Extract common properties
            property_map = {
                "P31": "instance_of",
                "P17": "country",
                "P36": "capital",
                "P1082": "population",
                "P625": "coordinates",
                "P571": "inception",
                "P576": "dissolved",
                "P856": "website"
            }
            
            for prop_id, prop_name in property_map.items():
                if prop_id in claims:
                    claim_values = []
                    for claim in claims[prop_id]:
                        value = extract_claim_value(claim)
                        if value:
                            claim_values.append(value)
                    if claim_values:
                        facts[prop_name] = claim_values[0] if len(claim_values) == 1 else claim_values
            
            return facts
        else:
            return {}
            
    except Exception as e:
        logger.debug(f"Error getting entity facts: {e}")
        return {}


def extract_claim_value(claim: Dict[str, Any]) -> Optional[str]:
    """Extract human-readable value from a Wikidata claim"""
    try:
        mainsnak = claim.get("mainsnak", {})
        datavalue = mainsnak.get("datavalue", {})
        value_type = datavalue.get("type")
        value = datavalue.get("value")
        
        if value_type == "string":
            return value
        elif value_type == "wikibase-entityid":
            return value.get("id")  # Return entity ID
        elif value_type == "quantity":
            amount = value.get("amount", "")
            unit = value.get("unit", "")
            return f"{amount} {unit}".strip()
        elif value_type == "time":
            return value.get("time", "")
        elif value_type == "globecoordinate":
            lat = value.get("latitude")
            lon = value.get("longitude")
            return f"{lat}, {lon}"
        else:
            return str(value)
            
    except Exception:
        return None


async def sparql_query(query: str) -> Dict[str, Any]:
    """
    Execute a SPARQL query against Wikidata
    
    Args:
        query: SPARQL query string
        
    Returns:
        Query results
    """
    try:
        headers = {
            "Accept": "application/sparql-results+json",
            "User-Agent": "AllieAI/1.0"
        }
        
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.get(
                WIKIDATA_SPARQL_URL,
                params={"query": query},
                headers=headers
            )
            
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
        logger.error(f"SPARQL query error: {e}")
        return {
            "success": False,
            "error": str(e)
        }
