"""
OpenStreetMap Nominatim Integration

Provides geocoding and distance calculations using OpenStreetMap data.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import httpx
import math

logger = logging.getLogger("allie.sources.nominatim")

NOMINATIM_API_URL = "https://nominatim.openstreetmap.org"


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate distance between two points on Earth using Haversine formula
    
    Returns:
        Distance in miles
    """
    # Earth radius in miles
    R = 3959.0
    
    # Convert to radians
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    
    # Haversine formula
    a = (math.sin(dLat / 2) ** 2 + 
         math.cos(lat1_rad) * math.cos(lat2_rad) * 
         math.sin(dLon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    distance = R * c
    return distance


async def geocode_location(location: str, client: httpx.AsyncClient) -> Optional[Tuple[float, float, str]]:
    """
    Geocode a location string to coordinates
    
    Returns:
        Tuple of (latitude, longitude, display_name) or None
    """
    try:
        headers = {
            "User-Agent": "AllieAI/1.0 (Educational AI Assistant; https://github.com/brockhager/allie-ai)"
        }
        
        params = {
            "q": location,
            "format": "json",
            "limit": 1
        }
        
        response = await client.get(
            f"{NOMINATIM_API_URL}/search",
            params=params,
            headers=headers
        )
        
        if response.status_code == 200:
            data = response.json()
            if data:
                result = data[0]
                return (
                    float(result["lat"]),
                    float(result["lon"]),
                    result.get("display_name", location)
                )
        
        return None
        
    except Exception as e:
        logger.error(f"Geocoding error for '{location}': {e}")
        return None


async def search_nominatim(query: str, max_results: int = 3) -> Dict[str, Any]:
    """
    Search Nominatim for geographic information and distance calculations
    
    Detects queries like "distance from X to Y" or "how far is X from Y"
    
    Args:
        query: Search query
        max_results: Maximum number of results
        
    Returns:
        Dictionary with search results
    """
    try:
        # Detect distance queries
        query_lower = query.lower()
        distance_patterns = [
            ("distance from", "to"),
            ("how far", "from"),
            ("how far is", "from"),
            ("how far is it from", "to")
        ]
        
        from_location = None
        to_location = None
        
        for start_pattern, separator in distance_patterns:
            if start_pattern in query_lower:
                # Extract locations
                after_start = query_lower.split(start_pattern, 1)[1].strip()
                if separator in after_start:
                    parts = after_start.split(separator, 1)
                    from_location = parts[0].strip()
                    to_location = parts[1].strip().rstrip('?').strip()
                    break
        
        if not from_location or not to_location:
            logger.info(f"Nominatim: Not a distance query: '{query}'")
            return {
                "success": False,
                "source": "nominatim",
                "results": [],
                "message": "Not a distance query",
                "query": query
            }
        
        logger.info(f"Nominatim: Distance query from '{from_location}' to '{to_location}'")
        
        # Geocode both locations
        async with httpx.AsyncClient(timeout=15.0) as client:
            from_coords = await geocode_location(from_location, client)
            to_coords = await geocode_location(to_location, client)
        
        if not from_coords or not to_coords:
            logger.warning(f"Nominatim: Could not geocode locations")
            return {
                "success": False,
                "source": "nominatim",
                "results": [],
                "message": "Could not find one or both locations",
                "query": query
            }
        
        # Calculate distance
        from_lat, from_lon, from_name = from_coords
        to_lat, to_lon, to_name = to_coords
        
        distance_miles = haversine_distance(from_lat, from_lon, to_lat, to_lon)
        distance_km = distance_miles * 1.60934
        
        # Estimate driving time (assuming average 60 mph)
        estimated_hours = distance_miles / 60
        hours = int(estimated_hours)
        minutes = int((estimated_hours - hours) * 60)
        
        result_text = (
            f"The distance from {from_name} to {to_name} is approximately "
            f"{distance_miles:.1f} miles ({distance_km:.1f} km). "
            f"Estimated driving time at 60 mph: {hours}h {minutes}m (actual time may vary based on route and traffic)."
        )
        
        logger.info(f"Nominatim: Calculated distance: {distance_miles:.1f} miles")
        
        return {
            "success": True,
            "source": "nominatim",
            "results": [{
                "text": result_text,
                "from_location": from_name,
                "to_location": to_name,
                "distance_miles": round(distance_miles, 1),
                "distance_km": round(distance_km, 1),
                "estimated_time": f"{hours}h {minutes}m",
                "source": "nominatim"
            }],
            "query": query
        }
        
    except Exception as e:
        logger.error(f"Nominatim search error: {e}")
        return {
            "success": False,
            "source": "nominatim",
            "results": [],
            "error": str(e),
            "query": query
        }
