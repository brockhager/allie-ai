"""
OpenLibrary Search Integration

Provides cultural, historical, and bibliographic information from OpenLibrary
and Internet Archive APIs.
"""

import logging
from typing import Dict, Any, List
import httpx
from urllib.parse import quote_plus

logger = logging.getLogger("allie.sources.openlibrary")

OPENLIBRARY_SEARCH_URL = "https://openlibrary.org/search.json"
OPENLIBRARY_WORKS_URL = "https://openlibrary.org/works"
OPENLIBRARY_AUTHORS_URL = "https://openlibrary.org/authors"


async def search_openlibrary(query: str, max_results: int = 5) -> Dict[str, Any]:
    """
    Search OpenLibrary for books, authors, and cultural information
    
    Args:
        query: Search query
        max_results: Maximum number of results
        
    Returns:
        Dictionary with search results
    """
    try:
        params = {
            "q": query,
            "limit": max_results,
            "fields": "key,title,author_name,first_publish_year,subject,publisher,isbn",
            "mode": "everything"
        }
        
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(OPENLIBRARY_SEARCH_URL, params=params)
            
            if response.status_code == 200:
                data = response.json()
                results = []
                
                for doc in data.get("docs", []):
                    work_key = doc.get("key", "")
                    title = doc.get("title", "")
                    authors = doc.get("author_name", [])
                    year = doc.get("first_publish_year")
                    subjects = doc.get("subject", [])[:5]  # Limit subjects
                    publishers = doc.get("publisher", [])[:3]  # Limit publishers
                    
                    # Build descriptive text
                    text_parts = []
                    if authors:
                        text_parts.append(f"by {', '.join(authors)}")
                    if year:
                        text_parts.append(f"({year})")
                    if subjects:
                        text_parts.append(f"Subjects: {', '.join(subjects[:3])}")
                    
                    results.append({
                        "work_key": work_key,
                        "title": title,
                        "text": " ".join(text_parts),
                        "authors": authors,
                        "year": year,
                        "subjects": subjects,
                        "publishers": publishers,
                        "url": f"https://openlibrary.org{work_key}",
                        "source": "openlibrary"
                    })
                
                if results:
                    logger.info(f"OpenLibrary: Found {len(results)} results for '{query}'")
                    return {
                        "success": True,
                        "source": "openlibrary",
                        "results": results,
                        "query": query
                    }
                else:
                    logger.info(f"OpenLibrary: No results for '{query}'")
                    return {
                        "success": False,
                        "source": "openlibrary",
                        "results": [],
                        "message": "No books or authors found",
                        "query": query
                    }
            else:
                logger.warning(f"OpenLibrary API returned status {response.status_code}")
                return {
                    "success": False,
                    "source": "openlibrary",
                    "results": [],
                    "error": f"HTTP {response.status_code}",
                    "query": query
                }
                
    except Exception as e:
        logger.error(f"OpenLibrary search error: {e}")
        return {
            "success": False,
            "source": "openlibrary",
            "results": [],
            "error": str(e),
            "query": query
        }


async def get_work_details(work_key: str) -> Dict[str, Any]:
    """
    Get detailed information about a work
    
    Args:
        work_key: OpenLibrary work key (e.g., /works/OL45804W)
        
    Returns:
        Dictionary with work details
    """
    try:
        url = f"https://openlibrary.org{work_key}.json"
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url)
            
            if response.status_code == 200:
                data = response.json()
                
                return {
                    "success": True,
                    "title": data.get("title"),
                    "description": data.get("description", {}).get("value", data.get("description", "")),
                    "subjects": data.get("subjects", []),
                    "first_sentence": data.get("first_sentence", {}).get("value", ""),
                    "covers": data.get("covers", [])
                }
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}"
                }
                
    except Exception as e:
        logger.error(f"Error getting work details: {e}")
        return {
            "success": False,
            "error": str(e)
        }


async def get_author_details(author_key: str) -> Dict[str, Any]:
    """
    Get detailed information about an author
    
    Args:
        author_key: OpenLibrary author key (e.g., /authors/OL23919A)
        
    Returns:
        Dictionary with author details
    """
    try:
        url = f"https://openlibrary.org{author_key}.json"
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url)
            
            if response.status_code == 200:
                data = response.json()
                
                bio = data.get("bio", {})
                if isinstance(bio, dict):
                    bio = bio.get("value", "")
                
                return {
                    "success": True,
                    "name": data.get("name"),
                    "bio": bio,
                    "birth_date": data.get("birth_date"),
                    "death_date": data.get("death_date"),
                    "links": data.get("links", [])
                }
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}"
                }
                
    except Exception as e:
        logger.error(f"Error getting author details: {e}")
        return {
            "success": False,
            "error": str(e)
        }
