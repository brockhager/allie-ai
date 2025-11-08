#!/usr/bin/env python3
"""
Test script for search functionality
"""

import asyncio
import httpx
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def search_web(query: str):
    """Search the web using DuckDuckGo instant answers"""
    try:
        # Use DuckDuckGo instant answer API (no API key required)
        url = f"https://api.duckduckgo.com/?q={query}&format=json&no_html=1&skip_disambig=1"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        async with httpx.AsyncClient(timeout=10.0, headers=headers) as client:
            response = await client.get(url)
            if response.status_code == 200:
                data = response.json()
                results = []

                # Extract instant answer
                if data.get("Answer"):
                    results.append({
                        "title": "Instant Answer",
                        "text": data["Answer"],
                        "source": "DuckDuckGo Instant Answer"
                    })

                # Extract abstract
                if data.get("AbstractText"):
                    results.append({
                        "title": data.get("Heading", "Abstract"),
                        "text": data["AbstractText"],
                        "source": "DuckDuckGo Abstract"
                    })

                # Extract definition
                if data.get("Definition"):
                    results.append({
                        "title": "Definition",
                        "text": data["Definition"],
                        "source": "DuckDuckGo Definition"
                    })

                # Extract related topics (limit to 3-5)
                if not results and data.get("RelatedTopics"):
                    for topic in data["RelatedTopics"][:5]:
                        if topic.get("Text"):
                            results.append({
                                "title": topic.get("FirstURL", "Related Topic"),
                                "text": topic["Text"],
                                "source": "DuckDuckGo Related Topics"
                            })

                return {
                    "query": query,
                    "results": results[:5],  # Limit to top 5
                    "success": True
                }
            else:
                return {
                    "query": query,
                    "results": [],
                    "success": False,
                    "error": f"Search failed with status {response.status_code}"
                }
    except Exception as e:
        logger.warning(f"Web search failed: {e}")
        return {
            "query": query,
            "results": [],
            "success": False,
            "error": str(e)
        }

async def search_wikipedia(query: str):
    """Search Wikipedia for authoritative background information"""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        # Use Wikipedia API for summary
        # First, search for the page title
        search_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{query.replace(' ', '_')}"
        async with httpx.AsyncClient(timeout=10.0, headers=headers) as client:
            response = await client.get(search_url)
            if response.status_code == 200:
                data = response.json()
                return {
                    "query": query,
                    "title": data.get("title", query),
                    "summary": data.get("extract", ""),
                    "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
                    "success": True
                }
            else:
                # Try search endpoint if direct page lookup fails
                search_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={query}&format=json&utf8=1"
                response = await client.get(search_url)
                if response.status_code == 200:
                    data = response.json()
                    if data.get("query", {}).get("search"):
                        # Get the top result
                        top_result = data["query"]["search"][0]
                        page_title = top_result["title"]

                        # Get summary for the top result
                        summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{page_title.replace(' ', '_')}"
                        summary_response = await client.get(summary_url)
                        if summary_response.status_code == 200:
                            summary_data = summary_response.json()
                            return {
                                "query": query,
                                "title": summary_data.get("title", page_title),
                                "summary": summary_data.get("extract", ""),
                                "url": summary_data.get("content_urls", {}).get("desktop", {}).get("page", ""),
                                "success": True
                            }

                return {
                    "query": query,
                    "title": query,
                    "summary": "",
                    "url": "",
                    "success": False,
                    "error": "No Wikipedia page found"
                }
    except Exception as e:
        logger.warning(f"Wikipedia search failed: {e}")
        return {
            "query": query,
            "title": query,
            "summary": "",
            "url": "",
            "success": False,
            "error": str(e)
        }

async def test_search_functions():
    """Test the search functions"""
    print("Testing Search Functions")
    print("=" * 50)

    # Test web search
    print("\n1. Testing DuckDuckGo web search:")
    print("-" * 40)
    web_result = await search_web("What is the capital of France?")
    print(f"Query: {web_result.get('query')}")
    print(f"Success: {web_result.get('success')}")
    if web_result.get('success'):
        print(f"Results found: {len(web_result.get('results', []))}")
        for i, result in enumerate(web_result.get('results', [])[:2], 1):
            print(f"  {i}. {result.get('text', '')[:100]}...")
    else:
        print(f"Error: {web_result.get('error')}")

    # Test Wikipedia search
    print("\n2. Testing Wikipedia search:")
    print("-" * 40)
    wiki_result = await search_wikipedia("Paris France")
    print(f"Query: {wiki_result.get('query')}")
    print(f"Success: {wiki_result.get('success')}")
    if wiki_result.get('success'):
        print(f"Title: {wiki_result.get('title')}")
        print(f"Summary: {wiki_result.get('summary', '')[:200]}...")
        print(f"URL: {wiki_result.get('url', '')}")
    else:
        print(f"Error: {wiki_result.get('error')}")

if __name__ == "__main__":
    asyncio.run(test_search_functions())