import asyncio
import sys
sys.path.insert(0, '.')

# Test the new query functions
from server import (
    query_duckduckgo, query_wikidata, query_dbpedia,
    query_wikipedia, query_openlibrary, query_musicbrainz,
    query_restcountries, query_arxiv, query_pubmed
)
import httpx

async def test_all_sources():
    """Test all the new query sources"""
    print("=" * 80)
    print("Testing All Knowledge Sources")
    print("=" * 80)

    test_topics = {
        "python programming": ["wikipedia", "openlibrary", "arxiv"],
        "united states": ["wikipedia", "restcountries", "duckduckgo"],
        "mozart": ["wikipedia", "musicbrainz", "dbpedia"],
        "machine learning": ["wikipedia", "arxiv", "pubmed"],
        "harry potter": ["wikipedia", "openlibrary", "duckduckgo"]
    }

    async with httpx.AsyncClient(timeout=httpx.Timeout(15.0)) as http_client:
        for topic, expected_sources in test_topics.items():
            print(f"\n--- Testing topic: '{topic}' ---")

            # Test all sources
            sources_to_test = [
                ("DuckDuckGo", query_duckduckgo),
                ("Wikidata", query_wikidata),
                ("DBpedia", query_dbpedia),
                ("Wikipedia", query_wikipedia),
                ("Open Library", query_openlibrary),
                ("MusicBrainz", query_musicbrainz),
                ("REST Countries", query_restcountries),
                ("ArXiv", query_arxiv),
                ("PubMed", query_pubmed)
            ]

            total_results = 0
            working_sources = 0

            for source_name, query_func in sources_to_test:
                try:
                    result = await query_func(http_client, topic)
                    success = result.get("success", False)
                    results_count = len(result.get("results", []))

                    status = "✓" if success else "✗"
                    print(f"  {status} {source_name}: {results_count} results")

                    if success:
                        working_sources += 1
                        total_results += results_count

                        # Show sample result
                        if results_count > 0:
                            sample = result["results"][0].get("text", "")[:80]
                            print(f"      Sample: {sample}...")

                except Exception as e:
                    print(f"  ✗ {source_name}: Exception - {e}")

            print(f"  Summary: {working_sources}/{len(sources_to_test)} sources working, {total_results} total results")

if __name__ == "__main__":
    asyncio.run(test_all_sources())