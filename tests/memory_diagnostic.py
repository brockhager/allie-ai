#!/usr/bin/env python3
"""
Comprehensive Advanced Memory System Diagnostic

Tests advanced memory functionality, hybrid memory integration, and detects old memory references.
"""

import sys
import os
from pathlib import Path
import json
from datetime import datetime

# Add project root and advanced-memory to path
project_root = Path(__file__).parent
advanced_memory_path = project_root / "advanced-memory"
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(advanced_memory_path) not in sys.path:
    sys.path.insert(0, str(advanced_memory_path))

def test_advanced_memory_module():
    """Test 1: Confirm advanced_memory module is loaded and accessible"""
    print("🔍 Test 1: Advanced Memory Module Loading")
    print("-" * 50)

    try:
        from db import AllieMemoryDB
        from hybrid import HybridMemory
        print("✓ Successfully imported AllieMemoryDB and HybridMemory from advanced_memory")

        # Test database connection
        db = AllieMemoryDB()
        stats = db.get_statistics()
        print(f"✓ Database connection successful. Current stats: {stats}")
        db.close()

        # Test hybrid memory
        hybrid = HybridMemory()
        hybrid_stats = hybrid.get_statistics()
        print(f"✓ Hybrid memory initialized. Stats: {hybrid_stats}")

        return True, {"db_stats": stats, "hybrid_stats": hybrid_stats}

    except Exception as e:
        print(f"✗ Failed to load advanced memory: {e}")
        return False, str(e)

def test_database_operations():
    """Test 2: Test insert/retrieve operations on facts table"""
    print("\n🔍 Test 2: Database Operations")
    print("-" * 50)

    try:
        from db import AllieMemoryDB

        db = AllieMemoryDB()

        # Test insert
        test_fact = f"Test fact inserted at {datetime.now().isoformat()}"
        result = db.add_fact(
            keyword="diagnostic_test",
            fact=test_fact,
            source="diagnostic_script",
            confidence=0.9,
            category="diagnostic",
            status="true",
            confidence_score=85
        )

        if result.get("status") == "added":
            fact_id = result.get("fact_id")
            print(f"✓ Successfully inserted test fact with ID: {fact_id}")
        else:
            print(f"✗ Failed to insert fact: {result}")
            return False, result

        # Test retrieve
        retrieved = db.get_fact("diagnostic_test")
        if retrieved and retrieved.get("fact") == test_fact:
            print("✓ Successfully retrieved test fact")
            print(f"  - Status: {retrieved.get('status')}")
            print(f"  - Confidence: {retrieved.get('confidence')}")
            print(f"  - Confidence Score: {retrieved.get('confidence_score')}")
            print(f"  - Category: {retrieved.get('category')}")
            print(f"  - Source: {retrieved.get('source')}")
        else:
            print(f"✗ Failed to retrieve fact: {retrieved}")
            return False, retrieved

        # Test update
        updated_fact = test_fact + " (updated)"
        update_result = db.update_fact("diagnostic_test", updated_fact, "diagnostic_script", confidence=0.95)
        if update_result.get("status") == "updated":
            print("✓ Successfully updated test fact")
        else:
            print(f"✗ Failed to update fact: {update_result}")
            return False, update_result

        # Test delete
        delete_result = db.delete_fact("diagnostic_test")
        if delete_result.get("status") == "deleted":
            print("✓ Successfully deleted test fact")
        else:
            print(f"✗ Failed to delete fact: {delete_result}")
            return False, delete_result

        db.close()
        return True, {"fact_id": fact_id, "retrieved": retrieved}

    except Exception as e:
        print(f"✗ Database operations failed: {e}")
        return False, str(e)

def test_hybrid_memory():
    """Test 3: Test hybrid memory functionality"""
    print("\n🔍 Test 3: Hybrid Memory Functionality")
    print("-" * 50)

    try:
        from hybrid import HybridMemory

        hybrid = HybridMemory()

        # Test add fact
        test_fact = f"Hybrid test fact {datetime.now().isoformat()}"
        add_result = hybrid.add_fact(test_fact, category="diagnostic", confidence=0.8, source="diagnostic")
        if add_result.get("status") == "stored":
            print("✓ Successfully added fact to hybrid memory")
        else:
            print(f"✗ Failed to add fact to hybrid memory: {add_result}")
            return False, add_result

        # Test search
        search_results = hybrid.search("hybrid test fact", limit=5)
        if search_results and len(search_results) > 0:
            print(f"✓ Successfully searched hybrid memory, found {len(search_results)} results")
            for result in search_results[:2]:  # Show first 2
                print(f"  - {result.get('fact', '')[:50]}...")
        else:
            print("✗ Search returned no results")
            return False, search_results

        # Test timeline
        timeline = hybrid.get_timeline()
        print(f"✓ Retrieved timeline with {len(timeline)} entries")

        # Test statistics
        stats = hybrid.get_statistics()
        print(f"✓ Current hybrid memory statistics: {stats}")

        return True, {"stats": stats, "search_results": len(search_results)}

    except Exception as e:
        print(f"✗ Hybrid memory test failed: {e}")
        return False, str(e)

def check_old_memory_references():
    """Test 4: Check for old memory references"""
    print("\n🔍 Test 4: Old Memory References Detection")
    print("-" * 50)

    old_refs = []

    # Check for old memory directory
    old_memory_dir = project_root / "backend" / "memory"
    if old_memory_dir.exists():
        print(f"⚠️  Old memory directory found: {old_memory_dir}")
        old_refs.append(f"Old memory directory: {old_memory_dir}")

        # List contents
        contents = list(old_memory_dir.glob("*"))
        print(f"  Contents: {[f.name for f in contents]}")

    # Search for imports from old memory module
    search_dirs = [project_root / "backend", project_root / "advanced-memory", project_root / "tests"]

    for search_dir in search_dirs:
        if search_dir.exists():
            for py_file in search_dir.glob("**/*.py"):
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # Check for old imports
                    if 'from memory.' in content or 'from backend.memory.' in content:
                        old_refs.append(f"Old import in {py_file}: {content.count('from memory.') + content.count('from backend.memory.')} references")

                    # Check for old API calls
                    if '/api/memory' in content:
                        old_refs.append(f"Old API endpoint in {py_file}: /api/memory references")

                except Exception as e:
                    print(f"Warning: Could not read {py_file}: {e}")

    if old_refs:
        print("⚠️  Found old memory references:")
        for ref in old_refs:
            print(f"  - {ref}")
        return False, old_refs
    else:
        print("✓ No old memory references found")
        return True, []

def query_stale_facts():
    """Test 5: Query for stale/unused facts"""
    print("\n🔍 Test 5: Stale Facts Detection")
    print("-" * 50)

    try:
        from db import AllieMemoryDB

        db = AllieMemoryDB()

        # Query for not_verified facts
        not_verified = db.get_all_facts(status_filter="not_verified", limit=100)
        print(f"✓ Found {len(not_verified)} facts with status 'not_verified'")

        # Query for low confidence facts
        low_confidence = []
        all_facts = db.get_all_facts(limit=1000)  # Get more to check confidence
        for fact in all_facts:
            if fact.get('confidence_score', 50) < 30:
                low_confidence.append(fact)

        print(f"✓ Found {len(low_confidence)} facts with confidence_score < 30")

        # Show sample of stale facts
        if not_verified:
            print("Sample not_verified facts:")
            for fact in not_verified[:3]:
                print(f"  - ID {fact['id']}: {fact['fact'][:50]}... (confidence: {fact.get('confidence_score', 'N/A')})")

        if low_confidence:
            print("Sample low confidence facts:")
            for fact in low_confidence[:3]:
                print(f"  - ID {fact['id']}: {fact['fact'][:50]}... (confidence: {fact.get('confidence_score', 'N/A')})")

        db.close()

        return True, {
            "not_verified_count": len(not_verified),
            "low_confidence_count": len(low_confidence),
            "total_facts": len(all_facts)
        }

    except Exception as e:
        print(f"✗ Failed to query stale facts: {e}")
        return False, str(e)

def analyze_hybrid_memory_role():
    """Test 6: Analyze hybrid memory's role and reconciliation logic"""
    print("\n🔍 Test 6: Hybrid Memory Analysis")
    print("-" * 50)

    try:
        # Read hybrid.py file
        hybrid_file = project_root / "advanced-memory" / "hybrid.py"
        if not hybrid_file.exists():
            print("✗ hybrid.py not found in advanced-memory")
            return False, "hybrid.py not found"

        with open(hybrid_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Analyze key aspects
        analysis = {}

        # Check if it uses advanced memory as backend
        if 'AllieMemoryDB' in content:
            analysis["uses_advanced_memory"] = True
            print("✓ Hybrid memory uses AllieMemoryDB as backend")
        else:
            analysis["uses_advanced_memory"] = False
            print("⚠️  Hybrid memory does not appear to use AllieMemoryDB")

        # Check for external source integration
        external_sources = []
        if 'duckduckgo' in content.lower():
            external_sources.append('DuckDuckGo')
        if 'wikidata' in content.lower():
            external_sources.append('Wikidata')
        if 'dbpedia' in content.lower():
            external_sources.append('DBpedia')

        if external_sources:
            analysis["external_sources"] = external_sources
            print(f"✓ Hybrid memory integrates with external sources: {', '.join(external_sources)}")
        else:
            analysis["external_sources"] = []
            print("⚠️  No external source integration found in hybrid memory")

        # Check for reconciliation logic
        if 'reconcile' in content.lower() or 'conflict' in content.lower():
            analysis["has_reconciliation"] = True
            print("✓ Hybrid memory has reconciliation/conflict resolution logic")
        else:
            analysis["has_reconciliation"] = False
            print("⚠️  No reconciliation logic found in hybrid memory")

        # Check for old memory references
        if 'from memory.' in content or 'memory.' in content:
            analysis["old_memory_refs"] = True
            print("⚠️  Hybrid memory contains references to old memory module")
        else:
            analysis["old_memory_refs"] = False
            print("✓ Hybrid memory does not reference old memory module")

        return True, analysis

    except Exception as e:
        print(f"✗ Failed to analyze hybrid memory: {e}")
        return False, str(e)

def generate_report(results):
    """Generate comprehensive diagnostic report"""
    print("\n" + "="*60)
    print("📊 ADVANCED MEMORY SYSTEM DIAGNOSTIC REPORT")
    print("="*60)

    report = {
        "timestamp": datetime.now().isoformat(),
        "tests": results,
        "recommendations": []
    }

    # Overall health assessment
    passed_tests = sum(1 for result in results.values() if result.get("passed", False))
    total_tests = len(results)

    print(f"Overall Health: {passed_tests}/{total_tests} tests passed")

    # Advanced Memory Status
    if results.get("module_test", {}).get("passed"):
        print("✅ Advanced Memory: Fully Functional")
        module_data = results["module_test"].get("data", {})
        if isinstance(module_data, dict):
            db_stats = module_data.get("db_stats", {})
            print(f"   - Total Facts: {db_stats.get('total_facts', 'Unknown')}")
            print(f"   - Active Facts: {db_stats.get('active_facts', 'Unknown')}")
        else:
            print("   - Could not retrieve database statistics")
    else:
        print("❌ Advanced Memory: Issues Detected")
        report["recommendations"].append("Fix advanced memory module loading issues")

    # Database Operations
    if results.get("db_test", {}).get("passed"):
        print("✅ Database Operations: Working")
    else:
        print("❌ Database Operations: Failed")
        report["recommendations"].append("Fix database insert/retrieve/update/delete operations")

    # Hybrid Memory
    if results.get("hybrid_test", {}).get("passed"):
        print("✅ Hybrid Memory: Functional")
        hybrid_analysis = results.get("hybrid_analysis", {}).get("data", {})
        if isinstance(hybrid_analysis, dict):
            if hybrid_analysis.get("uses_advanced_memory"):
                print("   - Uses advanced memory as backend: ✅")
            if hybrid_analysis.get("external_sources"):
                print(f"   - External sources: {', '.join(hybrid_analysis['external_sources'])}")
            if hybrid_analysis.get("has_reconciliation"):
                print("   - Has reconciliation logic: ✅")
    else:
        print("❌ Hybrid Memory: Issues Detected")

    # Old Memory References
    old_refs = results.get("old_refs", {}).get("data", [])
    if old_refs:
        print(f"⚠️  Old Memory References: {len(old_refs)} found")
        for ref in old_refs[:3]:  # Show first 3
            print(f"   - {ref}")
        report["recommendations"].append("Remove old memory references and migrate to advanced_memory")
    else:
        print("✅ Old Memory References: None found")

    # Stale Facts
    stale_data = results.get("stale_facts", {}).get("data", {})
    not_verified = stale_data.get("not_verified_count", 0)
    low_confidence = stale_data.get("low_confidence_count", 0)
    total = stale_data.get("total_facts", 0)

    if not_verified > 0 or low_confidence > 0:
        print(f"⚠️  Stale Facts Detected: {not_verified} not_verified, {low_confidence} low confidence")
        print(f"   Total facts in system: {total}")
        report["recommendations"].append(f"Review and clean up {not_verified + low_confidence} stale facts")
    else:
        print("✅ Stale Facts: None detected")

    # Recommendations
    print("\n📋 RECOMMENDATIONS:")
    if report["recommendations"]:
        for i, rec in enumerate(report["recommendations"], 1):
            print(f"{i}. {rec}")
    else:
        print("✅ No issues found - system is healthy!")

    # Save report to file
    report_file = project_root / "memory_diagnostic_report.json"
    
    # Convert datetime objects to strings for JSON serialization
    def convert_datetimes(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {k: convert_datetimes(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_datetimes(item) for item in obj]
        else:
            return obj
    
    serializable_report = convert_datetimes(report)
    
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_report, f, indent=2, ensure_ascii=False)

    print(f"\n📄 Full report saved to: {report_file}")

    return report

def main():
    """Run all diagnostic tests"""
    print("🚀 Starting Advanced Memory System Diagnostics")
    print(f"Time: {datetime.now().isoformat()}")
    print(f"Project: {project_root}")
    print()

    results = {}

    # Test 1: Module loading
    passed, data = test_advanced_memory_module()
    results["module_test"] = {"passed": passed, "data": data}

    # Test 2: Database operations
    passed, data = test_database_operations()
    results["db_test"] = {"passed": passed, "data": data}

    # Test 3: Hybrid memory
    passed, data = test_hybrid_memory()
    results["hybrid_test"] = {"passed": passed, "data": data}

    # Test 4: Old memory references
    passed, data = check_old_memory_references()
    results["old_refs"] = {"passed": passed, "data": data}

    # Test 5: Stale facts
    passed, data = query_stale_facts()
    results["stale_facts"] = {"passed": passed, "data": data}

    # Test 6: Hybrid memory analysis
    passed, data = analyze_hybrid_memory_role()
    results["hybrid_analysis"] = {"passed": passed, "data": data}

    # Generate report
    report = generate_report(results)

    print("\n🏁 Diagnostics Complete")
    return report

if __name__ == "__main__":
    main()