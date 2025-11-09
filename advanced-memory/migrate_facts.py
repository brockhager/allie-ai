#!/usr/bin/env python3
"""
Migration script to transfer facts from simple memory to advanced memory system

This will:
1. Read all facts from the simple memory system
2. Process them through the advanced learning pipeline
3. Preserve source URLs and metadata
4. Report migration statistics
"""

import sys
import os
import json

# Add parent and current directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)

from db import AllieMemoryDB
from learning_pipeline import LearningPipeline


def categorize_fact(keyword: str, fact: str) -> str:
    """
    Infer category from keyword and fact content
    
    Args:
        keyword: Fact keyword
        fact: Fact content
        
    Returns:
        Category string
    """
    keyword_lower = keyword.lower()
    fact_lower = fact.lower()
    
    # Science keywords
    if any(word in keyword_lower or word in fact_lower for word in 
           ['science', 'physics', 'chemistry', 'biology', 'atom', 'molecule', 
            'evolution', 'gravity', 'planet', 'star', 'space']):
        return 'science'
    
    # Technology keywords
    if any(word in keyword_lower or word in fact_lower for word in 
           ['technology', 'computer', 'software', 'programming', 'code', 
            'internet', 'web', 'app', 'digital', 'tech']):
        return 'technology'
    
    # Geography keywords
    if any(word in keyword_lower or word in fact_lower for word in 
           ['city', 'country', 'capital', 'continent', 'ocean', 'mountain',
            'river', 'geography', 'location', 'place']):
        return 'geography'
    
    # History keywords
    if any(word in keyword_lower or word in fact_lower for word in 
           ['history', 'historical', 'war', 'century', 'ancient', 'dynasty',
            'empire', 'revolution', 'founded', 'born', 'died']):
        return 'history'
    
    # Culture/Arts keywords
    if any(word in keyword_lower or word in fact_lower for word in 
           ['culture', 'art', 'music', 'literature', 'book', 'author',
            'artist', 'painting', 'sculpture', 'film', 'movie']):
        return 'cultural'
    
    # Personal facts
    if any(word in keyword_lower for word in 
           ['brock', 'user', 'favorite', 'likes', 'prefers']):
        return 'personal'
    
    # Default
    return 'general'


def infer_confidence(source: str) -> float:
    """
    Infer confidence based on source
    
    Args:
        source: Source URL or identifier
        
    Returns:
        Confidence score (0.0-1.0)
    """
    source_lower = source.lower()
    
    # High confidence sources
    if any(domain in source_lower for domain in 
           ['wikipedia', 'britannica', '.gov', '.edu', 'openlibrary']):
        return 0.9
    
    # Medium-high confidence
    if 'user' in source_lower or 'conversation' in source_lower:
        return 0.85
    
    # Medium confidence
    if any(domain in source_lower for domain in 
           ['.org', 'stackexchange', 'github']):
        return 0.8
    
    # Default confidence
    return 0.75


def migrate_facts(dry_run: bool = False, batch_size: int = 50):
    """
    Migrate all facts from hybrid memory JSON to advanced memory
    
    Args:
        dry_run: If True, don't actually insert facts
        batch_size: Number of facts to process at once
    """
    print("=" * 70)
    print("  ALLIE MEMORY MIGRATION: Hybrid JSON → Advanced MySQL")
    print("=" * 70)
    
    # Initialize advanced memory
    print("\n[1/5] Connecting to advanced memory database...")
    try:
        advanced_memory = AllieMemoryDB()
        pipeline = LearningPipeline(advanced_memory)
        print("  ✓ Connected to advanced memory")
    except Exception as e:
        print(f"  ✗ Failed to connect to advanced memory: {e}")
        return
    
    # Read from hybrid_memory.json
    print("\n[2/5] Reading facts from hybrid_memory.json...")
    try:
        json_path = os.path.join(parent_dir, 'data', 'hybrid_memory.json')
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        all_facts = data.get('facts', [])
        fact_count = data.get('fact_count', len(all_facts))
        
        print(f"  ✓ Found {fact_count} facts to migrate")
    except Exception as e:
        print(f"  ✗ Failed to read hybrid memory: {e}")
        import traceback
        traceback.print_exc()
        return
    
    if not all_facts:
        print("  ! No facts to migrate")
        return
    
    # Prepare facts for migration
    print("\n[3/5] Preparing facts for migration...")
    migration_facts = []
    
    for idx, fact_data in enumerate(all_facts):
        fact_text = fact_data.get('fact', '')
        source = fact_data.get('source', 'unknown')
        category = fact_data.get('category', 'general')
        confidence = fact_data.get('confidence', 0.8)
        
        if not fact_text or not fact_text.strip():
            continue
        
        # Extract keyword from fact (use first few words or create from fact)
        # For hybrid memory, facts don't have separate keywords, so we create them
        fact_lower = fact_text.lower()
        
        # Try to extract a meaningful keyword
        if ':' in fact_text[:100]:  # Pattern like "Name: description"
            keyword = fact_text.split(':')[0].strip()
        elif '=' in fact_text[:50]:  # Pattern like "property = value"
            keyword = fact_text.split('=')[0].strip()
        elif 'by' in fact_text[:100]:  # Pattern like "Book: by Author"
            keyword = fact_text.split('by')[0].split(':')[0].strip()
        else:
            # Use first significant words
            words = fact_text.split()[:5]
            keyword = ' '.join(words).strip('.,!?;:')
        
        # Ensure keyword is reasonable length
        if len(keyword) > 100:
            keyword = keyword[:100]
        
        # Generate unique keyword if needed
        keyword = f"{keyword}_{idx}" if not keyword else keyword
        
        migration_facts.append({
            'keyword': keyword,
            'fact': fact_text,
            'source': source,
            'confidence': confidence,
            'category': category
        })
    
    print(f"  ✓ Prepared {len(migration_facts)} facts")
    
    # Show preview
    print("\n  Preview (first 5 facts):")
    for fact in migration_facts[:5]:
        print(f"    - {fact['keyword']} [{fact['category']}] (confidence: {fact['confidence']})")
    
    if dry_run:
        print("\n  DRY RUN MODE - No facts will be migrated")
        print(f"\n  Would migrate {len(migration_facts)} facts")
        
        # Show category breakdown
        categories = {}
        for fact in migration_facts:
            cat = fact['category']
            categories[cat] = categories.get(cat, 0) + 1
        
        print("\n  Category breakdown:")
        for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            print(f"    - {cat}: {count}")
        
        return
    
    # Migrate facts in batches
    print(f"\n[4/5] Migrating facts (batch size: {batch_size})...")
    
    total = len(migration_facts)
    migrated = 0
    errors = 0
    
    for i in range(0, total, batch_size):
        batch = migration_facts[i:i+batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (total + batch_size - 1) // batch_size
        
        print(f"  Processing batch {batch_num}/{total_batches} ({len(batch)} facts)...", end=" ")
        
        try:
            result = pipeline.process_batch(batch, auto_resolve=True)
            migrated += result['added'] + result['updated']
            errors += result['rejected']
            print(f"✓ (added: {result['added']}, updated: {result['updated']}, rejected: {result['rejected']})")
        except Exception as e:
            print(f"✗ Error: {e}")
            errors += len(batch)
    
    # Get final statistics
    print("\n[5/5] Generating statistics...")
    stats = advanced_memory.get_statistics()
    
    print("\n" + "=" * 70)
    print("  MIGRATION COMPLETE")
    print("=" * 70)
    print(f"\n  Facts processed: {total}")
    print(f"  Successfully migrated: {migrated}")
    print(f"  Errors: {errors}")
    print(f"\n  Advanced Memory Statistics:")
    print(f"    Total facts: {stats['total_facts']}")
    print(f"    Average confidence: {stats['average_confidence']}")
    print(f"    Learning log entries: {stats['learning_log_entries']}")
    
    if stats.get('by_category'):
        print(f"\n  Facts by category:")
        for category, count in sorted(stats['by_category'].items(), key=lambda x: x[1], reverse=True):
            print(f"    - {category}: {count}")
    
    print("\n" + "=" * 70)
    
    # Close connection
    advanced_memory.close()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Migrate facts from simple to advanced memory')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Preview migration without actually migrating')
    parser.add_argument('--batch-size', type=int, default=50,
                       help='Number of facts to process per batch (default: 50)')
    
    args = parser.parse_args()
    
    try:
        migrate_facts(dry_run=args.dry_run, batch_size=args.batch_size)
    except KeyboardInterrupt:
        print("\n\n⚠ Migration interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
