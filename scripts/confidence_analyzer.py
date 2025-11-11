#!/usr/bin/env python3
"""Confidence Score Distribution Analyzer

Analyzes the distribution of confidence scores in KB entries to understand
if thresholds are properly tuned and identify patterns in fact quality.
"""

import mysql.connector
import json
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import statistics

def get_db_connection():
    """Connect to the Allie memory database."""
    # Load config from mysql.json if it exists
    config_file = Path(__file__).parent.parent / "config" / "mysql.json"
    config = {
        "host": "localhost",
        "user": "root",
        "password": "",
        "database": "allie_memory",
        "port": 3306
    }

    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                config.update(json.load(f))
        except Exception as e:
            print(f"Warning: Could not load MySQL config: {e}")

    return mysql.connector.connect(**config)

def get_kb_confidence_data(limit=1000):
    """Get confidence scores and related data from KB."""
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    query = """
    SELECT id, keyword, fact, source, confidence_score, created_at, updated_at
    FROM knowledge_base
    ORDER BY created_at DESC
    LIMIT %s
    """
    cursor.execute(query, (limit,))
    entries = cursor.fetchall()

    cursor.close()
    conn.close()
    return entries

def analyze_confidence_distribution(entries):
    """Analyze confidence score distribution and patterns."""
    if not entries:
        return {"error": "No KB entries found"}

    confidence_scores = [entry['confidence_score'] for entry in entries if entry['confidence_score'] is not None]

    if not confidence_scores:
        return {"error": "No confidence scores found"}

    # Basic statistics
    stats = {
        "count": len(confidence_scores),
        "mean": statistics.mean(confidence_scores),
        "median": statistics.median(confidence_scores),
        "min": min(confidence_scores),
        "max": max(confidence_scores),
        "stdev": statistics.stdev(confidence_scores) if len(confidence_scores) > 1 else 0
    }

    # Distribution buckets
    buckets = {
        "very_low": len([c for c in confidence_scores if c < 50]),
        "low": len([c for c in confidence_scores if 50 <= c < 70]),
        "medium": len([c for c in confidence_scores if 70 <= c < 85]),
        "high": len([c for c in confidence_scores if 85 <= c < 95]),
        "very_high": len([c for c in confidence_scores if c >= 95])
    }

    # Source analysis
    source_confidence = defaultdict(list)
    for entry in entries:
        if entry['confidence_score'] is not None:
            source_confidence[entry['source']].append(entry['confidence_score'])

    source_stats = {}
    for source, scores in source_confidence.items():
        source_stats[source] = {
            "count": len(scores),
            "avg_confidence": statistics.mean(scores),
            "min_confidence": min(scores),
            "max_confidence": max(scores)
        }

    # Threshold analysis
    current_threshold = 75  # Based on our worker threshold
    above_threshold = len([c for c in confidence_scores if c >= current_threshold])
    below_threshold = len([c for c in confidence_scores if c < current_threshold])

    return {
        "statistics": stats,
        "distribution": buckets,
        "source_analysis": source_stats,
        "threshold_analysis": {
            "current_threshold": current_threshold,
            "above_threshold": above_threshold,
            "below_threshold": below_threshold,
            "promotion_rate": above_threshold / len(confidence_scores) if confidence_scores else 0
        }
    }

def plot_confidence_distribution(entries):
    """Create confidence distribution visualizations."""
    if not entries:
        print("No data to plot")
        return

    confidence_scores = [entry['confidence_score'] for entry in entries if entry['confidence_score'] is not None]

    if not confidence_scores:
        print("No confidence scores to plot")
        return

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Histogram
    ax1.hist(confidence_scores, bins=20, edgecolor='black', alpha=0.7)
    ax1.set_title('Confidence Score Distribution')
    ax1.set_xlabel('Confidence Score')
    ax1.set_ylabel('Frequency')
    ax1.grid(True, alpha=0.3)

    # Box plot
    ax2.boxplot(confidence_scores, vert=False)
    ax2.set_title('Confidence Score Box Plot')
    ax2.set_xlabel('Confidence Score')
    ax2.grid(True, alpha=0.3)

    # Cumulative distribution
    sorted_scores = sorted(confidence_scores)
    y_vals = np.arange(len(sorted_scores)) / float(len(sorted_scores) - 1)
    ax3.plot(sorted_scores, y_vals, 'b-', linewidth=2)
    ax3.set_title('Cumulative Distribution Function')
    ax3.set_xlabel('Confidence Score')
    ax3.set_ylabel('Cumulative Probability')
    ax3.grid(True, alpha=0.3)

    # Source comparison
    sources = {}
    for entry in entries:
        if entry['confidence_score'] is not None:
            source = entry['source']
            if source not in sources:
                sources[source] = []
            sources[source].append(entry['confidence_score'])

    if sources:
        source_names = list(sources.keys())
        source_data = [sources[name] for name in source_names]
        ax4.boxplot(source_data, labels=source_names)
        ax4.set_title('Confidence by Source')
        ax4.set_ylabel('Confidence Score')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    plot_file = Path(__file__).parent / "confidence_distribution.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"Confidence distribution plot saved to: {plot_file}")

    return plot_file

def generate_confidence_report():
    """Generate comprehensive confidence analysis report."""
    print("🎯 CONFIDENCE SCORE DISTRIBUTION ANALYSIS")
    print("=" * 60)

    entries = get_kb_confidence_data()
    analysis = analyze_confidence_distribution(entries)

    if "error" in analysis:
        print(f"❌ {analysis['error']}")
        return

    stats = analysis['statistics']
    dist = analysis['distribution']
    sources = analysis['source_analysis']
    threshold = analysis['threshold_analysis']

    print(f"📊 Total entries analyzed: {stats['count']}")
    print()

    print("📈 Basic Statistics:")
    print(".2f")
    print(".2f")
    print(f"   Range: {stats['min']} - {stats['max']}")
    print(".2f")
    print()

    print("📋 Distribution Buckets:")
    print(f"   Very Low (< 50): {dist['very_low']} entries")
    print(f"   Low (50-69): {dist['low']} entries")
    print(f"   Medium (70-84): {dist['medium']} entries")
    print(f"   High (85-94): {dist['high']} entries")
    print(f"   Very High (≥ 95): {dist['very_high']} entries")
    print()

    print("🎯 Threshold Analysis:")
    print(f"   Current threshold: {threshold['current_threshold']}")
    print(f"   Above threshold: {threshold['above_threshold']} entries")
    print(f"   Below threshold: {threshold['below_threshold']} entries")
    print(".1%")
    print()

    print("📋 Source Performance:")
    for source, src_stats in sources.items():
        print(f"   {source}:")
        print(".2f")
        print(f"      Range: {src_stats['min_confidence']} - {src_stats['max_confidence']}")
    print()

    # Recommendations
    print("💡 RECOMMENDATIONS:")

    if stats['mean'] < 70:
        print("   ⚠️  Average confidence is low - review fact extraction patterns")
    elif stats['mean'] > 90:
        print("   ⚠️  Very high confidence - may be too restrictive")

    if threshold['promotion_rate'] < 0.3:
        print("   ⚠️  Low promotion rate - consider lowering threshold")
    elif threshold['promotion_rate'] > 0.8:
        print("   ⚠️  High promotion rate - consider raising threshold for quality")

    if dist['very_low'] > stats['count'] * 0.1:
        print("   ⚠️  Significant low-confidence entries - review pattern matching")

    # Create visualization
    plot_file = plot_confidence_distribution(entries)
    if plot_file:
        print(f"   📈 Distribution plots: {plot_file}")

    print("\n" + "=" * 60)

if __name__ == "__main__":
    generate_confidence_report()