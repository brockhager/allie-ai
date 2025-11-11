#!/usr/bin/env python3
"""KB Growth Rate Monitor

Tracks KB growth over time and analyzes trends to ensure automatic learning
is working effectively and thresholds are properly tuned.
"""

import mysql.connector
import json
from datetime import datetime, timedelta
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

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

def get_kb_growth_data(days=30):
    """Get KB entries with timestamps for growth analysis."""
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    query = """
    SELECT id, keyword, fact, source, confidence_score, created_at, updated_at
    FROM knowledge_base
    WHERE created_at >= DATE_SUB(NOW(), INTERVAL %s DAY)
    ORDER BY created_at ASC
    """
    cursor.execute(query, (days,))
    entries = cursor.fetchall()

    cursor.close()
    conn.close()
    return entries

def analyze_growth_rate(entries, days=30):
    """Analyze KB growth patterns and rates."""
    if not entries:
        return {"error": "No KB entries found in the specified period"}

    # Group by date
    daily_counts = defaultdict(int)
    confidence_scores = []
    sources = defaultdict(int)

    for entry in entries:
        date = entry['created_at'].date() if entry['created_at'] else datetime.now().date()
        daily_counts[date] += 1
        confidence_scores.append(entry['confidence_score'] or 0)
        sources[entry['source']] += 1

    # Calculate growth metrics
    total_entries = len(entries)
    avg_daily = total_entries / days if days > 0 else 0

    # Find peak growth days
    peak_day = max(daily_counts.items(), key=lambda x: x[1]) if daily_counts else (datetime.now().date(), 0)

    # Confidence analysis
    avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
    min_confidence = min(confidence_scores) if confidence_scores else 0
    max_confidence = max(confidence_scores) if confidence_scores else 0

    return {
        "total_entries": total_entries,
        "avg_daily_growth": avg_daily,
        "peak_growth_day": {"date": peak_day[0], "count": peak_day[1]},
        "confidence_stats": {
            "average": avg_confidence,
            "min": min_confidence,
            "max": max_confidence
        },
        "source_breakdown": dict(sources),
        "daily_breakdown": dict(daily_counts)
    }

def plot_growth_trend(entries, days=30):
    """Create a growth trend visualization."""
    if not entries:
        print("No data to plot")
        return

    # Prepare data for plotting
    dates = []
    cumulative = []
    daily = []

    daily_counts = defaultdict(int)
    for entry in entries:
        date = entry['created_at'].date() if entry['created_at'] else datetime.now().date()
        daily_counts[date] += 1

    # Sort dates
    sorted_dates = sorted(daily_counts.keys())
    running_total = 0

    for date in sorted_dates:
        dates.append(date)
        running_total += daily_counts[date]
        cumulative.append(running_total)
        daily.append(daily_counts[date])

    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Cumulative growth
    ax1.plot(dates, cumulative, 'b-', linewidth=2, marker='o')
    ax1.set_title(f'KB Growth Over Last {days} Days (Cumulative)')
    ax1.set_ylabel('Total KB Entries')
    ax1.grid(True, alpha=0.3)

    # Daily growth
    ax2.bar(dates, daily, color='green', alpha=0.7)
    ax2.set_title(f'Daily KB Additions (Last {days} Days)')
    ax2.set_ylabel('New Entries per Day')
    ax2.set_xlabel('Date')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    plot_file = Path(__file__).parent / "kb_growth_analysis.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"Growth analysis plot saved to: {plot_file}")

    return plot_file

def generate_growth_report(days=30):
    """Generate a comprehensive growth report."""
    print(f"🔍 KB GROWTH ANALYSIS REPORT (Last {days} days)")
    print("=" * 60)

    entries = get_kb_growth_data(days)
    analysis = analyze_growth_rate(entries, days)

    if "error" in analysis:
        print(f"❌ {analysis['error']}")
        return

    print(f"📊 Total new KB entries: {analysis['total_entries']}")
    print(".2f")
    print(f"📈 Peak growth day: {analysis['peak_growth_day']['date']} ({analysis['peak_growth_day']['count']} entries)")
    print()

    print("🎯 Confidence Score Analysis:")
    conf = analysis['confidence_stats']
    print(".2f")
    print(f"   Min: {conf['min']}, Max: {conf['max']}")
    print()

    print("📋 Source Breakdown:")
    for source, count in analysis['source_breakdown'].items():
        print(f"   {source}: {count} entries")
    print()

    print("📅 Recent Daily Activity:")
    daily = analysis['daily_breakdown']
    for date in sorted(daily.keys(), reverse=True)[:7]:  # Last 7 days
        print(f"   {date}: {daily[date]} new entries")
    print()

    # Recommendations
    print("💡 RECOMMENDATIONS:")
    avg_daily = analysis['avg_daily_growth']

    if avg_daily < 0.5:
        print("   ⚠️  Low growth rate - consider lowering confidence thresholds")
    elif avg_daily > 5:
        print("   ⚠️  High growth rate - review for quality vs quantity")
    else:
        print("   ✅ Growth rate looks healthy")

    if conf['average'] < 75:
        print("   ⚠️  Average confidence low - patterns may need tuning")
    elif conf['average'] > 95:
        print("   ⚠️  Very high confidence - may be too restrictive")

    # Create visualization
    plot_file = plot_growth_trend(entries, days)
    if plot_file:
        print(f"   📈 Growth trend plot: {plot_file}")

    print("\n" + "=" * 60)

if __name__ == "__main__":
    # Generate report for last 30 days
    generate_growth_report(30)