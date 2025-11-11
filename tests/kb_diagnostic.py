#!/usr/bin/env python3
"""
KB Diagnostic Report - Comprehensive Investigation
Objective: Investigate why KB only has 2 entries and isn't auto-adding new ones
"""

import sys
import mysql.connector
from mysql.connector import Error
from datetime import datetime
import json

def connect_db():
    """Connect to MySQL database"""
    try:
        connection = mysql.connector.connect(
            host='localhost',
            database='allie_memory',
            user='allie',
            password='StrongPassword123!'
        )
        return connection
    except Error as e:
        print(f"❌ Database connection failed: {e}")
        return None

def query_kb_table():
    """Query KB table directly"""
    print("\n=== 1. DATABASE VERIFICATION ===")

    conn = connect_db()
    if not conn:
        return

    try:
        cursor = conn.cursor(dictionary=True)

        # Count total entries
        cursor.execute("SELECT COUNT(*) as count FROM knowledge_base")
        count_result = cursor.fetchone()
        total_count = count_result['count']
        print(f"📊 Total KB entries: {total_count}")

        # Get all entries
        cursor.execute("SELECT * FROM knowledge_base ORDER BY updated_at DESC LIMIT 10")
        rows = cursor.fetchall()

        print(f"\n📋 KB Entries (showing {len(rows)}):")
        for row in rows:
            print(f"  ID {row['id']}: {row['keyword']} - {row['fact'][:60]}...")
            print(f"    Status: {row['status']} | Confidence: {row['confidence_score']} | Source: {row['source']}")
            print(f"    Updated: {row['updated_at']}")
            print()

        # Check schema
        cursor.execute("DESCRIBE knowledge_base")
        schema = cursor.fetchall()
        print("🏗️  KB Table Schema:")
        for col in schema:
            nullable = "NULL" if col['Null'] == 'YES' else "NOT NULL"
            default = f" DEFAULT {col['Default']}" if col['Default'] else ""
            print(f"  {col['Field']} {col['Type']} {nullable}{default}")

        cursor.close()

    except Error as e:
        print(f"❌ Query failed: {e}")
    finally:
        conn.close()

def check_learning_queue():
    """Check learning queue status"""
    print("\n=== 2. LEARNING QUEUE STATUS ===")

    conn = connect_db()
    if not conn:
        return

    try:
        cursor = conn.cursor(dictionary=True)

        # Count queue entries
        cursor.execute("SELECT COUNT(*) as count FROM learning_queue")
        queue_count = cursor.fetchone()['count']
        print(f"📋 Learning queue entries: {queue_count}")

        # Get pending items
        cursor.execute("SELECT * FROM learning_queue WHERE status = 'pending' ORDER BY created_at DESC LIMIT 5")
        pending = cursor.fetchall()

        print(f"\n⏳ Pending queue items (showing {len(pending)}):")
        for item in pending:
            print(f"  ID {item['id']}: {item['keyword']} - {item['fact'][:60]}...")
            print(f"    Status: {item['status']} | Confidence: {item['confidence']}")
            print(f"    Created: {item['created_at']}")
            print()

        cursor.close()

    except Error as e:
        print(f"❌ Queue query failed: {e}")
    finally:
        conn.close()

def check_learning_log():
    """Check learning log for KB operations"""
    print("\n=== 3. LEARNING LOG ANALYSIS ===")

    conn = connect_db()
    if not conn:
        return

    try:
        cursor = conn.cursor(dictionary=True)

        # Count log entries
        cursor.execute("SELECT COUNT(*) as count FROM learning_log")
        log_count = cursor.fetchone()['count']
        print(f"📝 Learning log entries: {log_count}")

        # Get recent KB-related operations
        cursor.execute("""
            SELECT * FROM learning_log
            WHERE change_type IN ('add', 'update')
            ORDER BY created_at DESC LIMIT 10
        """)
        logs = cursor.fetchall()

        print(f"\n🔄 Recent KB operations (showing {len(logs)}):")
        for log in logs:
            print(f"  {log['change_type'].upper()}: {log['keyword']} - {log['old_fact'][:40] if log['old_fact'] else 'NEW'}")
            print(f"    Fact ID: {log['fact_id']} | Confidence: {log['confidence']}")
            print(f"    Time: {log['created_at']}")
            print()

        cursor.close()

    except Error as e:
        print(f"❌ Log query failed: {e}")
    finally:
        conn.close()

def test_kb_insertion():
    """Test KB insertion directly"""
    print("\n=== 4. KB INSERTION TEST ===")

    # Import the DB module
    sys.path.insert(0, '.')
    import importlib.util
    spec = importlib.util.spec_from_file_location('db', 'advanced-memory/db.py')
    db_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(db_module)

    db = db_module.AllieMemoryDB()

    # Test insertion
    test_keyword = f"diagnostic_test_{int(datetime.now().timestamp())}"
    test_fact = "This is a diagnostic test fact to verify KB insertion works."

    print(f"🧪 Testing KB insertion: {test_keyword}")

    result = db.add_kb_fact(
        keyword=test_keyword,
        fact=test_fact,
        source="diagnostic_test",
        confidence_score=85
    )

    print(f"✅ Insertion result: {result}")

    # Verify it was added
    fact = db.get_kb_fact(test_keyword)
    if fact:
        print(f"✅ Verification: Fact found in KB (ID: {fact['id']})")
    else:
        print("❌ Verification failed: Fact not found in KB")

    return result

def check_worker_status():
    """Check if reconciliation worker is running"""
    print("\n=== 5. WORKER STATUS ===")

    import subprocess
    import platform

    try:
        if platform.system() == "Windows":
            # Check for Python processes running kb_worker.py
            result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq python.exe'], capture_output=True, text=True)
            if 'python.exe' in result.stdout:
                print("🤖 Python processes running (worker may be active)")
            else:
                print("❌ No Python processes found (worker likely not running)")
        else:
            # Unix-like systems
            result = subprocess.run(['pgrep', '-f', 'kb_worker'], capture_output=True, text=True)
            if result.returncode == 0:
                print("🤖 KB worker process found")
            else:
                print("❌ KB worker process not found")

    except Exception as e:
        print(f"❌ Worker check failed: {e}")

def main():
    """Run all diagnostics"""
    print("🔍 KB DIAGNOSTIC REPORT")
    print("=" * 50)
    print(f"Timestamp: {datetime.now()}")

    query_kb_table()
    check_learning_queue()
    check_learning_log()
    test_kb_insertion()
    check_worker_status()

    print("\n" + "=" * 50)
    print("📋 SUMMARY & RECOMMENDATIONS")
    print("=" * 50)

    # Re-query final count
    conn = connect_db()
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) as count FROM knowledge_base")
            final_count = cursor.fetchone()[0]
            cursor.close()
            conn.close()

            print(f"Final KB count: {final_count}")

            if final_count <= 2:
                print("❌ ISSUE: KB still has minimal entries - automatic ingestion not working")
                print("💡 RECOMMENDATIONS:")
                print("  1. Start reconciliation worker: python scripts/kb_worker.py")
                print("  2. Check feature flags in server config")
                print("  3. Verify learning pipeline is enabled")
                print("  4. Check server logs for ingestion errors")
            else:
                print("✅ KB growing - automatic ingestion appears functional")

        except Error as e:
            print(f"❌ Final count check failed: {e}")

if __name__ == "__main__":
    main()