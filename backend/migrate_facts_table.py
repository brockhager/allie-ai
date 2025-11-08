"""
Migrate facts table to add missing columns
"""

import mysql.connector
import json
from pathlib import Path

def load_config():
    """Load MySQL config"""
    config_path = Path(__file__).parent.parent / "config" / "mysql.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            return json.load(f)
    return {
        "host": "localhost",
        "user": "root",
        "password": "",
        "database": "allie_memory",
        "port": 3306
    }

def migrate():
    """Add missing columns to facts table"""
    config = load_config()
    
    print("Connecting to MySQL...")
    conn = mysql.connector.connect(**config)
    cursor = conn.cursor()
    
    # Check current structure
    print("\nChecking current table structure...")
    cursor.execute("DESCRIBE facts")
    columns = {row[0]: row[1] for row in cursor.fetchall()}
    print(f"Current columns: {list(columns.keys())}")
    
    # Add missing columns
    migrations = []
    
    if 'category' not in columns:
        migrations.append("ALTER TABLE facts ADD COLUMN category VARCHAR(100)")
        print("  - Will add 'category' column")
    
    if 'confidence' not in columns:
        migrations.append("ALTER TABLE facts ADD COLUMN confidence FLOAT DEFAULT 0.8")
        print("  - Will add 'confidence' column")
    
    if 'metadata' not in columns:
        migrations.append("ALTER TABLE facts ADD COLUMN metadata JSON")
        print("  - Will add 'metadata' column")
    
    if not migrations:
        print("\n✓ Table already has all required columns!")
        cursor.close()
        conn.close()
        return
    
    # Execute migrations
    print(f"\nExecuting {len(migrations)} migration(s)...")
    for migration in migrations:
        print(f"  Running: {migration}")
        cursor.execute(migration)
    
    conn.commit()
    print("\n✓ Migration completed successfully!")
    
    # Show final structure
    cursor.execute("DESCRIBE facts")
    print("\nFinal table structure:")
    for row in cursor.fetchall():
        print(f"  {row[0]}: {row[1]} {row[2]} {row[3]}")
    
    cursor.close()
    conn.close()

if __name__ == "__main__":
    try:
        migrate()
    except Exception as e:
        print(f"\n✗ Migration failed: {e}")
