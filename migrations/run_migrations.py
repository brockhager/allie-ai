#!/usr/bin/env python3
"""
Database Migration Runner for Allie Memory System Upgrade

Usage:
    python run_migrations.py [up|down|status]

Commands:
    up: Apply all pending migrations
    down: Rollback last migration
    status: Show current migration status
"""

import os
import sys
import mysql.connector
from mysql.connector import Error
from pathlib import Path
import logging
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

class MigrationRunner:
    def __init__(self, host='localhost', database='allie_memory', user='allie', password='StrongPassword123!'):
        self.host = host
        self.database = database
        self.user = user
        self.password = password
        self.connection = None
        self.migrations_dir = Path(__file__).parent

    def connect(self):
        """Establish database connection"""
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                database=self.database,
                user=self.user,
                password=self.password,
                autocommit=False  # We'll manage transactions
            )
            logger.info(f"Connected to MySQL database: {self.database}")
        except Error as e:
            logger.error(f"Failed to connect to database: {e}")
            raise

    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")

    def get_applied_migrations(self):
        """Get list of applied migrations"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version VARCHAR(255) PRIMARY KEY,
                    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            self.connection.commit()

            cursor.execute("SELECT version FROM schema_migrations ORDER BY applied_at")
            applied = [row[0] for row in cursor.fetchall()]
            cursor.close()
            return applied
        except Error as e:
            logger.error(f"Error getting applied migrations: {e}")
            return []

    def mark_migration_applied(self, version):
        """Mark a migration as applied"""
        try:
            cursor = self.connection.cursor()
            cursor.execute(
                "INSERT INTO schema_migrations (version) VALUES (%s)",
                (version,)
            )
            self.connection.commit()
            cursor.close()
            logger.info(f"Marked migration {version} as applied")
        except Error as e:
            logger.error(f"Error marking migration as applied: {e}")
            raise

    def unmark_migration(self, version):
        """Remove migration from applied list"""
        try:
            cursor = self.connection.cursor()
            cursor.execute(
                "DELETE FROM schema_migrations WHERE version = %s",
                (version,)
            )
            self.connection.commit()
            cursor.close()
            logger.info(f"Unmarked migration {version}")
        except Error as e:
            logger.error(f"Error unmarking migration: {e}")
            raise

    def get_migration_files(self):
        """Get all migration files in order"""
        migration_files = []
        for file in self.migrations_dir.glob("*.sql"):
            if file.name.startswith("001_"):
                migration_files.append(file)
        return sorted(migration_files)

    def apply_migration(self, migration_file):
        """Apply a single migration"""
        version = migration_file.stem
        logger.info(f"Applying migration: {version}")

        with open(migration_file, 'r', encoding='utf-8') as f:
            sql_content = f.read()

        # Split on migration markers
        up_section = self.extract_section(sql_content, "MIGRATION UP", "MIGRATION DOWN")

        if not up_section.strip():
            logger.error(f"No UP section found in {migration_file}")
            return False

        try:
            cursor = self.connection.cursor()

            # Execute each statement
            statements = [stmt.strip() for stmt in up_section.split(';') if stmt.strip() and not stmt.strip().startswith('--')]
            for statement in statements:
                if statement:
                    logger.debug(f"Executing: {statement[:100]}...")
                    cursor.execute(statement)

            self.connection.commit()
            cursor.close()

            self.mark_migration_applied(version)
            logger.info(f"Successfully applied migration: {version}")
            return True

        except Error as e:
            self.connection.rollback()
            logger.error(f"Failed to apply migration {version}: {e}")
            return False

    def rollback_migration(self, migration_file):
        """Rollback a single migration"""
        version = migration_file.stem
        logger.info(f"Rolling back migration: {version}")

        with open(migration_file, 'r', encoding='utf-8') as f:
            sql_content = f.read()

        # Extract DOWN section
        down_section = self.extract_section(sql_content, "MIGRATION DOWN", "VERIFICATION")

        if not down_section.strip():
            logger.warning(f"No DOWN section found in {migration_file}, skipping rollback")
            return True

        try:
            cursor = self.connection.cursor()

            # Execute rollback statements
            statements = [stmt.strip() for stmt in down_section.split(';') if stmt.strip() and not stmt.strip().startswith('--')]
            for statement in statements:
                if statement:
                    logger.debug(f"Executing rollback: {statement[:100]}...")
                    cursor.execute(statement)

            self.connection.commit()
            cursor.close()

            self.unmark_migration(version)
            logger.info(f"Successfully rolled back migration: {version}")
            return True

        except Error as e:
            self.connection.rollback()
            logger.error(f"Failed to rollback migration {version}: {e}")
            return False

    def extract_section(self, content, start_marker, end_marker):
        """Extract a section from SQL content between markers"""
        lines = content.split('\n')
        in_section = False
        section_lines = []

        for line in lines:
            if start_marker in line:
                in_section = True
                continue
            elif end_marker in line and in_section:
                break

            if in_section:
                section_lines.append(line)

        return '\n'.join(section_lines)

    def run_up(self):
        """Apply all pending migrations"""
        applied = self.get_applied_migrations()
        migration_files = self.get_migration_files()

        for migration_file in migration_files:
            version = migration_file.stem
            if version not in applied:
                if not self.apply_migration(migration_file):
                    logger.error(f"Migration failed: {version}")
                    return False
            else:
                logger.info(f"Migration already applied: {version}")

        logger.info("All migrations applied successfully")
        return True

    def run_down(self):
        """Rollback last applied migration"""
        applied = self.get_applied_migrations()
        if not applied:
            logger.info("No migrations to rollback")
            return True

        # Get last applied migration
        last_version = applied[-1]
        migration_file = self.migrations_dir / f"{last_version}.sql"

        if migration_file.exists():
            return self.rollback_migration(migration_file)
        else:
            logger.error(f"Migration file not found: {migration_file}")
            return False

    def run_status(self):
        """Show migration status"""
        applied = self.get_applied_migrations()
        migration_files = self.get_migration_files()

        print("Migration Status:")
        print("=" * 50)

        for migration_file in migration_files:
            version = migration_file.stem
            status = "APPLIED" if version in applied else "PENDING"
            print(f"{version}: {status}")

        print(f"\nTotal migrations: {len(migration_files)}")
        print(f"Applied: {len(applied)}")
        print(f"Pending: {len(migration_files) - len(applied)}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_migrations.py [up|down|status]")
        sys.exit(1)

    command = sys.argv[1]

    runner = MigrationRunner()
    try:
        runner.connect()

        if command == "up":
            success = runner.run_up()
        elif command == "down":
            success = runner.run_down()
        elif command == "status":
            runner.run_status()
            success = True
        else:
            print(f"Unknown command: {command}")
            success = False

        sys.exit(0 if success else 1)

    except Exception as e:
        logger.error(f"Migration runner failed: {e}")
        sys.exit(1)
    finally:
        runner.close()

if __name__ == "__main__":
    main()