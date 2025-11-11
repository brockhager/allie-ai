# Knowledge Base Migrations

This directory contains SQL migration files for the Knowledge Base system.

## Migration Files

### 001: Create knowledge_base table
- **Up**: `001_create_knowledge_base.sql`
- **Down**: `001_drop_knowledge_base.sql`

Creates the main `knowledge_base` table with columns for keyword, fact, source, status, confidence_score, provenance, and timestamps.

### 002: Alter learning_log for KB audit
- **Up**: `002_alter_learning_log.sql`
- **Down**: `002_drop_learning_log_columns.sql`

Adds `fact_id`, `reviewer`, and `reason` columns to `learning_log` table to support KB operation auditing.

## Running Migrations

### Apply All Migrations (Up)

```powershell
# Windows PowerShell
cd C:\Users\brock\allieai\allie-ai

Get-Content migrations\sql\001_create_knowledge_base.sql | mysql -u root -p"YOUR_PASSWORD" allie_memory
Get-Content migrations\sql\002_alter_learning_log.sql | mysql -u root -p"YOUR_PASSWORD" allie_memory
```

```bash
# Linux/Mac
cd /path/to/allie-ai

mysql -u root -p allie_memory < migrations/sql/001_create_knowledge_base.sql
mysql -u root -p allie_memory < migrations/sql/002_alter_learning_log.sql
```

### Rollback Migrations (Down)

```powershell
# Windows PowerShell
Get-Content migrations\sql\002_drop_learning_log_columns.sql | mysql -u root -p"YOUR_PASSWORD" allie_memory
Get-Content migrations\sql\001_drop_knowledge_base.sql | mysql -u root -p"YOUR_PASSWORD" allie_memory
```

```bash
# Linux/Mac
mysql -u root -p allie_memory < migrations/sql/002_drop_learning_log_columns.sql
mysql -u root -p allie_memory < migrations/sql/001_drop_knowledge_base.sql
```

## Verification

Check that tables were created/modified:

```sql
-- Check knowledge_base table
DESCRIBE knowledge_base;

-- Check learning_log columns
DESCRIBE learning_log;

-- Count KB facts
SELECT COUNT(*) FROM knowledge_base;
```

## Notes

- Migrations should be run in numerical order
- Always backup your database before running migrations
- Test migrations on a development database first
- The learning_log table must exist before running migration 002
