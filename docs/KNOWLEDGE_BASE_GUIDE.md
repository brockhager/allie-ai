# Knowledge Base (KB) System

## Overview

The Knowledge Base (KB) system provides curated, verified facts that take precedence in Allie's retrieval pipeline. KB facts are stored in MySQL and integrated with the hybrid memory system.

## Features

- **Curated Facts**: Manually reviewed and verified information
- **Status Management**: Track facts as `true`, `false`, `pending`, or `needs_review`
- **Confidence Scoring**: 0-100 score based on source reliability and verification
- **Audit Logging**: All KB operations logged to `learning_log` table
- **API & UI**: RESTful endpoints and browser-based admin interface
- **Hybrid Integration**: KB facts take priority over external sources
- **Worker Support**: Automated reconciliation via `kb_worker.py`

## Database Schema

### `knowledge_base` Table

```sql
CREATE TABLE IF NOT EXISTS knowledge_base (
    id INT AUTO_INCREMENT PRIMARY KEY,
    keyword VARCHAR(255) NOT NULL UNIQUE,
    fact TEXT NOT NULL,
    source VARCHAR(255) NOT NULL,
    confidence_score INT DEFAULT 50,
    provenance TEXT,
    status ENUM('true', 'false', 'pending', 'needs_review') DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_keyword (keyword),
    INDEX idx_status (status),
    INDEX idx_confidence (confidence_score)
);
```

### Field Descriptions

- **keyword**: Primary lookup key (e.g., "Paris", "capital_of_France")
- **fact**: The verified fact text
- **source**: Origin of the fact (e.g., "wikipedia", "manual_entry")
- **confidence_score**: 0-100, higher = more confident
- **provenance**: Details about fact origin/verification
- **status**: 
  - `true`: Verified correct, use in responses
  - `false`: Verified incorrect, exclude from responses
  - `pending`: Awaiting review
  - `needs_review`: Requires human verification

## Migrations

### Running Migrations

```bash
# Create KB table
mysql -u root -p allie_memory < migrations/sql/001_create_knowledge_base.sql

# Rollback (if needed)
mysql -u root -p allie_memory < migrations/sql/001_drop_knowledge_base.sql
```

Migration files are in `migrations/sql/`:
- `001_create_knowledge_base.sql` - Creates the KB table
- `001_drop_knowledge_base.sql` - Removes the KB table

## API Endpoints

### Base URLs

Both endpoint patterns are supported:
- `/api/kb/*` (short form)
- `/api/knowledge-base/*` (verbose form)

### List KB Facts

```http
GET /api/knowledge-base?status=true&limit=50&offset=0
```

Query parameters:
- `status` (optional): Filter by status (`true`, `false`, `pending`, `needs_review`)
- `limit` (optional): Max results (default: 50)
- `offset` (optional): Pagination offset (default: 0)

Response:
```json
[
  {
    "id": 1,
    "keyword": "Paris",
    "fact": "Paris is the capital of France",
    "source": "wikipedia",
    "confidence_score": 98,
    "status": "true",
    "created_at": "2025-11-10T10:30:00",
    "updated_at": "2025-11-10T10:30:00"
  }
]
```

### Get Single KB Fact

```http
GET /api/knowledge-base/{id}
```

Returns single fact by ID or 404 if not found.

### Create KB Fact

```http
POST /api/knowledge-base
Content-Type: application/json

{
  "keyword": "Berlin",
  "fact": "Berlin is the capital of Germany",
  "source": "manual_entry",
  "confidence_score": 95,
  "provenance": "verified by admin",
  "status": "true"
}
```

Required fields: `keyword`, `fact`, `source`

Optional: `confidence_score` (default: 50), `provenance`, `status` (default: "pending")

### Update KB Fact

```http
PATCH /api/knowledge-base/{id}
Content-Type: application/json
X-User-Role: admin

{
  "fact": "Updated fact text",
  "status": "true",
  "confidence_score": 98
}
```

Requires `X-User-Role: admin` header. Updates only provided fields.

### Delete KB Fact

```http
DELETE /api/knowledge-base/{id}
X-User-Role: admin
```

Requires `X-User-Role: admin` header. Permanently removes the fact.

## Admin UI

### Accessing the UI

1. Start the server:
   ```bash
   cd backend
   uvicorn server:app --reload --host 0.0.0.0 --port 8000
   ```

2. Open browser to: `http://localhost:8000/ui`

3. Click **"Knowledge Base"** button in the header

The KB UI provides:
- View all KB facts in a table
- Filter by status
- Add new facts
- Edit existing facts
- Delete facts
- Real-time confidence scoring

### UI Features

- **Status Color Coding**:
  - 🟢 Green: `true` (verified)
  - 🔴 Red: `false` (incorrect)
  - 🟡 Yellow: `pending` (needs review)
  - 🟠 Orange: `needs_review` (flagged)

- **Inline Editing**: Click any fact to edit
- **Role-Based Access**: Admin operations require proper role header
- **Audit Trail**: All changes logged to `learning_log`

## Python API (AllieMemoryDB)

### Adding Facts

```python
from db import AllieMemoryDB

db = AllieMemoryDB()
db.add_kb_fact(
    keyword="Tokyo",
    fact="Tokyo is the capital of Japan",
    source="verified_source",
    confidence_score=97,
    provenance="Official government source",
    status="true"
)
```

### Retrieving Facts

```python
# Get single fact
fact = db.get_kb_fact("Tokyo")

# Get all facts (with filters)
all_facts = db.get_all_kb_facts(status="true", limit=100)
```

### Updating Facts

```python
db.update_kb_fact(
    keyword="Tokyo",
    new_fact="Tokyo is the capital and largest city of Japan",
    status="true",
    confidence_score=99,
    reviewer="admin_user",
    reason="Added more detail"
)
```

### Deleting Facts

```python
db.delete_kb_fact(
    keyword="Tokyo",
    reviewer="admin_user",
    reason="Duplicate entry"
)
```

## Hybrid Memory Integration

The hybrid memory system checks KB first before external sources:

```python
from hybrid import HybridMemory

hybrid = HybridMemory()
results = hybrid.search("Paris")

# If KB has a 'true' entry for "Paris", it returns immediately
# If KB has a 'false' entry, results are excluded with warning
# Otherwise, falls back to regular search
```

### KB Priority Rules

1. **KB Status = `true`**: Return immediately with high confidence
2. **KB Status = `false`**: Exclude from results, add warning
3. **KB Status = `pending` or `needs_review`**: Falls through to regular search
4. **No KB entry**: Regular multi-source search

## Worker Script

The KB worker (`scripts/kb_worker.py`) processes the learning queue:

### Running the Worker

```bash
cd scripts
python kb_worker.py
```

### What It Does

1. Polls `learning_queue` for unprocessed entries
2. Validates suggested facts
3. Promotes high-confidence facts to KB (status: `true`)
4. Marks low-confidence facts for review (status: `needs_review`)
5. Logs all actions to `learning_log`

### Worker Logic

```python
# High confidence (≥ 0.85) → auto-promote to KB as 'true'
# Medium confidence (0.5-0.85) → mark as 'needs_review'
# Low confidence (< 0.5) → mark as 'pending'
```

### Scheduling the Worker

**Windows (Task Scheduler)**:
```powershell
# Run every hour
schtasks /create /tn "AllieKBWorker" /tr "python C:\path\to\scripts\kb_worker.py" /sc hourly
```

**Linux (cron)**:
```bash
# Run every hour
0 * * * * cd /path/to/allie-ai && python scripts/kb_worker.py
```

## Testing

### Running Unit Tests

```bash
cd advanced-memory
python -m pytest test_kb_integration.py -v
```

Or run directly:
```bash
python test_kb_integration.py
```

### Test Coverage

The test suite (`test_kb_integration.py`) covers:

- ✅ KB CRUD operations
- ✅ Confidence scoring
- ✅ Status filtering
- ✅ Hybrid memory KB preference
- ✅ Audit logging
- ✅ Learning queue integration

### Manual Smoke Test

1. Start server:
   ```bash
   cd backend
   uvicorn server:app --host 0.0.0.0 --port 8000
   ```

2. Create a KB fact:
   ```bash
   curl -X POST http://localhost:8000/api/kb \
     -H "Content-Type: application/json" \
     -d '{"keyword":"smoke_test","fact":"Test fact","source":"test","confidence_score":95,"status":"true"}'
   ```

3. List KB facts:
   ```bash
   curl http://localhost:8000/api/kb
   ```

4. Test hybrid preference:
   ```python
   from advanced-memory.hybrid import HybridMemory
   hybrid = HybridMemory()
   results = hybrid.search("smoke_test")
   print(results)  # Should return KB fact with category='knowledge_base'
   ```

## Confidence Scoring

Confidence scores help prioritize facts:

### Score Ranges

- **90-100**: Highly verified (e.g., official sources)
- **70-89**: Reliable (e.g., Wikipedia, established sources)
- **50-69**: Moderate (e.g., single source, needs verification)
- **0-49**: Low (e.g., unverified, conflicting information)

### Calculating Confidence

```python
base_score = source_weights.get(source, 50)  # Base from source type
agreement_bonus = (num_agreeing_sources - 1) * 5  # Multiple sources agree
recency_bonus = 5 if recent else 0  # Recent information
ambiguity_penalty = -10 if ambiguous else 0  # Ambiguous query

confidence = min(100, max(0, base_score + agreement_bonus + recency_bonus + ambiguity_penalty))
```

## Audit Logging

All KB operations are logged to the `learning_log` table:

### Log Entry Structure

```sql
SELECT * FROM learning_log WHERE action_type LIKE 'kb_%' ORDER BY timestamp DESC LIMIT 10;
```

Fields logged:
- `action_type`: `kb_add`, `kb_update`, `kb_delete`
- `details`: JSON with operation details
- `fact_id`: ID of affected KB entry
- `reviewer`: User/system that performed action
- `reason`: Why the action was taken
- `timestamp`: When it occurred

### Querying Logs

```python
# Get recent KB changes
cursor.execute("""
    SELECT * FROM learning_log 
    WHERE action_type IN ('kb_add', 'kb_update', 'kb_delete')
    ORDER BY timestamp DESC 
    LIMIT 50
""")
```

## Best Practices

### Adding Facts

1. Use specific, searchable keywords (e.g., "Paris", not "city in France")
2. Write complete, accurate fact text
3. Include provenance for verification trail
4. Start with `status='pending'` unless certain
5. Set appropriate confidence scores based on source reliability

### Reviewing Facts

1. Verify facts from multiple sources before marking `true`
2. Mark clearly incorrect facts as `false` to prevent their use
3. Use `needs_review` for ambiguous or conflicting information
4. Document review decisions in audit logs

### Maintaining Quality

1. Periodically review `pending` and `needs_review` facts
2. Update confidence scores as new information emerges
3. Remove outdated or superseded facts
4. Run the worker regularly to process learning queue
5. Monitor `learning_log` for system health

## Troubleshooting

### KB Facts Not Showing in Search

1. Check fact status: `SELECT * FROM knowledge_base WHERE keyword = 'your_keyword';`
2. Verify status is `true` (only `true` facts are returned immediately)
3. Check hybrid memory initialization: ensure `AllieMemoryDB` is available

### Worker Not Processing Queue

1. Verify MySQL connection in worker script
2. Check `learning_queue` table exists and has entries
3. Look for errors in worker output
4. Ensure worker has write permissions to `learning_log`

### API Endpoints Not Responding

1. Verify server is running: `curl http://localhost:8000/api/kb`
2. Check server logs for errors
3. Ensure MySQL is running and accessible
4. Verify `knowledge_base` table exists

### Tests Failing

1. Ensure MySQL is running and accessible
2. Check database credentials in environment or config
3. Verify test database exists: `CREATE DATABASE IF NOT EXISTS allie_memory;`
4. Run cleanup: `DELETE FROM knowledge_base WHERE keyword LIKE 'test_%';`

## Migration from Legacy System

If upgrading from file-based memory:

1. Run migration SQL to create `knowledge_base` table
2. Keep existing `hybrid_memory.json` for backward compatibility
3. Gradually promote high-confidence facts to KB
4. Use worker to automate fact promotion
5. Monitor dual system for consistency

## Security Considerations

### Current Implementation

- Simple header-based role check (`X-User-Role: admin`)
- Suitable for development and trusted environments
- **Not production-ready**

### Production Recommendations

1. Implement proper authentication (OAuth2, JWT)
2. Use role-based access control (RBAC) with real user sessions
3. Add API rate limiting
4. Enable HTTPS for all endpoints
5. Sanitize all user inputs
6. Add CSRF protection for UI operations

## Future Enhancements

Potential improvements:

- [ ] Version control for fact updates (history tracking)
- [ ] Bulk import/export functionality
- [ ] Advanced search with fuzzy matching
- [ ] Automatic conflict resolution
- [ ] Machine learning for confidence scoring
- [ ] Multi-language support
- [ ] Fact expiration/TTL
- [ ] Source credibility scoring
- [ ] Integration with external fact-checking APIs

## Support & Resources

- Server endpoints: `http://localhost:8000/api/kb/*`
- Admin UI: `http://localhost:8000/ui` → "Knowledge Base" tab
- Database: MySQL `allie_memory` database
- Tests: `advanced-memory/test_kb_integration.py`
- Worker: `scripts/kb_worker.py`
- Logs: `learning_log` table in MySQL

## Example Workflow

Complete workflow for adding and using a KB fact:

```python
# 1. Add fact via Python
from db import AllieMemoryDB
db = AllieMemoryDB()
db.add_kb_fact(
    keyword="Python",
    fact="Python is a high-level programming language created by Guido van Rossum",
    source="verified_documentation",
    confidence_score=98,
    status="true"
)

# 2. Search via hybrid memory
from hybrid import HybridMemory
hybrid = HybridMemory()
results = hybrid.search("Python")
print(results['results'][0])  # KB fact returned with high confidence

# 3. Update via API
import requests
requests.patch(
    "http://localhost:8000/api/kb/1",
    json={"confidence_score": 99},
    headers={"X-User-Role": "admin"}
)

# 4. View in UI
# Open browser → http://localhost:8000/ui → Click "Knowledge Base"
# See fact in table, click to edit, save changes

# 5. Check audit log
cursor = db.connection.cursor(dictionary=True)
cursor.execute("SELECT * FROM learning_log WHERE action_type LIKE 'kb_%' ORDER BY timestamp DESC LIMIT 5")
logs = cursor.fetchall()
for log in logs:
    print(f"{log['timestamp']}: {log['action_type']} - {log['details']}")
```

---

**Last Updated**: November 10, 2025  
**Version**: 1.0  
**Status**: Production Ready (with security hardening needed for public deployment)
