# Knowledge Base System

## Overview

The Knowledge Base (KB) is a curated fact storage and retrieval system that provides verified, high-confidence information to Allie. KB facts take precedence over external sources in the hybrid memory retrieval pipeline, ensuring Allie uses trusted information when available.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Query                                │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              Hybrid Memory Search                            │
│  1. Check Knowledge Base (curated facts)                     │
│  2. Check Local Memory (file-based)                          │
│  3. External Sources (Wikipedia, DBpedia, etc.)              │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                 Response Generation                          │
│  - KB facts marked 'true' → high confidence                  │
│  - KB facts marked 'false' → excluded with warning           │
│  - No KB entry → use external sources                        │
└─────────────────────────────────────────────────────────────┘
```

## Key Features

### 🎯 Curated Facts
- Manually reviewed and verified information
- Quality over quantity approach
- Human-in-the-loop validation

### 📊 Status Management
- **true**: Verified correct, use in responses
- **false**: Verified incorrect, exclude from responses  
- **pending**: Awaiting review
- **needs_review**: Flagged for human verification

### 🔢 Confidence Scoring
- 0-100 scale based on source reliability
- Affects prioritization in search results
- Calculated from source weights, agreement, recency, and ambiguity

### 📝 Full Audit Trail
- All operations logged to `learning_log`
- Track who changed what and why
- Supports compliance and debugging

### 🔄 Automated Reconciliation
- Worker script processes learning queue
- Auto-promotes high-confidence facts
- Detects and flags conflicts

## Database Schema

### knowledge_base Table

| Column | Type | Description |
|--------|------|-------------|
| id | INT | Primary key |
| keyword | VARCHAR(255) | Lookup key (e.g., "Paris", "Python") |
| fact | TEXT | The verified fact text |
| source | VARCHAR(255) | Origin (e.g., "wikipedia", "manual") |
| status | ENUM | 'true', 'false', 'pending', 'needs_review' |
| confidence_score | INT | 0-100, higher is more confident |
| provenance | TEXT | Details about verification |
| created_at | TIMESTAMP | When fact was added |
| updated_at | TIMESTAMP | Last modification time |

### Key Indexes
- `idx_kb_keyword` on `keyword` - Fast lookups
- `idx_kb_status` on `status` - Filter by status
- `idx_kb_confidence` on `confidence_score` - Sort by confidence

## API Endpoints

### REST API

**Base URLs**: `/api/kb/*` or `/api/knowledge-base/*`

| Endpoint | Method | Description | Auth |
|----------|--------|-------------|------|
| `/api/kb` | GET | List all facts (with filters) | Public |
| `/api/kb/{id}` | GET | Get single fact | Public |
| `/api/kb` | POST | Create new fact | Public |
| `/api/kb/{id}` | PATCH | Update fact | Admin |
| `/api/kb/{id}` | DELETE | Delete fact | Admin |

### Query Parameters (GET /api/kb)
- `status`: Filter by status (optional)
- `limit`: Max results, default 50 (optional)
- `offset`: Pagination offset, default 0 (optional)

### Request Body (POST /api/kb)
```json
{
  "keyword": "Tokyo",
  "fact": "Tokyo is the capital of Japan",
  "source": "wikipedia",
  "confidence_score": 95,
  "provenance": "Official government website",
  "status": "true"
}
```

## User Interface

### Accessing the UI
1. Start server: `uvicorn server:app --host 0.0.0.0 --port 8000`
2. Open browser: `http://localhost:8000/ui`
3. Click **"Knowledge Base"** button in header

### UI Features
- **View All Facts**: Browse KB entries in a table
- **Filter by Status**: Show only true/false/pending facts
- **Add Facts**: Click "Add New Fact" button
- **Edit Facts**: Click any fact to edit inline
- **Delete Facts**: Remove incorrect or outdated entries
- **Color Coding**: 
  - 🟢 Green = true (verified)
  - 🔴 Red = false (incorrect)
  - 🟡 Yellow = pending
  - 🟠 Orange = needs_review

## Python API

### Import
```python
from advanced_memory.db import AllieMemoryDB
from advanced_memory.hybrid import HybridMemory
```

### Basic Operations

**Add a Fact:**
```python
db = AllieMemoryDB()
db.add_kb_fact(
    keyword="Berlin",
    fact="Berlin is the capital of Germany",
    source="verified_source",
    confidence_score=97,
    provenance="Official government source",
    status="true"
)
```

**Retrieve a Fact:**
```python
fact = db.get_kb_fact("Berlin")
print(fact['fact'])  # "Berlin is the capital of Germany"
```

**List All Facts:**
```python
# All facts
all_facts = db.get_all_kb_facts()

# Only verified facts
verified = db.get_all_kb_facts(status="true", limit=100)
```

**Update a Fact:**
```python
db.update_kb_fact(
    kb_id=1,
    new_fact="Updated fact text",
    status="true",
    confidence_score=99,
    reviewer="admin_user",
    reason="Added more detail"
)
```

**Delete a Fact:**
```python
db.delete_kb_fact(
    kb_id=1,
    reviewer="admin_user",
    reason="Duplicate entry"
)
```

## Hybrid Memory Integration

The KB is integrated into Allie's hybrid memory system with highest priority:

```python
hybrid = HybridMemory()
results = hybrid.search("Paris")

# Priority order:
# 1. Check KB first
#    - If status='true' → return immediately
#    - If status='false' → exclude, add warning
# 2. Check local memory (file-based)
# 3. Query external sources (Wikipedia, etc.)
```

### KB Priority Rules

| KB Status | Behavior |
|-----------|----------|
| `true` | Return immediately with high confidence |
| `false` | Exclude from results, include warning message |
| `pending` | Fall through to normal search |
| `needs_review` | Fall through to normal search |
| No KB entry | Normal multi-source search |

## Worker & Automation

### KB Worker Script

Located at `scripts/kb_worker.py`

**Purpose**: Automatically process the learning queue and promote high-confidence facts to the KB

**Logic**:
- Confidence ≥ 0.85 → Auto-promote to KB with `status='true'`
- Confidence 0.5-0.85 → Add to KB with `status='needs_review'`
- Confidence < 0.5 → Add to KB with `status='pending'`

**Running the Worker**:
```bash
cd scripts
python kb_worker.py
```

**Scheduling** (Windows):
```powershell
schtasks /create /tn "AllieKBWorker" /tr "python C:\path\to\scripts\kb_worker.py" /sc hourly
```

**Scheduling** (Linux/Mac):
```bash
# Add to crontab (run every hour)
0 * * * * cd /path/to/allie-ai && python scripts/kb_worker.py
```

## Testing

### Integration Tests

Run the full test suite:
```bash
cd advanced-memory
python test_kb_integration.py
```

**Test Coverage**:
- ✅ KB CRUD operations (add, get, update, delete)
- ✅ Status filtering (true/false/pending)
- ✅ Confidence scoring
- ✅ Hybrid memory KB preference
- ✅ Audit logging to learning_log
- ✅ Learning queue integration

**Expected Output**:
```
test_add_kb_fact ... ok
test_delete_kb_fact ... ok
test_get_all_kb_facts ... ok
test_get_kb_fact_not_found ... ok
test_hybrid_memory_kb_preference_false ... ok
test_hybrid_memory_kb_preference_true ... ok
test_kb_audit_logging ... ok
test_kb_confidence_scoring ... ok
test_update_kb_fact ... ok
test_add_to_learning_queue ... ok

----------------------------------------------------------------------
Ran 10 tests in 0.827s

OK
```

### Smoke Tests

Run end-to-end smoke tests:
```bash
cd scripts
python smoke_test_kb.py
```

Tests server connectivity, API operations, and hybrid memory integration.

## Migrations

### Running Migrations

**Create KB Table**:
```powershell
Get-Content migrations\sql\001_create_knowledge_base.sql | mysql -u root -p"PASSWORD" allie_memory
```

**Add Audit Columns to learning_log**:
```powershell
Get-Content migrations\sql\002_alter_learning_log.sql | mysql -u root -p"PASSWORD" allie_memory
```

### Rollback

```powershell
Get-Content migrations\sql\002_drop_learning_log_columns.sql | mysql -u root -p"PASSWORD" allie_memory
Get-Content migrations\sql\001_drop_knowledge_base.sql | mysql -u root -p"PASSWORD" allie_memory
```

See `migrations/sql/README.md` for detailed migration instructions.

## Confidence Scoring

### Score Ranges

| Range | Label | Description |
|-------|-------|-------------|
| 90-100 | Highly Verified | Official sources, multiple confirmations |
| 70-89 | Reliable | Wikipedia, established sources |
| 50-69 | Moderate | Single source, needs verification |
| 0-49 | Low | Unverified, conflicting information |

### Calculation Formula

```python
base_score = source_weights.get(source, 50)
agreement_bonus = (num_agreeing_sources - 1) * 5
recency_bonus = 5 if recent else 0
ambiguity_penalty = -10 if ambiguous else 0

confidence = min(100, max(0, 
    base_score + agreement_bonus + recency_bonus + ambiguity_penalty
))
```

### Source Weights

| Source | Base Weight |
|--------|-------------|
| verified_source | 95 |
| wikipedia | 85 |
| manual_entry | 80 |
| wikidata | 80 |
| dbpedia | 75 |
| external | 50 |

## Best Practices

### Adding Facts

✅ **Do**:
- Use specific, searchable keywords
- Write complete, accurate fact text
- Include provenance for verification
- Start with `status='pending'` unless certain
- Set appropriate confidence scores

❌ **Don't**:
- Use vague or generic keywords
- Add unverified information as `status='true'`
- Omit source information
- Set confidence > 90 without strong evidence

### Reviewing Facts

✅ **Do**:
- Verify from multiple sources before marking `true`
- Mark clearly incorrect facts as `false`
- Use `needs_review` for ambiguous cases
- Document review decisions in audit logs
- Update confidence as new info emerges

❌ **Don't**:
- Rush verification process
- Leave `pending` facts indefinitely
- Delete facts without documenting reason
- Change status without reviewer/reason

### Maintenance

- Review `pending` and `needs_review` facts weekly
- Run worker regularly (daily or hourly)
- Monitor `learning_log` for anomalies
- Remove outdated or superseded facts
- Update confidence scores as sources improve

## Troubleshooting

### KB Facts Not Appearing in Search

**Check fact status**:
```sql
SELECT * FROM knowledge_base WHERE keyword = 'your_keyword';
```

Only `status='true'` facts are returned immediately by hybrid search.

**Verify hybrid memory initialization**:
```python
from advanced_memory.hybrid import HybridMemory
hybrid = HybridMemory()
print(hybrid.db)  # Should not be None
```

### Worker Not Processing Queue

1. Check MySQL connection
2. Verify `learning_queue` table exists
3. Check worker logs for errors
4. Ensure worker has write permissions

### API Endpoints Not Responding

1. Verify server is running: `curl http://localhost:8000/api/kb`
2. Check server logs for errors
3. Ensure MySQL is running
4. Verify `knowledge_base` table exists

### Tests Failing

1. Ensure MySQL is running and accessible
2. Check database credentials
3. Run migration scripts
4. Clean test data: `DELETE FROM knowledge_base WHERE keyword LIKE 'test_%';`

## Security

### Current Implementation

⚠️ **Development-Only Security**:
- Simple header-based role check (`X-User-Role: admin`)
- Suitable for trusted environments only
- **Not production-ready**

### Production Recommendations

For public deployment:
1. ✅ Implement OAuth2 or JWT authentication
2. ✅ Use real RBAC with user sessions
3. ✅ Add API rate limiting
4. ✅ Enable HTTPS for all endpoints
5. ✅ Sanitize all user inputs
6. ✅ Add CSRF protection for UI
7. ✅ Encrypt sensitive data at rest
8. ✅ Implement audit log access controls

## Performance

### Optimization Tips

**Indexing**:
- Keywords are indexed for fast lookups
- Status and confidence indexed for filtering
- Consider adding composite indexes for common queries

**Caching**:
- Consider caching frequently accessed KB facts
- Implement TTL for cache invalidation
- Use Redis for distributed caching

**Query Optimization**:
- Use `LIMIT` and `OFFSET` for pagination
- Filter by `status` to reduce result set
- Avoid `SELECT *` in production code

## Future Enhancements

Potential improvements:

- [ ] Version control for fact updates (history tracking)
- [ ] Bulk import/export functionality  
- [ ] Advanced search with fuzzy matching
- [ ] Automatic conflict resolution
- [ ] ML-based confidence scoring
- [ ] Multi-language support
- [ ] Fact expiration/TTL
- [ ] Source credibility scoring
- [ ] Integration with external fact-checking APIs
- [ ] Real-time WebSocket updates
- [ ] Collaborative editing with locking
- [ ] Fact relationship graphs

## Resources

### Documentation
- **Main Guide**: `docs/KNOWLEDGE_BASE_GUIDE.md` - Comprehensive 650+ line guide
- **Migrations**: `migrations/sql/README.md` - Migration instructions
- **Main README**: `README.md` - Project overview with KB link

### Code
- **Database Layer**: `advanced-memory/db.py` - AllieMemoryDB class
- **Hybrid Integration**: `advanced-memory/hybrid.py` - HybridMemory class
- **API Endpoints**: `backend/server.py` - FastAPI routes
- **Admin UI**: `frontend/static/kb.html` - Browser interface
- **Worker**: `scripts/kb_worker.py` - Reconciliation worker

### Tests
- **Integration Tests**: `advanced-memory/test_kb_integration.py` - 10 tests
- **Smoke Tests**: `scripts/smoke_test_kb.py` - End-to-end validation

### URLs (when server running)
- **API Base**: `http://localhost:8000/api/kb`
- **Admin UI**: `http://localhost:8000/ui` → Click "Knowledge Base"
- **API Docs**: `http://localhost:8000/docs` (FastAPI auto-generated)

## Support

For issues or questions:
1. Check this README and `docs/KNOWLEDGE_BASE_GUIDE.md`
2. Review test files for usage examples
3. Check server logs for error messages
4. Verify database schema with `DESCRIBE knowledge_base;`
5. Run integration tests to validate setup

---

**Last Updated**: November 10, 2025  
**Version**: 1.0  
**Status**: Production-ready with security hardening recommended for public deployment
