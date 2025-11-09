# 🗄️ MySQL Memory System Migration

## Overview
Allie has transitioned from an in-memory linked list system to a **persistent, self-correcting MySQL-based memory architecture** with advanced learning capabilities.

## Architecture

### Database Structure

**Database:** `allie_memory`

#### Tables:

1. **`facts`** - Main memory storage
   ```sql
   - id (INT, AUTO_INCREMENT, PRIMARY KEY)
   - keyword (VARCHAR(255), INDEXED)
   - fact (TEXT)
   - source (VARCHAR(255))
   - confidence (FLOAT, DEFAULT 0.8)
   - category (VARCHAR(100), DEFAULT 'general')
   - created_at (TIMESTAMP)
   - updated_at (TIMESTAMP, AUTO-UPDATE)
   ```

2. **`learning_log`** - Audit trail for all changes
   ```sql
   - id (INT, AUTO_INCREMENT, PRIMARY KEY)
   - keyword (VARCHAR(255), INDEXED)
   - old_fact (TEXT)
   - new_fact (TEXT)
   - source (VARCHAR(255))
   - confidence (FLOAT)
   - change_type (ENUM: 'add', 'update', 'delete', 'validate')
   - changed_at (TIMESTAMP)
   ```

3. **`learning_queue`** - Pending facts for validation
   ```sql
   - id (INT, AUTO_INCREMENT, PRIMARY KEY)
   - keyword (VARCHAR(255), INDEXED)
   - fact (TEXT)
   - source (VARCHAR(255))
   - confidence (FLOAT, DEFAULT 0.5)
   - category (VARCHAR(100))
   - status (ENUM: 'pending', 'validated', 'rejected', 'processed')
   - created_at (TIMESTAMP)
   - processed_at (TIMESTAMP, NULLABLE)
   ```

4. **`fact_clusters`** - Grouping related facts
   ```sql
   - id (INT, AUTO_INCREMENT, PRIMARY KEY)
   - cluster_name (VARCHAR(255), UNIQUE)
   - description (TEXT)
   - created_at (TIMESTAMP)
   ```

5. **`cluster_memberships`** - Fact-to-cluster relationships
   ```sql
   - id (INT, AUTO_INCREMENT, PRIMARY KEY)
   - cluster_id (INT, FOREIGN KEY)
   - fact_id (INT, FOREIGN KEY)
   - relevance_score (FLOAT, DEFAULT 1.0)
   ```

## Core Components

### 1. Database Connector (`memory/db.py`)

**Class:** `AllieMemoryDB`

**Methods:**
- `add_fact(keyword, fact, source, confidence, category)` - Add new fact
- `get_fact(keyword)` - Retrieve fact by keyword
- `search_facts(query, limit)` - Search for facts
- `update_fact(keyword, new_fact, source, confidence)` - Update existing fact
- `delete_fact(keyword)` - Delete a fact
- `timeline(limit, include_deleted)` - Get chronological fact history
- `add_to_learning_queue(...)` - Queue fact for validation
- `get_learning_queue(status, limit)` - Get queued facts
- `process_queue_item(queue_id, action, confidence)` - Process queued fact
- `create_cluster(cluster_name, description)` - Create fact cluster
- `add_to_cluster(cluster_name, fact_id, relevance_score)` - Add fact to cluster
- `get_cluster_facts(cluster_name)` - Get all facts in cluster
- `get_statistics()` - Memory statistics

### 2. Learning Pipeline (`memory/learning_pipeline.py`)

**Class:** `LearningPipeline`

**5-Stage Process:**

#### Stage 1: **Ingest**
- Accepts new facts into learning queue
- Assigns initial confidence based on source credibility
- Sources weighted: user (1.0), wikipedia (0.9), wikidata (0.85), etc.

#### Stage 2: **Validate**
- Checks fact against external sources
- Counts agreements/disagreements
- Calculates confidence scores
- Updates queue item with validated confidence

#### Stage 3: **Compare**
- Compares with existing memory facts
- Determines if update is needed
- Considers confidence scores and recency
- Returns action recommendation (update/add/keep/no_change)

#### Stage 4: **Decide**
- Makes final decision based on comparison
- Determines whether to execute action
- Reasons about confidence and data freshness

#### Stage 5: **Confirm**
- Executes the decided action
- Updates memory or rejects fact
- Logs all changes to learning_log

**Methods:**
- `ingest_fact(keyword, fact, source, category)` - Stage 1
- `validate_fact(queue_id, external_check)` - Stage 2
- `compare_fact(queue_id)` - Stage 3
- `decide_action(queue_id, comparison_result)` - Stage 4
- `confirm_action(queue_id, decision)` - Stage 5
- `process_full_pipeline(keyword, fact, source, category)` - Run all stages

## Memory Retrieval Logic

**Priority Order:**
1. Check MySQL for stored facts
2. If no fact exists → query external sources
3. If external sources provide fresher info → update MySQL
4. Return most accurate fact to user

**Code Flow:**
```python
# 1. Check memory first
fact = memory_db.get_fact(keyword)

if fact:
    # 2. Check if fact is recent (< 30 days old)
    if is_recent(fact['updated_at']):
        return fact
    else:
        # 3. Validate against external sources
        external_facts = await query_external_sources(keyword)
        if external_facts:
            # 4. Compare and update if needed
            pipeline.process_full_pipeline(keyword, external_facts[0], 'validation')

# 5. If no memory fact, search externally
external_facts = await query_external_sources(keyword)
if external_facts:
    # 6. Add to memory through learning pipeline
    pipeline.process_full_pipeline(keyword, external_facts[0], external_facts[0]['source'])

return fact or external_facts[0]
```

## Enhanced Learning Features

### 1. **Confidence Scoring**
- Range: 0.0 to 1.0
- Based on:
  - Source credibility (wikipedia: 0.9, user: 1.0, web: 0.6)
  - Cross-source agreement
  - Recency of information
  - External validation results

### 2. **Fact Clustering**
- Groups related facts by topic/entity
- Enables contextual learning
- Faster retrieval for related queries
- Example clusters:
  - "Solar System" → all planet facts
  - "World War II" → historical events
  - "Python Programming" → language facts

### 3. **Learning Queue**
- Batch validation and processing
- Prevents immediate pollution
- Allows bulk operations
- Enables rollback of bad data
- Statuses: pending → validated → processed/rejected

### 4. **Learning History Tracking**
- Complete audit trail in `learning_log`
- Tracks every change: add, update, delete, validate
- Includes old_fact and new_fact for comparison
- Timestamp and source attribution
- Enables "undo" functionality

### 5. **External Source Voting**
- When sources disagree, uses majority voting
- Weighs votes by source credibility
- Confidence = (weighted average + agreement ratio) / 2
- Flags low-confidence facts for review

### 6. **Feedback Hooks** (Coming Soon)
- User corrections: Update fact confidence
- Alternative suggestions: Add to queue for validation
- Source explanations: Show where facts came from

## Setup Instructions

### 1. Install MySQL
```bash
# Windows (using Chocolatey)
choco install mysql

# Or download from: https://dev.mysql.com/downloads/mysql/
```

### 2. Create Database
```sql
CREATE DATABASE allie_memory;
CREATE USER 'allie'@'localhost' IDENTIFIED BY 'your_password';
GRANT ALL PRIVILEGES ON allie_memory.* TO 'allie'@'localhost';
FLUSH PRIVILEGES;
```

### 3. Install Python Dependencies
```bash
pip install mysql-connector-python
```

### 4. Configure Connection
```python
from memory.db import AllieMemoryDB

# Initialize database (creates tables automatically)
memory_db = AllieMemoryDB(
    host='localhost',
    database='allie_memory',
    user='allie',
    password='your_password'
)
```

### 5. Initialize Learning Pipeline
```python
from memory.learning_pipeline import LearningPipeline

# Setup external sources
external_sources = {
    'wikipedia': search_wikipedia,
    'wikidata': search_wikidata,
    'duckduckgo': search_duckduckgo
}

# Create pipeline
pipeline = LearningPipeline(memory_db, external_sources)
```

## Usage Examples

### Basic Memory Operations
```python
# Add a fact
result = memory_db.add_fact(
    keyword="Eiffel Tower",
    fact="The Eiffel Tower is 330 meters tall",
    source="wikipedia",
    confidence=0.9,
    category="geography"
)

# Get a fact
fact = memory_db.get_fact("Eiffel Tower")
print(fact['fact'])  # "The Eiffel Tower is 330 meters tall"

# Update a fact
memory_db.update_fact(
    keyword="Eiffel Tower",
    new_fact="The Eiffel Tower is 330 meters (1,083 ft) tall",
    source="official_source",
    confidence=0.95
)

# Search facts
results = memory_db.search_facts("tower", limit=5)

# Get timeline
timeline = memory_db.timeline(limit=10)
```

### Learning Pipeline
```python
# Process fact through full pipeline
result = await pipeline.process_full_pipeline(
    keyword="Mars",
    fact="Mars is the fourth planet from the Sun",
    source="wikipedia",
    category="astronomy"
)

print(result['stages']['validate']['final_confidence'])
print(result['stages']['decide']['decision']['action'])
```

### Learning Queue Management
```python
# Add to queue
memory_db.add_to_learning_queue(
    keyword="Python",
    fact="Python was created by Guido van Rossum",
    source="web",
    confidence=0.6
)

# Get pending queue
pending = memory_db.get_learning_queue('pending', limit=10)

# Process queue item
for item in pending:
    # Validate
    validation = await pipeline.validate_fact(item['id'])
    
    # Compare
    comparison = await pipeline.compare_fact(item['id'])
    
    # Decide and confirm
    decision = await pipeline.decide_action(item['id'], comparison)
    result = await pipeline.confirm_action(item['id'], decision)
```

### Fact Clustering
```python
# Create cluster
memory_db.create_cluster("Solar System", "Facts about planets and celestial bodies")

# Add facts to cluster
memory_db.add_to_cluster("Solar System", fact_id=1, relevance_score=1.0)
memory_db.add_to_cluster("Solar System", fact_id=2, relevance_score=0.9)

# Get cluster facts
solar_system_facts = memory_db.get_cluster_facts("Solar System")
```

### Statistics
```python
stats = memory_db.get_statistics()
print(f"Total facts: {stats['total_facts']}")
print(f"Average confidence: {stats['average_confidence']}")
print(f"By category: {stats['by_category']}")
print(f"Queue status: {stats['queue_status']}")
```

## Migration from Linked List

### Old System (Linked List)
```python
# Old way
fact_node = memory.head
while fact_node:
    if fact_node.keyword == "Mars":
        return fact_node.fact
    fact_node = fact_node.next
```

### New System (MySQL)
```python
# New way
fact = memory_db.get_fact("Mars")
return fact['fact'] if fact else None
```

### Benefits of MySQL System
✅ **Persistent** - Survives restarts  
✅ **Scalable** - Handles millions of facts  
✅ **Queryable** - Complex searches and filtering  
✅ **Auditable** - Complete change history  
✅ **Concurrent** - Multiple processes can access  
✅ **Self-correcting** - Validates and updates facts  
✅ **Organized** - Clustering and categorization  

## Performance Considerations

### Indexes
- `keyword` indexed for fast lookups
- `category` indexed for filtered queries
- `updated_at` indexed for timeline queries
- Composite indexes on frequent query patterns

### Caching Strategy
- Cache frequently accessed facts in memory
- Cache duration: 5 minutes for high-confidence facts
- Always validate before serving outdated cache

### Batch Operations
- Use learning queue for bulk inserts
- Process in batches of 100
- Reduces database load

## Troubleshooting

### Connection Issues
```python
# Test connection
try:
    memory_db = AllieMemoryDB()
    print("✓ Connected successfully")
except Exception as e:
    print(f"✗ Connection failed: {e}")
```

### Missing Tables
Tables are created automatically on first connection. If issues occur:
```python
memory_db._initialize_tables()
```

### Performance Issues
```sql
-- Check table sizes
SELECT table_name, table_rows 
FROM information_schema.tables 
WHERE table_schema = 'allie_memory';

-- Optimize tables
OPTIMIZE TABLE facts, learning_log, learning_queue;

-- Analyze indexes
ANALYZE TABLE facts;
```

## Future Enhancements

- [ ] NLP-based fact similarity matching
- [ ] Automatic conflict resolution
- [ ] Fact deprecation and archival
- [ ] Multi-lingual support
- [ ] Graph-based fact relationships
- [ ] Real-time fact validation webhooks
- [ ] ML-based confidence prediction
- [ ] Distributed memory across multiple databases

---

**Status:** ✅ Implemented and ready for use  
**Last Updated:** November 8, 2025
