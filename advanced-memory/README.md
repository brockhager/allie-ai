# Advanced Memory System for Allie

A comprehensive MySQL-based memory system with self-correcting capabilities, confidence scoring, and intelligent conflict resolution.

## 🎯 Overview

This advanced memory system replaces the simple linked-list approach with a persistent, feature-rich MySQL database that includes:

- **5-Table Architecture**: Facts, learning log, learning queue, fact clusters, and cluster memberships
- **5-Stage Learning Pipeline**: Ingest → Validate → Compare → Decide → Confirm
- **Confidence Scoring**: Source credibility weighting and fact confidence tracking
- **Conflict Resolution**: Automatic detection and resolution of conflicting facts
- **Learning Queue**: Review and validate facts before adding to memory
- **Fact Clustering**: Group related facts for better organization
- **Audit Trail**: Complete history of all memory changes

## 📋 Prerequisites

- MySQL Server (v5.7+)
- Python 3.8+
- mysql-connector-python library

## 🚀 Setup

### 1. Database Setup

The system will automatically create all necessary tables when first initialized. However, ensure your MySQL database exists:

```sql
-- Already created: allie_memory database
-- Already created: allie user with password
```

### 2. Python Dependencies

```bash
pip install mysql-connector-python
```

### 3. Configuration

The database connector uses these default credentials:

```python
host = 'localhost'
database = 'allie_memory'
user = 'allie'
password = 'StrongPassword123!'
```

To use different credentials, pass them when initializing:

```python
from db import AllieMemoryDB

memory = AllieMemoryDB(
    host='your_host',
    database='your_database',
    user='your_user',
    password='your_password'
)
```

## 📚 Usage

### Basic Memory Operations

```python
from db import AllieMemoryDB

# Initialize
memory = AllieMemoryDB()

# Add a fact
result = memory.add_fact(
    keyword='brock',
    fact='Brock is 6 feet 2 inches tall',
    source='user',
    confidence=0.9,
    category='personal'
)

# Get a fact
fact = memory.get_fact('brock')
print(f"{fact['keyword']}: {fact['fact']}")

# Search facts
results = memory.search_facts('height', limit=10)

# Update a fact
memory.update_fact(
    keyword='brock',
    new_fact='Brock is 6 feet 2 inches tall and works as a developer',
    source='conversation',
    confidence=0.85
)

# Delete a fact
memory.delete_fact('brock')

# Get timeline
recent_facts = memory.timeline(limit=50)
```

### Learning Pipeline

```python
from db import AllieMemoryDB
from learning_pipeline import LearningPipeline

# Initialize
memory = AllieMemoryDB()
pipeline = LearningPipeline(memory)

# Process a single fact through the pipeline
result = pipeline.process_fact(
    keyword='python',
    fact='Python is a high-level programming language',
    source='web_search',
    base_confidence=0.8,
    category='programming',
    auto_resolve=True
)

print(f"Status: {result['final_status']}")
print(f"Confidence: {result['confidence']}")

# Process multiple facts in batch
facts = [
    {
        'keyword': 'javascript',
        'fact': 'JavaScript runs in web browsers',
        'source': 'user',
        'confidence': 0.9,
        'category': 'programming'
    },
    {
        'keyword': 'mysql',
        'fact': 'MySQL is a relational database',
        'source': 'quick_teach',
        'confidence': 0.95,
        'category': 'database'
    }
]

batch_result = pipeline.process_batch(facts)
print(f"Added: {batch_result['added']}, Updated: {batch_result['updated']}")
```

### Learning Queue Management

```python
# Add fact to queue for review
memory.add_to_learning_queue(
    keyword='ai',
    fact='AI will replace all programmers',
    source='web_search',
    confidence=0.4,
    category='technology'
)

# Get pending items
pending = memory.get_learning_queue(status='pending')
for item in pending:
    print(f"{item['keyword']}: {item['fact']} (confidence: {item['confidence']})")

# Process queued item
memory.process_queue_item(
    queue_id=1,
    action='validate',  # or 'reject' or 'process'
    confidence=0.8
)
```

### Fact Clustering

```python
# Create a cluster
memory.create_cluster(
    cluster_name='programming_languages',
    description='Facts about programming languages'
)

# Add facts to cluster
memory.add_to_cluster('programming_languages', fact_id=1, relevance_score=0.95)
memory.add_to_cluster('programming_languages', fact_id=2, relevance_score=0.90)

# Get all facts in a cluster
cluster_facts = memory.get_cluster_facts('programming_languages')
for fact in cluster_facts:
    print(f"{fact['keyword']}: {fact['fact']} (relevance: {fact['relevance_score']})")
```

### Statistics and Monitoring

```python
# Get memory statistics
stats = memory.get_statistics()
print(f"Total facts: {stats['total_facts']}")
print(f"Average confidence: {stats['average_confidence']}")
print(f"Facts by category: {stats['by_category']}")
print(f"Queue status: {stats['queue_status']}")

# Get pipeline statistics
pipeline_stats = pipeline.get_pipeline_stats()
print(f"Pending review: {pipeline_stats['pending_review']}")
print(f"Total processed: {pipeline_stats['processed']}")
```

## 🔄 Learning Pipeline Stages

### Stage 1: Ingest
- Receives new fact
- Adjusts confidence based on source credibility
- Source credibility scores:
  - `quick_teach`: 0.95 (user-verified bulk teaching)
  - `user`: 0.9 (direct user input)
  - `conversation`: 0.85 (learned from chat)
  - `external_api`: 0.75
  - `web_search`: 0.7
  - `inference`: 0.6 (derived facts)
  - `unknown`: 0.5

### Stage 2: Validate
- Checks fact quality
- Validates keyword and content
- Detects uncertain language
- Extracts keywords

### Stage 3: Compare
- Compares with existing knowledge
- Detects conflicts
- Calculates similarity

### Stage 4: Decide
- Determines action:
  - **Add**: New fact, no conflicts
  - **Update**: Refinement of existing fact
  - **Replace**: New fact has higher confidence
  - **Keep**: Existing fact has higher confidence
  - **Merge**: Similar confidence, needs review
  - **Queue**: Conflict detected, manual review needed
  - **Skip**: Identical to existing fact

### Stage 5: Confirm
- Applies decision to database
- Updates learning log
- Returns final status

## 🔧 Integration with Allie

### Option 1: Replace Existing Memory

```python
# In backend/server.py
from advanced_memory.db import AllieMemoryDB
from advanced_memory.learning_pipeline import LearningPipeline

# Replace current memory initialization
memory = AllieMemoryDB()
pipeline = LearningPipeline(memory)

# Update endpoints to use new memory
@app.post("/api/learn")
async def learn(request: LearnRequest):
    result = pipeline.process_fact(
        keyword=request.keyword,
        fact=request.fact,
        source=request.source or 'user',
        base_confidence=0.8,
        category='general'
    )
    return {"status": "success", "result": result}
```

### Option 2: Gradual Migration

Keep both systems running and migrate gradually:

```python
# Use simple memory for reading
from memory.db import MemoryDB
simple_memory = MemoryDB()

# Use advanced memory for new learning
from advanced_memory.db import AllieMemoryDB
from advanced_memory.learning_pipeline import LearningPipeline
advanced_memory = AllieMemoryDB()
pipeline = LearningPipeline(advanced_memory)

# Migrate existing facts
for fact in simple_memory.timeline():
    pipeline.process_fact(
        keyword=fact['keyword'],
        fact=fact['fact'],
        source='migration',
        base_confidence=0.8
    )
```

## 📊 Database Schema

### `facts` Table
```sql
- id (INT, PRIMARY KEY)
- keyword (VARCHAR(255), INDEXED)
- fact (TEXT)
- source (VARCHAR(255))
- confidence (FLOAT)
- category (VARCHAR(100), INDEXED)
- created_at (TIMESTAMP)
- updated_at (TIMESTAMP, INDEXED)
```

### `learning_log` Table
```sql
- id (INT, PRIMARY KEY)
- keyword (VARCHAR(255), INDEXED)
- old_fact (TEXT)
- new_fact (TEXT)
- source (VARCHAR(255))
- confidence (FLOAT)
- change_type (ENUM: add, update, delete, validate)
- changed_at (TIMESTAMP, INDEXED)
```

### `learning_queue` Table
```sql
- id (INT, PRIMARY KEY)
- keyword (VARCHAR(255), INDEXED)
- fact (TEXT)
- source (VARCHAR(255))
- confidence (FLOAT)
- category (VARCHAR(100))
- status (ENUM: pending, validated, rejected, processed, INDEXED)
- created_at (TIMESTAMP)
- processed_at (TIMESTAMP)
```

### `fact_clusters` Table
```sql
- id (INT, PRIMARY KEY)
- cluster_name (VARCHAR(255), UNIQUE, INDEXED)
- description (TEXT)
- created_at (TIMESTAMP)
```

### `cluster_memberships` Table
```sql
- id (INT, PRIMARY KEY)
- cluster_id (INT, FOREIGN KEY)
- fact_id (INT, FOREIGN KEY)
- relevance_score (FLOAT)
- UNIQUE(cluster_id, fact_id)
```

## 🧪 Testing

Run the test script to verify everything works:

```bash
python advanced-memory/test_advanced_memory.py
```

This will:
1. Test database connection
2. Add test facts
3. Test search functionality
4. Test learning pipeline
5. Test queue management
6. Test clustering
7. Display statistics

## 🎓 Examples

### Teaching Allie About Programming

```python
from advanced_memory.db import AllieMemoryDB
from advanced_memory.learning_pipeline import LearningPipeline

memory = AllieMemoryDB()
pipeline = LearningPipeline(memory)

programming_facts = [
    {
        'keyword': 'python',
        'fact': 'Python is a high-level, interpreted programming language',
        'source': 'quick_teach',
        'confidence': 0.95,
        'category': 'programming'
    },
    {
        'keyword': 'javascript',
        'fact': 'JavaScript is primarily used for web development',
        'source': 'quick_teach',
        'confidence': 0.95,
        'category': 'programming'
    },
    {
        'keyword': 'sql',
        'fact': 'SQL is used to manage and query relational databases',
        'source': 'quick_teach',
        'confidence': 0.95,
        'category': 'programming'
    }
]

result = pipeline.process_batch(programming_facts)
print(f"Taught Allie {result['added']} programming facts!")

# Create a cluster
memory.create_cluster('programming', 'Programming languages and concepts')
```

### Handling Conflicting Information

```python
# Initial fact
pipeline.process_fact(
    keyword='earth_age',
    fact='Earth is approximately 4.5 billion years old',
    source='user',
    base_confidence=0.9,
    category='science'
)

# Conflicting fact with lower confidence
result = pipeline.process_fact(
    keyword='earth_age',
    fact='Earth is 10,000 years old',
    source='web_search',
    base_confidence=0.3,
    category='science',
    auto_resolve=True
)

# Pipeline keeps higher confidence fact
print(result['final_status'])  # Outputs: skipped or kept original
```

## 🔒 Security Notes

- Change default MySQL credentials in production
- Use environment variables for sensitive data
- Implement proper user authentication
- Sanitize all user inputs
- Use prepared statements (already implemented)

## 📈 Performance Tips

- Index columns are already created for optimal queries
- Use batch processing for multiple facts
- Regularly archive old learning_log entries
- Monitor queue size and process pending items
- Use clustering to group related facts

## 🐛 Troubleshooting

### Connection Issues
```python
# Test connection
from advanced_memory.db import AllieMemoryDB
try:
    memory = AllieMemoryDB()
    print("✓ Connection successful")
except Exception as e:
    print(f"✗ Connection failed: {e}")
```

### Missing Tables
Tables are created automatically. If you encounter issues:
```sql
-- Drop and recreate database
DROP DATABASE allie_memory;
CREATE DATABASE allie_memory;
```

### Queue Buildup
If too many items in queue:
```python
# Batch process pending items
pending = memory.get_learning_queue('pending', limit=100)
for item in pending:
    memory.process_queue_item(item['id'], 'process')
```

## 📝 TODO / Future Enhancements

- [ ] NLP-based semantic similarity for conflict detection
- [ ] External fact verification via API (Wikipedia, etc.)
- [ ] Web UI for queue management
- [ ] Automated clustering using ML
- [ ] Export/import functionality
- [ ] Fact versioning system
- [ ] Integration with LLM for fact generation
- [ ] Real-time confidence adjustment based on usage

## 📄 License

Part of Allie AI Assistant project

## 🤝 Contributing

This is a personal project for Allie. Modify as needed for your use case.

---

**Need help?** Check the test script or refer to the inline documentation in `db.py` and `learning_pipeline.py`.
