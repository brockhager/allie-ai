# Allie AI

Allie AI is a full‑stack project that combines a FastAPI backend with a React frontend to create a persistent chat interface. The backend handles storing conversations and generating responses, while the frontend provides a simple chat box for interacting with Allie.

📚 **[Full Documentation](./docs/readme.md)** - Complete technical documentation, architecture details, and recent developments

🧠 **[Memory Validation System](./docs/MEMORY_VALIDATION_SYSTEM.md)** - Automatic knowledge validation and accuracy maintenance

📊 **[MySQL Memory System](./docs/MYSQL_MEMORY_SYSTEM.md)** - Persistent, self-correcting memory architecture with learning pipeline

📖 **[Knowledge Base Guide](./docs/KNOWLEDGE_BASE_GUIDE.md)** - Curated fact system with API, UI, worker, and hybrid memory integration

🚀 **[Quick Teach - Fast Learning System](./quick-teach/README.md)** - Teach Allie multiple topics quickly with bulk learning tools

# Features:

FastAPI backend with endpoints for conversations and UI.

Conversations are saved to data/backup.json and backed up periodically.

Uses TinyLlama model with PEFT fine-tuning for responses.

Simple HTML frontend served by the backend.

# Project Structure:

backend/ contains the FastAPI server (server.py) and Python dependencies (requirements.txt)

frontend/static/ contains the HTML UI (ui.html)

scripts/ contains training scripts (train_allie.py), test scripts (test_app.py, test_automatic_learning.py, test_search.py), and batch files for running them (train_allie.bat, run_server.bat)

Start Allie Server.bat - Quick server startup script (in root directory)

data/ contains conversation data (conversations.json, backup.json, dataset.jsonl) and backups/

outputs/ contains training checkpoints and outputs

allie_finetuned/ contains the fine-tuned model adapter

allie-finetuned/ contains additional training checkpoints

# Setup Instructions:

Backend (FastAPI):

Create and activate a Python virtual environment.

cd backend

Install dependencies with pip install -r requirements.txt.

Run the server with uvicorn server:app --reload --host 127.0.0.1 --port 8000.

Test by visiting http://127.0.0.1:8000/ui for the chat UI.

Training:

cd scripts

pip install -r requirements.txt

python train_allie.py

Frontend:

The frontend is a simple HTML page served by the backend at /ui.

# Testing Endpoints:

To get history: curl http://127.0.0.1:8000/api/conversations/list

To send a message: curl -X POST http://127.0.0.1:8000/api/conversations -H "Content-Type: application/json" -d '{"prompt":"Hello Allie"}'

# Notes:

The backend runs in Python inside a virtual environment.

The frontend is a static HTML page served by FastAPI.

Model files are large; ensure sufficient disk space.

Backup files are in data/backups/.

# Future Improvements:

Streaming responses for smoother chat

Polished UI with chat bubbles, timestamps, and avatars

Deployment guide for hosting backend and frontend together

Authentication and user accounts for multi‑user conversations

## Learning Score System

The UI displays a **Learning Score** (range 1-1000) that reflects Allie's knowledge base growth and quality. This metric is based on verified facts in the `knowledge_base` table and provides real-time feedback on learning progress.

### How the Score is Calculated

The scoring system is implemented in `frontend/static/ui.html` (`updateLearningScore()`) and uses KB statistics from the `/api/hybrid-memory/statistics` endpoint.

**Score Formula:**
```
rawScore = base + quality + growth + confidence
finalScore = (rawScore × 10) + growthBonus
finalScore = clamp(finalScore, 1, 1000)
```

**Components (0-100 raw points):**

1. **Base Score (0-40 points)** - Active KB facts
   - `min((kb_active / 10) × 10, 40)`
   - 10 active facts = 10 points, 40+ facts = max 40 points

2. **Quality Ratio (0-25 points)** - Proportion of verified facts
   - `(kb_active / kb_total) × 25`
   - Higher quality ratio = better score
   - Penalizes accumulation of unverified facts

3. **Growth Score (0-25 points)** - Recent learning activity
   - `min((kb_recent_7d / kb_total) × 100, 25)`
   - Rewards recent fact additions (7-day window)
   - Example: 3/12 facts in last 7 days = ~25 points

4. **Confidence Weighting (0-10 points)** - Average fact confidence
   - `(kb_avg_confidence / 100) × 10`
   - Based on confidence_score field (0-100)
   - Higher confidence = better score

**Growth Bonus:**
- `kb_recent_24h × 10` points added to scaled score
- Immediate feedback when new facts are learned
- Example: Adding 3 facts gives +30 bonus points

### Example Calculations

| Scenario | KB Active | KB Total | Recent 7d | Recent 24h | Avg Confidence | Raw Score | Bonus | Final Score |
|----------|-----------|----------|-----------|------------|----------------|-----------|-------|-------------|
| Empty KB | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **1** (min) |
| Starting | 8 | 8 | 8 | 8 | 87 | 74.7 | 80 | **827** |
| Growing | 12 | 12 | 3 | 3 | 85 | 90.5 | 30 | **935** |
| Mature | 55 | 55 | 10 | 5 | 94 | 98.4 | 50 | **1000** (max) |

### KB Statistics Used

The score calculation queries these statistics from `get_kb_statistics()` in `advanced-memory/db.py`:

- `kb_total`: Total facts in knowledge base
- `kb_active`: Facts with status='true' (verified/accepted)
- `kb_recent_7d`: Facts added in last 7 days
- `kb_recent_24h`: Facts added in last 24 hours
- `kb_avg_confidence`: Average confidence_score of all facts

### Recalculation Triggers

The score updates automatically when:
- New facts are added to knowledge base
- Facts are verified or status changes
- Confidence scores are updated
- UI periodically refreshes (every 30 seconds)

### Testing

Unit tests: `tests/test_learning_score_kb.py`
- Verifies formula calculations
- Tests proportional increases
- Validates range boundaries (1-1000)
- Checks confidence weighting

Integration tests: `tests/test_learning_score_integration.py`
- Simulates conversation → KB growth
- Tests compound growth from multiple conversations
- Verifies confidence quality impact

Run tests:
```bash
python -m pytest tests/test_learning_score_kb.py -v
python -m pytest tests/test_learning_score_integration.py -v
```

### Design Rationale

**Why knowledge_base instead of facts table?**
- Knowledge base contains verified, authoritative facts
- Facts table includes unverified/experimental entries
- KB better reflects actual learning progress

**Why dynamic scaling instead of fixed denominator?**
- Proportional scoring: adding facts immediately increases score
- Avoids "stuck at low score" problem
- Rewards both quantity and quality of learning

**Why include growth bonus?**
- Provides immediate positive feedback
- Encourages continued learning
- Makes score feel responsive to user actions
