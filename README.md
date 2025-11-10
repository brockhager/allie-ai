# Allie AI

Allie AI is a full‑stack project that combines a FastAPI backend with a React frontend to create a persistent chat interface. The backend handles storing conversations and generating responses, while the frontend provides a simple chat box for interacting with Allie.

📚 **[Full Documentation](./docs/readme.md)** - Complete technical documentation, architecture details, and recent developments

🧠 **[Memory Validation System](./docs/MEMORY_VALIDATION_SYSTEM.md)** - Automatic knowledge validation and accuracy maintenance

�️ **[MySQL Memory System](./docs/MYSQL_MEMORY_SYSTEM.md)** - Persistent, self-correcting memory architecture with learning pipeline

�🚀 **[Quick Teach - Fast Learning System](./quick-teach/README.md)** - Teach Allie multiple topics quickly with bulk learning tools

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

## Learning Score (UI) — tuning notes

The frontend exposes a "Learning Score" displayed in the right-hand panel. This is a conservative, heuristic metric (range 0–1000) intended as a rough indicator of how well Allie is learning and how reliable the stored facts are.

Key points about the current heuristic (implemented in `frontend/static/ui.html` -> `updateLearningScore()`):

- The score is computed from several components that sum to a 0–100 base, and then scaled by 10 to produce the 0–1000 display value.
- Positive contributions include:
	- active facts (weighted, up to ~30 points)
	- fact quality ratio (active / total facts, up to ~30 points)
	- category/topic diversity (up to ~15 points)
	- quality conversations (up to ~15 points)
	- engagement (total conversations, up to ~10 points)
- Penalties subtract from the raw 0–100 base before scaling. Penalty signals include:
	- user `falseReports` (strong penalty multiplier) — these are the heaviest negative signals
	- stored negative examples (user-provided corrections saved to memory)
	- outdated facts
- Current penalty multipliers (tunable):
	- falseReports × 5
	- negativeExamples × 2.0
	- outdatedFacts × 0.5
	- penalty capped at 80 points

If the score appears too high or too low for your environment, tweak the multipliers and caps in `frontend/static/ui.html` or request I add a small debug overlay that shows the component breakdown (active facts, quality ratio, diversity, engagement, penalty) to assist tuning.
