Allie AI
A simple full‑stack project combining a FastAPI backend with a React frontend to create a persistent chat UI. The backend handles conversation storage and responses, while the frontend provides a lightweight interface for chatting with Allie.

🚀 Features
FastAPI backend with endpoints for:

POST /api/conversations → send a new message and get a response

GET /api/conversations/list → retrieve past conversation history

Persistence: conversations survive server restarts via backup.json

React frontend with a chat box UI

CORS enabled so frontend (port 3000) can talk to backend (port 8000)

📂 Project Structure
Code
allie-ai/
  backend/
    server.py          # FastAPI app
    requirements.txt   # Python dependencies
    backup.json        # Persistent conversation storage
  frontend/
    package.json       # React project config
    src/
      App.js           # Root component
      ChatUI.jsx       # Chat UI component
      ChatUI.css       # Styling for chat bubbles
⚙️ Setup Instructions
Backend (FastAPI)
Create and activate a virtual environment:

bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
.\venv\Scripts\activate    # Windows
Install dependencies:

bash
pip install -r requirements.txt
Run the server:

bash
uvicorn server:app --reload --host 127.0.0.1 --port 8000
Test it:

http://127.0.0.1:8000/ping → should return {"status":"ok"}

Frontend (React)
Navigate to the frontend folder:

bash
cd frontend
Install dependencies:

bash
npm install
Start the dev server:

bash
npm start
Open http://localhost:3000 to use the chat UI.

🧪 Testing Endpoints
You can test backend endpoints directly:

bash
# Get history
curl http://127.0.0.1:8000/api/conversations/list

# Send a message
curl -X POST http://127.0.0.1:8000/api/conversations \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Hello Allie"}'
📌 Notes
Backend runs in Python venv, frontend runs in Node.js — two separate processes.

Make sure CORS is enabled in server.py so the browser can connect.

Commit backup.json only if you want to share sample conversations; otherwise add it to .gitignore.
