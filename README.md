# Allie AI

Allie AI is a full‑stack project that combines a FastAPI backend with a React frontend to create a persistent chat interface. The backend handles storing conversations and generating responses, while the frontend provides a simple chat box for interacting with Allie.

# Features:

FastAPI backend with two main endpoints:

POST /api/conversations to send a new message and get a response

GET /api/conversations/list to retrieve past conversation history

Conversations are saved so they survive server restarts

React frontend with a chat box UI

CORS enabled so the frontend (running on port 3000) can talk to the backend (running on port 8000)

# Project Structure:

backend/ contains the FastAPI server (server.py), Python dependencies (requirements.txt), and the conversation backup file (backup.json)

frontend/ contains the React app (package.json) and source files (App.js, ChatUI.jsx, ChatUI.css)

# Setup Instructions:

Backend (FastAPI):

Create and activate a Python virtual environment.

Install dependencies with pip install -r requirements.txt.

Run the server with uvicorn server:app --reload --host 127.0.0.1 --port 8000.

Test by visiting http://127.0.0.1:8000/ping — it should return {"status":"ok"}.

Frontend (React):

Go into the frontend folder.

Install dependencies with npm install.

Start the development server with npm start.

Open http://localhost:3000 in your browser to use the chat UI.

# Testing Endpoints:

To get history: curl http://127.0.0.1:8000/api/conversations/list

To send a message: curl -X POST http://127.0.0.1:8000/api/conversations -H "Content-Type: application/json" -d '{"prompt":"Hello Allie"}'

# Notes:

The backend runs in Python inside a virtual environment, while the frontend runs in Node.js — they are separate processes.

Make sure CORS is enabled in server.py so the browser can connect.

You can commit backup.json if you want to share sample conversations, or add it to .gitignore if not.

# Future Improvements:

Streaming responses for smoother chat

Polished UI with chat bubbles, timestamps, and avatars

Deployment guide for hosting backend and frontend together

Authentication and user accounts for multi‑user conversations
