#!/usr/bin/env python3
"""
Minimal test server for frontend testing
Serves static files without ML dependencies
"""
import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

app = FastAPI(title="Allie Frontend Test Server")

# Get the static directory
STATIC_DIR = Path(__file__).parent / "frontend" / "static"
if not STATIC_DIR.exists():
    STATIC_DIR.mkdir(parents=True, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Redirect root to /UI page"""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/ui", status_code=302)

@app.get("/ui", response_class=HTMLResponse)
async def get_ui():
    """Serve the tabbed interface"""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return index_path.read_text(encoding="utf-8")
    else:
        return HTMLResponse(content="<h1>Allie - Tabbed Interface</h1><p>Interface not found</p>", status_code=404)

@app.get("/chat", response_class=HTMLResponse)
async def get_chat():
    """Serve the Allie chat UI"""
    ui_path = STATIC_DIR / "ui.html"
    if ui_path.exists():
        return ui_path.read_text(encoding="utf-8")
    else:
        return HTMLResponse(content="<h1>Allie Chat UI</h1><p>Chat interface not found</p>", status_code=404)

@app.get("/fact-check", response_class=HTMLResponse)
async def fact_check_ui():
    """Serve the fact-check UI page"""
    fact_check_path = STATIC_DIR / "fact-check.html"
    if fact_check_path.exists():
        return HTMLResponse(fact_check_path.read_text(encoding="utf-8"))
    else:
        return HTMLResponse("<html><body><h3>Fact Check UI not installed</h3></body></html>", status_code=404)

# Mock API endpoints for testing
@app.get("/api/conversations")
async def mock_conversations():
    """Mock conversations endpoint"""
    return []

@app.get("/api/learning/status")
async def mock_learning_status():
    """Mock learning status endpoint"""
    return {
        "enabled": False,
        "message": "Learning system not available in test server. Use the full server on port 8001."
    }

@app.get("/api/hybrid-memory/statistics")
async def mock_memory_stats():
    """Mock memory statistics endpoint"""
    return {
        "status": "success",
        "statistics": {
            "total_facts": 0,
            "active_facts": 0,
            "outdated_facts": 0,
            "indexed_keywords": 0,
            "average_confidence": 0,
            "pending_review": 0,
            "categories": {},
            "sources": {}
        }
    }

if __name__ == "__main__":
    import uvicorn
    print("Starting minimal test server on http://localhost:8002")
    print("Press Ctrl+C to stop")
    try:
        uvicorn.run(app, host="127.0.0.1", port=8002, log_level="info", access_log=False)
    except KeyboardInterrupt:
        print("Server stopped")