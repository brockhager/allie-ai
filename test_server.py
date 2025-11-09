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
    """Serve the main tabbed interface"""
    index_path = STATIC_DIR / "index_test.html"
    if index_path.exists():
        return index_path.read_text(encoding="utf-8")
    else:
        return HTMLResponse(content="<h1>Allie - AI Assistant</h1><p>Main interface not found</p>", status_code=404)

@app.get("/ui", response_class=HTMLResponse)
async def get_ui():
    """Serve the Allie UI page (for iframe embedding)"""
    ui_path = STATIC_DIR / "ui.html"
    if ui_path.exists():
        return ui_path.read_text(encoding="utf-8")
    else:
        return HTMLResponse(content="<h1>Allie UI</h1><p>UI file not found</p>", status_code=404)

@app.get("/fact-check", response_class=HTMLResponse)
async def fact_check_ui():
    """Serve the fact-check UI page"""
    fact_check_path = STATIC_DIR / "fact-check.html"
    if fact_check_path.exists():
        return HTMLResponse(fact_check_path.read_text(encoding="utf-8"))
    else:
        return HTMLResponse("<html><body><h3>Fact Check UI not installed</h3></body></html>", status_code=404)

if __name__ == "__main__":
    import uvicorn
    print("Starting minimal test server on http://localhost:8001")
    print("Press Ctrl+C to stop")
    try:
        uvicorn.run(app, host="127.0.0.1", port=8001, log_level="info", access_log=False)
    except KeyboardInterrupt:
        print("Server stopped")