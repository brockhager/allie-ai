import os
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, Body, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
conversation_history = []

# -------------------------
# Config / logging
# -------------------------
APP_ROOT = Path(__file__).parent
STATIC_DIR = APP_ROOT.parent / "frontend" / "static"
DATA_DIR = APP_ROOT.parent / "data"
DATA_DIR.mkdir(exist_ok=True)
_CONV_FILE = DATA_DIR / "conversations.json"

LLAMA_URL = os.environ.get("LLAMA_URL", "http://127.0.0.1:8080/completion")
LLAMA_TIMEOUT_SECONDS = float(os.environ.get("LLAMA_TIMEOUT_SECONDS", "10.0"))
LLAMA_MAX_RETRIES = int(os.environ.get("LLAMA_MAX_RETRIES", "4"))
LLAMA_BACKOFF_BASE = float(os.environ.get("LLAMA_BACKOFF_BASE", "0.25"))

logger = logging.getLogger("allie")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

# Load backup.json if it exists
try:
    with open(DATA_DIR / "backup.json", "r", encoding="utf-8") as f:
        conversation_history = json.load(f)
except FileNotFoundError:
    conversation_history = []
except json.JSONDecodeError:
    conversation_history = []



# -------------------------
# Model setup
# -------------------------
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
base_model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model = PeftModel.from_pretrained(base_model, str(APP_ROOT.parent / "allie_finetuned"))

# -------------------------
# FastAPI app
# -------------------------
app = FastAPI(title="Allie")

# mount static files and ui route
if not STATIC_DIR.exists():
    STATIC_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/ui", response_class=HTMLResponse)
async def ui_index():
    ui_path = STATIC_DIR / "ui.html"
    if not ui_path.exists():
        return HTMLResponse("<html><body><h3>UI not installed</h3></body></html>", status_code=404)
    return HTMLResponse(ui_path.read_text(encoding="utf-8"))


# -------------------------
# httpx AsyncClient (module-level for connection reuse)
# -------------------------
_http_client: Optional[httpx.AsyncClient] = None

def get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(timeout=httpx.Timeout(LLAMA_TIMEOUT_SECONDS, connect=LLAMA_TIMEOUT_SECONDS))
    return _http_client

# -------------------------
# Conversation helpers
# -------------------------
def _read_all_conversations() -> List[Dict[str, Any]]:
    if _CONV_FILE.exists():
        with _CONV_FILE.open("r", encoding="utf-8") as f:
            return json.load(f)
    return []

def _write_all_conversations(objs: List[Dict[str, Any]]) -> None:
    with _CONV_FILE.open("w", encoding="utf-8") as f:
        json.dump(objs, f, indent=2, ensure_ascii=False)



# -------------------------
# Routes
# -------------------------
@app.get("/api/conversations")
async def list_conversations_file():
    # For compatibility with UI, return empty list so UI creates default
    return []

@app.get("/api/conversations/list")
async def list_conversations_memory():
    return {"conversations": conversation_history}

@app.post("/api/conversations")
async def create_conversation(payload: Dict[str, Any] = Body(...)):
    prompt = payload.get("prompt", "")
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=200)
    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Store in memory
    conversation_history.append({"prompt": prompt, "response": reply})

    # ✅ Step 2: Persist to file
    with open(DATA_DIR / "backup.json", "w", encoding="utf-8") as f:
        json.dump(conversation_history, f, indent=2)

    return {"response": reply}





@app.post("/api/generate")
async def generate_response(payload: Dict[str, Any] = Body(...)):
    prompt = payload.get("prompt", "")
    max_tokens = payload.get("max_tokens", 200)
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=max_tokens)
    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {"text": reply}

@app.put("/api/conversations/{conv_id}")
async def update_conversation(conv_id: str, payload: Dict[str, Any] = Body(...)):
    # Dummy implementation - do nothing
    return {"status": "ok"}

@app.delete("/api/conversations/{conv_id}")
async def delete_conversation(conv_id: str):
    # Dummy implementation - do nothing
    return {"status": "ok"}





@app.post("/api/conversations/backup")
async def backup_conversations():
    try:
        with open(DATA_DIR / "backup.json", "w", encoding="utf-8") as f:
            json.dump(conversation_history, f, indent=2)
        return {"status": "backup successful"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# -------------------------
# Graceful shutdown - close httpx client
# -------------------------
@app.on_event("shutdown")
async def _shutdown_event():
    global _http_client
    if _http_client is not None:
        await _http_client.aclose()
        _http_client = None
        logger.info("httpx client closed")

        # ... all your FastAPI routes above ...


# run backups

import shutil, time, os

BACKUP_DIR = str(DATA_DIR / "backups")
MAX_BACKUPS = 10

def backup_conversations():
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    src = str(_CONV_FILE)
    dst = os.path.join(BACKUP_DIR, f"conversations-{timestamp}.json")
    try:
        shutil.copy(src, dst)
        print(f"[Backup] conversations.json saved to {dst}")
    except Exception as e:
        print(f"[Backup error] {e}")

    # --- cleanup logic ---
    try:
        files = sorted(
            [f for f in os.listdir(BACKUP_DIR) if f.startswith("conversations-")],
            key=lambda f: os.path.getmtime(os.path.join(BACKUP_DIR, f))
        )
        if len(files) > MAX_BACKUPS:
            for old in files[:-MAX_BACKUPS]:
                os.remove(os.path.join(BACKUP_DIR, old))
                print(f"[Backup cleanup] deleted {old}")
    except Exception as e:
        print(f"[Cleanup error] {e}")

# Run once immediately to test
backup_conversations()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
