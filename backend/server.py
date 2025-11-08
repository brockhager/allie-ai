import os
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

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

# Load real model for chat
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    import torch

    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    logger.info(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # Load existing adapter if available
    adapter_paths = [
        Path("../allie_finetuned"),
        Path("../allie-finetuned/checkpoint-150"),
        Path("../allie_finetuned/checkpoint-100")
    ]

    for adapter_path in adapter_paths:
        if adapter_path.exists():
            logger.info(f"Loading adapter from {adapter_path}")
            model = PeftModel.from_pretrained(model, str(adapter_path))
            break

    logger.info("Model loaded successfully")

except Exception as e:
    logger.warning(f"Failed to load real model: {e}. Using dummy model.")
    # Dummy model for testing (fallback)
    class DummyModel:
        def generate(self, **kwargs):
            return ["Dummy response from model"]

    class DummyTokenizer:
        def __call__(self, prompt, return_tensors=None):
            return {"input_ids": [1, 2, 3]}
        def decode(self, tokens, skip_special_tokens=None):
            return "This is a dummy response for testing purposes."

    tokenizer = DummyTokenizer()
    model = DummyModel()

# Import learning orchestrator
try:
    import subprocess
    import sys
    LEARNING_ENABLED = True
    logger.info("Learning system enabled (subprocess mode)")
except Exception as e:
    logger.warning(f"Learning system not available: {e}")
    LEARNING_ENABLED = False

# Load backup.json if it exists
try:
    with open(DATA_DIR / "backup.json", "r", encoding="utf-8") as f:
        conversation_history = json.load(f)
except FileNotFoundError:
    conversation_history = []
except json.JSONDecodeError:
    conversation_history = []



# -------------------------

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

    # Format as chat message for TinyLlama
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_length=inputs['input_ids'].shape[1] + 200, do_sample=True, temperature=0.7, top_p=0.9)
    reply = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

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

    # Format as chat message for TinyLlama
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_length=inputs['input_ids'].shape[1] + max_tokens, do_sample=True, temperature=0.7, top_p=0.9)
    reply = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

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

# ------------------------- Learning Management Endpoints -------------------------

@app.get("/api/learning/status")
async def get_learning_status():
    """Get current learning system status"""
    if not LEARNING_ENABLED:
        return {"enabled": False, "message": "Learning system not available"}

    try:
        # Run the orchestrator to get status
        scripts_dir = APP_ROOT.parent / "scripts"
        result = subprocess.run(
            [sys.executable, str(scripts_dir / "learning_orchestrator.py")],
            capture_output=True,
            text=True,
            cwd=str(scripts_dir),
            timeout=30
        )

        if result.returncode == 0:
            # Parse the output
            lines = result.stdout.strip().split('\n')
            should_learn = "True" in lines[0]
            current_status = "False" in lines[1]

            return {
                "enabled": True,
                "should_learn": should_learn,
                "is_active": current_status,
                "message": "Status retrieved successfully"
            }
        else:
            return {"enabled": True, "error": result.stderr}

    except Exception as e:
        logger.error(f"Error getting learning status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/learning/start")
async def start_learning_episode():
    """Start a new learning episode"""
    if not LEARNING_ENABLED:
        raise HTTPException(status_code=503, detail="Learning system not available")

    try:
        # Check if learning should be triggered first
        scripts_dir = APP_ROOT.parent / "scripts"
        check_result = subprocess.run(
            [sys.executable, str(scripts_dir / "learning_orchestrator.py")],
            capture_output=True,
            text=True,
            cwd=str(scripts_dir),
            timeout=30
        )

        if "Should learn: True" not in check_result.stdout:
            raise HTTPException(status_code=400, detail="Learning conditions not met")

        # Start learning in background (don't wait for completion)
        subprocess.Popen(
            [sys.executable, str(scripts_dir / "learning_orchestrator.py"), "--start-learning"],
            cwd=str(scripts_dir)
        )

        return {"episode_id": f"learn_{datetime.now().strftime('%Y%m%d_%H%M%S')}", "status": "started"}

    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Learning check timed out")
    except Exception as e:
        logger.error(f"Error starting learning episode: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/learning/history")
async def get_learning_history():
    """Get learning episode history"""
    if not LEARNING_ENABLED:
        return {"enabled": False, "history": []}

    try:
        # For now, return empty history since we don't have persistent storage
        return {"enabled": True, "history": [], "message": "History not yet implemented"}
    except Exception as e:
        logger.error(f"Error getting learning history: {e}")
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
    uvicorn.run(app, host="0.0.0.0", port=8001)
