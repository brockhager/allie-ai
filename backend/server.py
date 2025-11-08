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
        device_map="auto",
        local_files_only=True  # Only use local files, don't download
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

# File management utilities
MAX_FILES_PER_FOLDER = 30

def cleanup_folder(folder_path: Path, max_files: int = MAX_FILES_PER_FOLDER):
    """Clean up old files in a folder, keeping only the most recent max_files"""
    if not folder_path.exists():
        return

    try:
        # Get all files in the folder (not subdirectories)
        files = [f for f in folder_path.iterdir() if f.is_file()]

        if len(files) <= max_files:
            return

        # Sort by modification time (newest first)
        files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        # Remove old files
        files_to_remove = files[max_files:]
        for old_file in files_to_remove:
            try:
                old_file.unlink()
                logger.info(f"Cleaned up old file: {old_file}")
            except Exception as e:
                logger.warning(f"Failed to remove {old_file}: {e}")

        logger.info(f"Cleaned up {len(files_to_remove)} files from {folder_path}")

    except Exception as e:
        logger.warning(f"Failed to cleanup {folder_path}: {e}")

def cleanup_all_folders():
    """Clean up all monitored folders"""
    folders_to_cleanup = [
        DATA_DIR / "backups",
        Path("../outputs"),
        Path("../allie_finetuned"),
        Path("../allie-finetuned/checkpoint-100"),
        Path("../allie-finetuned/checkpoint-150"),
        Path("../scripts/logs") if Path("../scripts/logs").exists() else None,
        DATA_DIR  # Main data directory
    ]

    for folder in folders_to_cleanup:
        if folder and folder.exists():
            cleanup_folder(folder)
class AllieMemory:
    """Enhanced memory system for Allie"""
    def __init__(self, memory_file: Path):
        self.memory_file = memory_file
        self.knowledge_base = self.load_memory()
        self.conversation_summaries = []
        self.max_memories = 1000

    def load_memory(self) -> Dict[str, Any]:
        """Load persistent memory"""
        if self.memory_file.exists():
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {"facts": [], "preferences": {}, "learned_concepts": []}
        return {"facts": [], "preferences": {}, "learned_concepts": []}

    def save_memory(self):
        """Save memory to disk"""
        with open(self.memory_file, 'w', encoding='utf-8') as f:
            json.dump(self.knowledge_base, f, indent=2, ensure_ascii=False)
        # Clean up memory-related files
        cleanup_folder(self.memory_file.parent)

    def add_fact(self, fact: str, importance: float = 0.5, category: str = "general"):
        """Add an important fact to memory"""
        new_fact = {
            "fact": fact,
            "importance": importance,
            "category": category,
            "timestamp": datetime.now().isoformat(),
            "usage_count": 0
        }
        self.knowledge_base["facts"].append(new_fact)

        # Keep only most important facts
        self.knowledge_base["facts"].sort(key=lambda x: x["importance"] * (1 + x["usage_count"]), reverse=True)
        self.knowledge_base["facts"] = self.knowledge_base["facts"][:self.max_memories]

        self.save_memory()

    def recall_facts(self, query: str, limit: int = 5) -> List[str]:
        """Recall relevant facts based on query"""
        relevant_facts = []
        query_lower = query.lower()

        for fact in self.knowledge_base["facts"]:
            fact_text = fact["fact"].lower()
            # Simple keyword matching (could be improved with embeddings)
            if any(word in fact_text for word in query_lower.split()):
                relevant_facts.append(fact)
                fact["usage_count"] += 1

        # Sort by relevance and recency
        relevant_facts.sort(key=lambda x: (x["usage_count"], x["importance"]), reverse=True)
        self.save_memory()

        return [f["fact"] for f in relevant_facts[:limit]]

    def add_conversation_summary(self, summary: str, key_points: List[str]):
        """Add conversation summary"""
        summary_entry = {
            "summary": summary,
            "key_points": key_points,
            "timestamp": datetime.now().isoformat()
        }
        self.conversation_summaries.append(summary_entry)

        # Keep last 50 summaries
        self.conversation_summaries = self.conversation_summaries[-50:]

    def get_recent_context(self, limit: int = 3) -> str:
        """Get recent conversation context"""
        if not self.conversation_summaries:
            return ""

        recent = self.conversation_summaries[-limit:]
        context = "Recent conversation context:\n"
        for entry in recent:
            context += f"- {entry['summary']}\n"
            for point in entry['key_points'][:2]:  # Limit key points
                context += f"  • {point}\n"
        return context

from automatic_learner import AutomaticLearner

# Initialize memory system
MEMORY_FILE = DATA_DIR / "allie_memory.json"
allie_memory = AllieMemory(MEMORY_FILE)

# Initialize automatic learning system
auto_learner = AutomaticLearner(allie_memory)

# Initial cleanup on startup
cleanup_all_folders()
async def search_web(query: str) -> str:
    """Search the web using DuckDuckGo instant answers"""
    try:
        # Use DuckDuckGo instant answer API (no API key required)
        url = f"https://api.duckduckgo.com/?q={query}&format=json&no_html=1&skip_disambig=1"
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url)
            if response.status_code == 200:
                data = response.json()
                # Extract useful information
                result = ""
                if data.get("AbstractText"):
                    result += data["AbstractText"] + "\n"
                if data.get("Answer"):
                    result += data["Answer"] + "\n"
                if data.get("Definition"):
                    result += data["Definition"] + "\n"
                if not result and data.get("RelatedTopics"):
                    # Fallback to related topics
                    for topic in data["RelatedTopics"][:2]:  # Limit to first 2
                        if topic.get("Text"):
                            result += topic["Text"] + "\n"
                return result.strip() or f"I searched for '{query}' but couldn't find specific information."
            else:
                return f"Search failed for '{query}'."
    except Exception as e:
        logger.warning(f"Web search failed: {e}")
        return f"I couldn't access the internet to search for '{query}' right now."

@app.post("/api/conversations")
async def create_conversation_api(payload: Dict[str, Any] = Body(...)):
    """Create a new conversation (for UI sync)"""
    conv = payload
    if "id" not in conv:
        conv["id"] = f"c-{len(conversation_history)}"
    conversation_history.append(conv)
    # Save to backup
    with open(DATA_DIR / "backup.json", "w", encoding="utf-8") as f:
        json.dump(conversation_history, f, indent=2)
    return conv

@app.put("/api/conversations/{conv_id}")
async def update_conversation_api(conv_id: str, payload: Dict[str, Any] = Body(...)):
    """Update a conversation"""
    for i, conv in enumerate(conversation_history):
        if conv.get("id") == conv_id:
            old_conv = conversation_history[i]
            conversation_history[i] = payload

            # Process new messages for automatic learning
            if payload.get("messages"):
                old_messages = old_conv.get("messages", [])
                new_messages = payload["messages"]

                # Find new user messages
                for j, msg in enumerate(new_messages):
                    if j >= len(old_messages) or msg != old_messages[j]:
                        if msg.get("role") == "me" and msg.get("text"):
                            # Process user message for learning
                            learning_result = auto_learner.process_message(msg["text"], "user")
                            if learning_result["learning_actions"]:
                                logger.info(f"Learned {learning_result['total_facts_learned']} facts from user message")

            # Generate conversation summary if it has grown
            if payload.get("messages") and len(payload["messages"]) > len(old_conv.get("messages", [])):
                try:
                    summary = await generate_conversation_summary(payload)
                    key_points = extract_key_points(payload)
                    allie_memory.add_conversation_summary(summary, key_points)
                except Exception as e:
                    logger.warning(f"Failed to summarize conversation: {e}")

            # Save to backup
            with open(DATA_DIR / "backup.json", "w", encoding="utf-8") as f:
                json.dump(conversation_history, f, indent=2)
            return payload
    raise HTTPException(status_code=404, detail="Conversation not found")

async def generate_conversation_summary(conversation: Dict[str, Any]) -> str:
    """Generate a summary of the conversation"""
    messages = conversation.get("messages", [])
    if len(messages) < 4:  # Need at least a few exchanges
        return f"Conversation with {len(messages)} messages"

    # Simple summary based on first and last messages
    first_user = next((m["text"] for m in messages if m.get("role") == "me"), "Unknown topic")
    last_assistant = next((m["text"] for m in reversed(messages) if m.get("role") == "them"), "")

    if len(last_assistant) > 100:
        last_assistant = last_assistant[:100] + "..."

    return f"Discussion about: {first_user[:50]}... Result: {last_assistant}"

def extract_key_points(conversation: Dict[str, Any]) -> List[str]:
    """Extract key points from conversation"""
    messages = conversation.get("messages", [])
    key_points = []

    # Look for important information in responses
    for msg in messages:
        if msg.get("role") == "them" and len(msg.get("text", "")) > 20:
            text = msg["text"]
            # Extract sentences that might contain facts
            sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 10]
            key_points.extend(sentences[:2])  # Take first 2 substantial sentences

    return key_points[:5]  # Limit to 5 key points

@app.delete("/api/conversations/{conv_id}")
async def delete_conversation_api(conv_id: str):
    """Delete a conversation"""
    global conversation_history
    conversation_history = [c for c in conversation_history if c.get("id") != conv_id]
    # Save to backup
    with open(DATA_DIR / "backup.json", "w", encoding="utf-8") as f:
        json.dump(conversation_history, f, indent=2)
    return {"status": "deleted"}

# Keep the old generate endpoint as /api/generate
@app.post("/api/generate")
async def generate_response(payload: Dict[str, Any] = Body(...)):
    prompt = payload.get("prompt", "")
    max_tokens = payload.get("max_tokens", 200)
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")

    # Process user input for automatic learning
    learning_result = auto_learner.process_message(prompt, "user")
    learning_confirmation = auto_learner.generate_learning_response(learning_result["learning_actions"])

    # Get relevant memories and context
    relevant_facts = allie_memory.recall_facts(prompt)
    recent_context = allie_memory.get_recent_context()

    # Check if this query might need web search
    search_keywords = ["current", "today", "latest", "news", "weather", "price", "stock", "score", "result", "update", "now"]
    needs_search = any(keyword in prompt.lower() for keyword in search_keywords)

    search_results = ""
    if needs_search:
        search_results = await search_web(prompt)

    # Build enhanced context
    context_parts = []
    if relevant_facts:
        context_parts.append("Relevant information I remember:\n" + "\n".join(f"- {fact}" for fact in relevant_facts))
    if recent_context:
        context_parts.append(recent_context)
    if search_results:
        context_parts.append(f"Current web search results: {search_results}")

    enhanced_prompt = prompt
    if context_parts:
        enhanced_prompt = f"{prompt}\n\nContext:\n" + "\n\n".join(context_parts)

    # Format as chat message for TinyLlama with system prompt
    from datetime import datetime
    current_date = datetime.now().strftime("%B %d, %Y")
    system_content = f"""You are Allie, a helpful and friendly AI assistant. Today's date is {current_date}.

You have access to:
- Your long-term memory of important facts and information
- Recent conversation context
- Current web search results when needed

Use this information naturally in your responses. If you learn something new and important, remember it for future conversations."""

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": enhanced_prompt}
    ]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_length=inputs['input_ids'].shape[1] + max_tokens, do_sample=True, temperature=0.7, top_p=0.9, pad_token_id=tokenizer.eos_token_id)
    reply = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    # Process assistant response for additional learning
    assistant_learning = auto_learner.process_message(reply, "assistant")
    if assistant_learning["learning_actions"]:
        assistant_confirmation = auto_learner.generate_learning_response(assistant_learning["learning_actions"])
        reply += assistant_confirmation

    # Add learning confirmation to response
    final_reply = reply + learning_confirmation

    return {"text": final_reply}

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
        # Clean up backup folder
        cleanup_folder(DATA_DIR / "backups")
        return {"status": "backup successful"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/memory/add")
async def add_memory(payload: Dict[str, Any] = Body(...)):
    """Manually add a fact to Allie's memory"""
    fact = payload.get("fact", "")
    importance = payload.get("importance", 0.5)
    category = payload.get("category", "general")

    if not fact:
        raise HTTPException(status_code=400, detail="Fact is required")

    allie_memory.add_fact(fact, importance, category)
    return {"status": "fact_added", "fact": fact}

@app.get("/api/memory/recall")
async def recall_memory(query: str = "", limit: int = 5):
    """Recall relevant facts from memory"""
    if not query:
        # Return all facts if no query
        facts = allie_memory.knowledge_base.get("facts", [])[:limit]
        return {"facts": [f["fact"] for f in facts]}

    facts = allie_memory.recall_facts(query, limit)
    return {"query": query, "facts": facts}

@app.get("/api/memory/stats")
async def memory_stats():
    """Get memory statistics"""
    facts = allie_memory.knowledge_base.get("facts", [])
    return {
        "total_facts": len(facts),
        "categories": list(set(f["category"] for f in facts if "category" in f)),
        "most_used": sorted(facts, key=lambda x: x.get("usage_count", 0), reverse=True)[:3]
    }

@app.get("/api/learning/status")
async def get_learning_status():
    """Get current learning system status"""
    if not LEARNING_ENABLED:
        return {"enabled": False, "message": "Learning system not available"}

    try:
        # Import the orchestrator directly instead of using subprocess
        import sys
        scripts_dir = APP_ROOT.parent / "scripts"
        sys.path.insert(0, str(scripts_dir))

        from learning_orchestrator import IncrementalLearningOrchestrator

        # Create orchestrator instance and get status
        orchestrator = IncrementalLearningOrchestrator()
        status = orchestrator.get_status()

        return {
            "enabled": True,
            "is_active": status.get("is_active", False),
            "should_learn": status.get("learning_ready", False),
            "current_episode": status.get("current_episode"),
            "reason": "Ready for learning" if status.get("learning_ready") else "Learning conditions not met",
            "system_resources": status.get("system_resources", {}),
            "data_stats": status.get("data_stats", {})
        }

    except Exception as e:
        logger.error(f"Error getting learning status: {e}")
        return {"enabled": True, "error": str(e), "is_active": False, "should_learn": False}

@app.post("/api/learning/start")
async def start_learning_episode():
    """Start a new learning episode"""
    if not LEARNING_ENABLED:
        raise HTTPException(status_code=503, detail="Learning system not available")

    try:
        # Import the orchestrator directly
        import sys
        scripts_dir = APP_ROOT.parent / "scripts"
        sys.path.insert(0, str(scripts_dir))

        from learning_orchestrator import IncrementalLearningOrchestrator

        # Create orchestrator and check if learning should be triggered
        orchestrator = IncrementalLearningOrchestrator()
        should_learn, reason = orchestrator.should_trigger_learning()

        if not should_learn:
            raise HTTPException(status_code=400, detail=f"Learning conditions not met: {reason}")

        # Start learning episode
        episode_id = orchestrator.start_learning_episode()

        return {"episode_id": episode_id, "status": "started", "message": "Learning episode started successfully"}

    except HTTPException:
        raise
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
