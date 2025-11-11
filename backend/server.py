import os
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
import asyncio
import sys

import httpx
from fastapi import FastAPI, Body, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

# Ensure project root is on sys.path so `import backend.*` works when running via uvicorn
# Insert the parent directory of this backend package (the repository root)
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.sources.retrieval import search_with_memory_first, search_all_sources
from backend.sources.duckduckgo import search_duckduckgo
# Import lightweight context utils (pronoun resolution)
from backend.context_utils import enhance_query_with_context

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

# Cache for external API responses
_api_cache = {}
_CACHE_TTL_SECONDS = 300  # 5 minutes cache TTL

def _get_cache_key(query: str, api_type: str) -> str:
    """Generate a cache key for API responses"""
    import hashlib
    return hashlib.md5(f"{api_type}:{query}".encode()).hexdigest()

def _get_cached_response(cache_key: str) -> Optional[Dict[str, Any]]:
    """Get cached response if still valid"""
    if cache_key in _api_cache:
        cached_data, timestamp = _api_cache[cache_key]
        if datetime.now() - timestamp < timedelta(seconds=_CACHE_TTL_SECONDS):
            return cached_data
        else:
            # Remove expired cache entry
            del _api_cache[cache_key]
    return None

def _set_cached_response(cache_key: str, data: Dict[str, Any]):
    """Cache API response with timestamp"""
    _api_cache[cache_key] = (data, datetime.now())

    # Clean up old cache entries (keep cache size reasonable)
    if len(_api_cache) > 100:
        # Remove oldest entries
        sorted_entries = sorted(_api_cache.items(), key=lambda x: x[1][1])
        for old_key, _ in sorted_entries[:20]:  # Remove 20 oldest
            del _api_cache[old_key]

# Simple fact database for common questions
SIMPLE_FACTS = {
    "capital of france": "The capital of France is Paris.",
    "capital of france is": "The capital of France is Paris.",
    "what is the capital of france": "The capital of France is Paris.",
    "france capital": "The capital of France is Paris.",
    "paris": "Paris is the capital and most populous city of France.",
    "capital of the united states": "The capital of the United States is Washington, D.C.",
    "capital of usa": "The capital of the United States is Washington, D.C.",
    "washington dc": "Washington, D.C. is the capital of the United States.",
    "president of the united states": "As of my last knowledge, Joe Biden is the President of the United States.",
    "who is the president": "As of my last knowledge, Joe Biden is the President of the United States.",
    "current president": "As of my last knowledge, Joe Biden is the President of the United States.",
}

# Global model and tokenizer variables
tokenizer = None
model = None

# Load real model for chat
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    
    logger.info("Loading TinyLlama model...")
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load model with appropriate device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        low_cpu_mem_usage=True
    )
    
    if device == "cpu":
        model = model.to(device)
    
    logger.info("TinyLlama model loaded successfully")
    
except Exception as e:
    logger.warning(f"Failed to load real model, using dummy model: {e}")
    # Dummy model for testing (fallback)
    class DummyModel:
        def __init__(self):
            self.device = "cpu"
        
        def generate(self, **kwargs):
            # Return tensor-like object with shape attribute
            class DummyOutput:
                def __init__(self):
                    self.shape = [1, 10]  # batch_size=1, seq_len=10
                def __getitem__(self, idx):
                    return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Dummy token IDs
            
            return [DummyOutput()]

    class DummyTokenizer:
        def __init__(self):
            self.eos_token_id = 2
        
        def __call__(self, prompt, return_tensors=None):
            # Return object with .to() method to match real tokenizer behavior
            class DummyTensor:
                """Mock tensor with shape attribute"""
                def __init__(self, data):
                    self.data = data
                    # Calculate shape: (batch_size, sequence_length)
                    if isinstance(data, list) and len(data) > 0:
                        self.shape = (len(data), len(data[0]) if isinstance(data[0], list) else 1)
                    else:
                        self.shape = (1, 1)
                
                def __getitem__(self, key):
                    return self.data[key]
            
            class TokenizerOutput:
                def __init__(self):
                    self.data = {"input_ids": DummyTensor([[1, 2, 3]])}
                
                def to(self, device):
                    # Return self to support chaining
                    return self
                
                def __getitem__(self, key):
                    return self.data[key]
                
                def keys(self):
                    return self.data.keys()
                
                def values(self):
                    return self.data.values()
                
                def items(self):
                    return self.data.items()
            
            output = TokenizerOutput()
            # Make it subscriptable like a dict and add tensor as attribute
            output.input_ids = output.data["input_ids"]
            return output
        
        def decode(self, tokens, skip_special_tokens=None):
            return "This is a dummy response for testing purposes."
        
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            # Simple template for testing
            return "Test prompt"

    tokenizer = DummyTokenizer()
    model = DummyModel()

# Import learning orchestrator
try:
    import subprocess
    import sys
    LEARNING_ENABLED = True
    logger.info("Learning system enabled")
except Exception as e:
    logger.warning(f"Learning system not available: {e}")
    LEARNING_ENABLED = False



# -------------------------

from contextlib import asynccontextmanager
from fastapi import FastAPI

# Background task for automatic learning
_auto_learning_task = None
_last_learning_check = datetime.now()
_last_learning_trigger = None
_learning_cooldown_minutes = 3  # Reduced from 5 to 3 minutes for more frequent learning cycles

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown"""
    # Startup
    logger.info("Starting up Allie server...")
    
    try:
        # The initialization code that was in startup_event
        if LEARNING_ENABLED:
            # Background learning is handled by the separate reconciliation_worker.py script
            # which can be scheduled to run periodically via Windows Task Scheduler
            logger.info("Learning system enabled - reconciliation handled by external worker")
        
        logger.info("Allie server startup complete")
        yield
        
    except Exception as e:
        logger.error(f"Error during server operation: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise
    finally:
        # Shutdown
        logger.info("Shutting down Allie server...")
        
        try:
            # Close HTTP client
            global _http_client
            if _http_client is not None:
                await _http_client.aclose()
                _http_client = None
                logger.info("httpx client closed")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

# FastAPI app
app = FastAPI(title="Allie", lifespan=lifespan)

# mount static files and ui route
if not STATIC_DIR.exists():
    STATIC_DIR.mkdir(parents=True, exist_ok=True)

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
    if not fact_check_path.exists():
        return HTMLResponse("<html><body><h3>Fact Check UI not installed</h3></body></html>", status_code=404)
    return HTMLResponse(fact_check_path.read_text(encoding="utf-8"))


@app.get("/quick-teach", response_class=HTMLResponse)
async def quick_teach_ui():
    """Serve the Quick Teach UI page"""
    quick_path = STATIC_DIR / "quick-teach.html"
    if not quick_path.exists():
        return HTMLResponse("<html><body><h3>Quick Teach UI not installed</h3></body></html>", status_code=404)
    return HTMLResponse(quick_path.read_text(encoding="utf-8"))

@app.get("/admin-dashboard", response_class=HTMLResponse)
async def admin_dashboard_ui():
    """Serve the Admin Dashboard page"""
    admin_path = STATIC_DIR / "admin-dashboard.html"
    if not admin_path.exists():
        return HTMLResponse("<html><body><h3>Admin Dashboard not installed</h3></body></html>", status_code=404)
    return HTMLResponse(admin_path.read_text(encoding="utf-8"))

@app.get("/learning-log", response_class=HTMLResponse)
async def learning_log_ui():
    """Serve the Learning Log UI page"""
    log_path = STATIC_DIR / "learning-log.html"
    if not log_path.exists():
        return HTMLResponse("<html><body><h3>Learning Log UI not found</h3></body></html>", status_code=404)
    return HTMLResponse(log_path.read_text(encoding="utf-8"))

@app.get("/kb", response_class=HTMLResponse)
async def kb_ui():
    """Serve the Knowledge Base UI page"""
    kb_path = STATIC_DIR / "kb.html"
    if not kb_path.exists():
        return HTMLResponse("<html><body><h3>Knowledge Base UI not found</h3></body></html>", status_code=404)
    return HTMLResponse(kb_path.read_text(encoding="utf-8"))

@app.post("/api/admin/run-reconciliation")
async def run_reconciliation_admin():
    """Run reconciliation worker from admin dashboard"""
    try:
        import subprocess
        import sys
        from pathlib import Path

        # Path to the reconciliation worker script
        script_dir = Path(__file__).parent.parent / "scripts"
        worker_script = script_dir / "reconciliation_worker.py"

        if not worker_script.exists():
            raise HTTPException(status_code=500, detail="Reconciliation worker script not found")

        # Run the worker with --once flag
        result = subprocess.run([
            sys.executable, str(worker_script), "--once"
        ], capture_output=True, text=True, timeout=300)  # 5 minute timeout

        if result.returncode == 0:
            # Try to parse the output to get the number of items processed
            processed_count = 0
            for line in result.stdout.split('\n'):
                if 'Batch completed:' in line and 'successful' in line:
                    # Extract number from "Batch completed: X successful"
                    try:
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if part == 'completed:':
                                processed_count = int(parts[i+1])
                                break
                    except (ValueError, IndexError):
                        pass

            return {
                "success": True,
                "message": "Reconciliation completed successfully",
                "processed_count": processed_count,
                "output": result.stdout[-500:]  # Last 500 chars of output
            }
        else:
            return {
                "success": False,
                "error": f"Reconciliation failed with exit code {result.returncode}",
                "output": result.stderr[-500:] if result.stderr else result.stdout[-500:]
            }

    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=500, detail="Reconciliation timed out after 5 minutes")
    except Exception as e:
        logger.error(f"Error running reconciliation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/learning/queue/stats")
async def get_learning_queue_stats():
    """Get statistics about the learning queue"""
    try:
        # Query the learning_queue table
        cursor = None
        connection = None
        try:
            import mysql.connector
            connection = mysql.connector.connect(
                host='localhost',
                user='allie',
                password='StrongPassword123!',
                database='allie_memory'
            )
            cursor = connection.cursor()

            # Get pending items count
            cursor.execute("SELECT COUNT(*) FROM learning_queue WHERE status='pending'")
            pending = cursor.fetchone()[0]

            # Get processed items count (processed today)
            cursor.execute("""
                SELECT COUNT(*) FROM learning_queue
                WHERE status='processed'
                AND DATE(processed_at) = CURDATE()
            """)
            processed_today = cursor.fetchone()[0]

            return {
                "pending": pending,
                "processed_today": processed_today,
                "total": pending + processed_today
            }

        except mysql.connector.Error as e:
            logger.error(f"Database error getting queue stats: {e}")
            # Return default values if database query fails
            return {
                "pending": 0,
                "processed_today": 0,
                "total": 0,
                "error": str(e)
            }
        finally:
            if cursor:
                cursor.close()
            if connection:
                connection.close()

    except Exception as e:
        logger.error(f"Error getting learning queue stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
    def __init__(self, memory_file):
        # Accept either a Path or a string path
        try:
            self.memory_file = Path(memory_file)
        except Exception:
            self.memory_file = Path(str(memory_file))
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

    def remove_fact(self, fact_text: str) -> bool:
        """Remove a fact from memory by exact text match"""
        original_count = len(self.knowledge_base["facts"])
        self.knowledge_base["facts"] = [
            f for f in self.knowledge_base["facts"] 
            if f["fact"].strip().lower() != fact_text.strip().lower()
        ]
        removed = len(self.knowledge_base["facts"]) < original_count
        if removed:
            self.save_memory()
        return removed

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

from backend.automatic_learner import AutomaticLearner

def extract_president_name(text):
    """Extract president names from text"""
    words = text.split()
    
    # First, try to find known president names anywhere in the text
    known_presidents = ["biden", "trump", "obama", "bush", "clinton", "reagan", "carter", "ford", "nixon", "johnson", "kennedy"]
    for word in words:
        if word.lower() in known_presidents:
            return word
    
    # If no known presidents found, look for capitalized words near "president"
    for i, word in enumerate(words):
        if "president" in word.lower():
            # Look for capitalized words in a wider range around the keyword
            candidates = []
            search_range = range(max(0, i - 8), min(len(words), i + 6))
            
            for j in search_range:
                if j == i:  # Skip the keyword itself
                    continue
                word_candidate = words[j]
                if (word_candidate[0].isupper() and
                    word_candidate.lower() not in ["the", "of", "and", "or", "but", "in", "on", "at", "to", "for", "with", "by", "states", "united", "is", "are", "was", "were", "has", "have"]):
                    candidates.append(word_candidate)
            
            return candidates[-1] if candidates else None
    
    return None

def extract_number_after_keyword(text, keyword):
    """Extract numbers after keywords"""
    import re
    text_lower = text.lower()
    
    # Find keyword position
    keyword_pos = text_lower.find(keyword)
    if keyword_pos == -1:
        return None
    
    # Extract text after keyword
    after_keyword = text[keyword_pos + len(keyword):]
    
    # Find first number
    numbers = re.findall(r'\d+(?:,\d+)*(?:\.\d+)?', after_keyword)
    return numbers[0] if numbers else None

def extract_info_after_keyword(text, keyword):
    """Extract information after keywords"""
    text_lower = text.lower()
    words = text.split()
    
    for i, word in enumerate(words):
        if keyword in word.lower():
            # Extract next few significant words
            result = []
            for j in range(i + 1, min(i + 4, len(words))):
                if words[j] not in ["the", "a", "an", "in", "on", "at"]:
                    result.append(words[j])
                    if len(result) >= 2:  # Get up to 2 significant words
                        break
            return " ".join(result) if result else None
    return None

# Initialize advanced memory system
sys.path.insert(0, str(APP_ROOT.parent / "advanced-memory"))
from db import AllieMemoryDB
from learning_pipeline import LearningPipeline
from hybrid import HybridMemory as AdvancedHybridMemory

advanced_memory = AllieMemoryDB()
learning_pipeline = LearningPipeline(advanced_memory)

# Use hybrid memory from advanced-memory (new version)
hybrid_memory = AdvancedHybridMemory()

# Initialize legacy memory system for backward compatibility
MEMORY_FILE = DATA_DIR / "allie_memory.json"
allie_memory = AllieMemory(MEMORY_FILE)

# Initialize automatic learner with advanced memory
auto_learner = AutomaticLearner(allie_memory, advanced_memory, learning_queue=advanced_memory)

# Initial cleanup on startup
cleanup_all_folders()

# -------------------------
# Auto-learning helper function
# -------------------------
async def check_and_trigger_auto_learning():
    """
    Check if conditions are met for triggering an auto-learning episode.
    This function is called when user feedback indicates a fact needs review.
    """
    global _last_learning_check, _last_learning_trigger
    
    try:
        current_time = datetime.now()
        
        # Check cooldown period
        if _last_learning_trigger is not None:
            time_since_last = (current_time - _last_learning_trigger).total_seconds() / 60
            if time_since_last < _learning_cooldown_minutes:
                logger.debug(f"Auto-learning on cooldown. {_learning_cooldown_minutes - time_since_last:.1f} minutes remaining")
                return
        
        # Check if we have enough data to justify learning
        stats = hybrid_memory.get_statistics()
        total_facts = stats.get("total_facts", 0)
        
        # Need at least 10 facts to trigger learning
        if total_facts < 10:
            logger.debug(f"Not enough facts for auto-learning (have {total_facts}, need 10)")
            return
        
        # Check for negative examples or feedback reports
        neg_examples_file = DATA_DIR / "negative_examples.jsonl"
        feedback_file = DATA_DIR / "feedback_reports.jsonl"
        
        has_feedback = (
            (neg_examples_file.exists() and neg_examples_file.stat().st_size > 0) or
            (feedback_file.exists() and feedback_file.stat().st_size > 0)
        )
        
        if not has_feedback:
            logger.debug("No feedback data to trigger auto-learning")
            return
        
        # Conditions met - log that we would trigger learning
        logger.info(f"Auto-learning conditions met: {total_facts} facts, feedback available")
        logger.info("Auto-learning task is currently disabled for testing")
        
        # Update tracking
        _last_learning_check = current_time
        _last_learning_trigger = current_time
        
        # In the future, this would trigger actual learning:
        # await auto_learner.run_learning_episode()
        
    except Exception as e:
        logger.error(f"Error in check_and_trigger_auto_learning: {e}")

async def search_web(query: str) -> Dict[str, Any]:
    """Search the web using DuckDuckGo instant answers"""
    # Check cache first
    cache_key = _get_cache_key(query, "web")
    cached_result = _get_cached_response(cache_key)
    if cached_result:
        logger.info(f"Web search cache hit for: {query}")
        return cached_result

    logger.info(f"Web search cache miss for: {query}")
    try:
        # Use DuckDuckGo instant answer API (no API key required)
        url = f"https://api.duckduckgo.com/?q={query}&format=json&no_html=1&skip_disambig=1"
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url)
            if response.status_code == 200:
                data = response.json()
                results = []
                
                # Extract instant answer
                if data.get("Answer"):
                    results.append({
                        "title": "Instant Answer",
                        "text": data["Answer"],
                        "source": "DuckDuckGo Instant Answer"
                    })
                
                # Extract abstract
                if data.get("AbstractText"):
                    results.append({
                        "title": data.get("Heading", "Abstract"),
                        "text": data["AbstractText"],
                        "source": "DuckDuckGo Abstract"
                    })
                
                # Extract definition
                if data.get("Definition"):
                    results.append({
                        "title": "Definition",
                        "text": data["Definition"],
                        "source": "DuckDuckGo Definition"
                    })
                
                # Extract related topics (limit to 3-5)
                if not results and data.get("RelatedTopics"):
                    for topic in data["RelatedTopics"][:5]:
                        if topic.get("Text"):
                            results.append({
                                "title": topic.get("FirstURL", "Related Topic"),
                                "text": topic["Text"],
                                "source": "DuckDuckGo Related Topics"
                            })
                
                result = {
                    "query": query,
                    "results": results[:5],  # Limit to top 5
                    "success": True
                }
                _set_cached_response(cache_key, result)
                return result
            else:
                result = {
                    "query": query,
                    "results": [],
                    "success": False,
                    "error": f"Search failed with status {response.status_code}"
                }
                _set_cached_response(cache_key, result)
                return result
    except Exception as e:
        logger.warning(f"Web search failed: {e}")
        result = {
            "query": query,
            "results": [],
            "success": False,
            "error": str(e)
        }
        _set_cached_response(cache_key, result)
        return result

async def search_wikidata(query: str) -> Dict[str, Any]:
    """Search Wikidata for structured factual data"""
    # Check cache first
    cache_key = _get_cache_key(query, "wikidata")
    cached_result = _get_cached_response(cache_key)
    if cached_result:
        logger.info(f"Wikidata search cache hit for: {query}")
        return cached_result

    logger.info(f"Wikidata search cache miss for: {query}")
    try:
        # Use Wikidata API for structured data
        # First, search for entities
        search_url = f"https://www.wikidata.org/w/api.php?action=wbsearchentities&search={query}&language=en&format=json&limit=5"
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(search_url)
            if response.status_code == 200:
                data = response.json()
                results = []
                
                if data.get("search"):
                    for entity in data["search"][:3]:  # Limit to top 3
                        entity_id = entity.get("id")
                        if entity_id:
                            # Get detailed information for each entity
                            detail_url = f"https://www.wikidata.org/wiki/Special:EntityData/{entity_id}.json"
                            detail_response = await client.get(detail_url)
                            if detail_response.status_code == 200:
                                detail_data = detail_response.json()
                                entities = detail_data.get("entities", {})
                                if entity_id in entities:
                                    entity_data = entities[entity_id]
                                    claims = entity_data.get("claims", {})
                                    
                                    # Extract key properties
                                    extracted_info = {
                                        "label": entity.get("label", ""),
                                        "description": entity.get("description", ""),
                                        "id": entity_id,
                                        "properties": {}
                                    }
                                    
                                    # Extract common properties
                                    property_map = {
                                        "P17": "country",  # country
                                        "P19": "place_of_birth",  # place of birth
                                        "P20": "place_of_death",  # place of death
                                        "P27": "citizenship",  # citizenship
                                        "P31": "instance_of",  # instance of
                                        "P36": "capital",  # capital
                                        "P50": "author",  # author
                                        "P57": "director",  # director
                                        "P69": "educated_at",  # educated at
                                        "P106": "occupation",  # occupation
                                        "P108": "employer",  # employer
                                        "P119": "place_of_burial",  # place of burial
                                        "P131": "located_in",  # located in
                                        "P136": "genre",  # genre
                                        "P144": "based_on",  # based on
                                        "P161": "cast_member",  # cast member
                                        "P166": "award_received",  # award received
                                        "P569": "date_of_birth",  # date of birth
                                        "P570": "date_of_death",  # date of death
                                        "P577": "publication_date",  # publication date
                                        "P580": "start_time",  # start time
                                        "P582": "end_time",  # end time
                                        "P625": "coordinate_location",  # coordinate location
                                        "P856": "official_website",  # official website
                                        "P910": "topic_main_category",  # topic's main category
                                        "P921": "main_subject",  # main subject
                                        "P973": "described_at_url",  # described at URL
                                        "P1014": "Getty_AAT_ID",  # Art & Architecture Thesaurus ID
                                        "P214": "VIAF_ID",  # VIAF ID
                                        "P244": "LCAuth_ID",  # Library of Congress authority ID
                                        "P345": "IMDb_ID",  # IMDb ID
                                        "P496": "ORCID_iD",  # ORCID iD
                                        "P818": "arXiv_ID",  # arXiv ID
                                        "P846": "GBIF_taxon_ID",  # GBIF taxon ID
                                        "P850": "WoRMS_ID",  # World Register of Marine Species ID
                                    }
                                    
                                    for prop_id, prop_name in property_map.items():
                                        if prop_id in claims:
                                            prop_claims = claims[prop_id]
                                            if prop_claims:
                                                # Take the first/main value
                                                main_value = prop_claims[0].get("mainsnak", {}).get("datavalue", {})
                                                if main_value:
                                                    if main_value.get("type") == "string":
                                                        extracted_info["properties"][prop_name] = main_value.get("value", "")
                                                    elif main_value.get("type") == "wikibase-entityid":
                                                        entity_id = main_value.get("value", {}).get("id")
                                                        if entity_id:
                                                            # Try to get the label for this entity
                                                            extracted_info["properties"][prop_name] = f"Q{entity_id}"
                                    
                                    results.append(extracted_info)
                
                result = {
                    "query": query,
                    "results": results,
                    "success": True
                }
                _set_cached_response(cache_key, result)
                return result
            else:
                result = {
                    "query": query,
                    "results": [],
                    "success": False,
                    "error": f"Wikidata search failed with status {response.status_code}"
                }
                _set_cached_response(cache_key, result)
                return result
    except Exception as e:
        logger.warning(f"Wikidata search failed: {e}")
        result = {
            "query": query,
            "results": [],
            "success": False,
            "error": str(e)
        }
        _set_cached_response(cache_key, result)
        return result

async def search_dbpedia(query: str) -> Dict[str, Any]:
    """Search DBpedia for structured factual data"""
    # Check cache first
    cache_key = _get_cache_key(query, "dbpedia")
    cached_result = _get_cached_response(cache_key)
    if cached_result:
        logger.info(f"DBpedia search cache hit for: {query}")
        return cached_result

    logger.info(f"DBpedia search cache miss for: {query}")
    try:
        # Use DBpedia Spotlight for entity extraction and linking
        spotlight_url = f"https://api.dbpedia-spotlight.org/en/annotate?text={query}&confidence=0.5&support=20"
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(spotlight_url, headers={"Accept": "application/json"})
            if response.status_code == 200:
                data = response.json()
                results = []
                
                if "Resources" in data:
                    for resource in data["Resources"][:3]:  # Limit to top 3
                        # Get additional information from DBpedia
                        dbpedia_uri = resource.get("@URI", "")
                        if dbpedia_uri:
                            # Extract DBpedia resource name
                            resource_name = dbpedia_uri.split("/")[-1]
                            
                            # Query for basic properties
                            sparql_url = "https://dbpedia.org/sparql"
                            sparql_query = f"""
                            SELECT ?property ?value WHERE {{
                                <{dbpedia_uri}> ?property ?value .
                                FILTER(LANG(?value) = "" || LANGMATCHES(LANG(?value), "en"))
                            }} LIMIT 20
                            """
                            
                            sparql_response = await client.get(
                                sparql_url, 
                                params={"query": sparql_query, "format": "json"}
                            )
                            
                            properties = {}
                            if sparql_response.status_code == 200:
                                sparql_data = sparql_response.json()
                                bindings = sparql_data.get("results", {}).get("bindings", [])
                                
                                for binding in bindings:
                                    prop_uri = binding.get("property", {}).get("value", "")
                                    value = binding.get("value", {}).get("value", "")
                                    
                                    if prop_uri and value:
                                        # Extract property name from URI
                                        prop_name = prop_uri.split("/")[-1] if "/" in prop_uri else prop_uri
                                        prop_name = prop_name.split("#")[-1] if "#" in prop_name else prop_name
                                        
                                        # Clean up common properties
                                        if "birthDate" in prop_name or "birth_date" in prop_name:
                                            properties["birth_date"] = value
                                        elif "deathDate" in prop_name or "death_date" in prop_name:
                                            properties["death_date"] = value
                                        elif "birthPlace" in prop_name or "birth_place" in prop_name:
                                            properties["birth_place"] = value
                                        elif "deathPlace" in prop_name or "death_place" in prop_name:
                                            properties["death_place"] = value
                                        elif "occupation" in prop_name:
                                            properties["occupation"] = value
                                        elif "country" in prop_name:
                                            properties["country"] = value
                                        elif "capital" in prop_name:
                                            properties["capital"] = value
                            
                            result = {
                                "label": resource.get("@surfaceForm", ""),
                                "uri": dbpedia_uri,
                                "confidence": float(resource.get("@similarityScore", 0)),
                                "support": int(resource.get("@support", 0)),
                                "types": resource.get("@types", "").split(",") if resource.get("@types") else [],
                                "properties": properties
                            }
                            results.append(result)
                
                result = {
                    "query": query,
                    "results": results,
                    "success": True
                }
                _set_cached_response(cache_key, result)
                return result
            else:
                result = {
                    "query": query,
                    "results": [],
                    "success": False,
                    "error": f"DBpedia search failed with status {response.status_code}"
                }
                _set_cached_response(cache_key, result)
                return result
    except Exception as e:
        logger.warning(f"DBpedia search failed: {e}")
        result = {
            "query": query,
            "results": [],
            "success": False,
            "error": str(e)
        }
        _set_cached_response(cache_key, result)
        return result

# DEPRECATED: Wikipedia API now returns 403 errors
# Replaced with multi-source retrieval system
async def search_wikipedia(query: str) -> Dict[str, Any]:
    """
    Legacy wrapper for Wikipedia search - now uses multi-source retrieval
    Returns results in Wikipedia-compatible format for backward compatibility
    """
    logger.warning(f"search_wikipedia called for '{query}' - using multi-source retrieval instead")
    
    # Use new retrieval system
    try:
        # Quick search using DuckDuckGo only (fastest)
        ddg_result = await search_duckduckgo(query, max_results=3)
        
        if ddg_result.get("success") and ddg_result.get("results"):
            # Format as Wikipedia-compatible response
            first_result = ddg_result["results"][0]
            text = first_result.get("text", "")
            
            return {
                "query": query,
                "title": first_result.get("title", query),
                "summary": text,
                "url": first_result.get("url", ""),
                "success": True,
                "source": "duckduckgo"  # Note the real source
            }
        else:
            return {
                "query": query,
                "title": query,
                "summary": "",
                "url": "",
                "success": False,
                "error": "No results found",
                "source": "duckduckgo"
            }
    except Exception as e:
        logger.error(f"Legacy Wikipedia wrapper error: {e}")
        return {
            "query": query,
            "title": query,
            "summary": "",
            "url": "",
            "success": False,
            "error": str(e)
        }

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
    global conversation_history
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

    # If conversation doesn't exist, create it
    conversation_history.append(payload)

    # Process messages for automatic learning on new conversation
    if payload.get("messages"):
        for msg in payload["messages"]:
            if msg.get("role") == "me" and msg.get("text"):
                learning_result = auto_learner.process_message(msg["text"], "user")
                if learning_result["learning_actions"]:
                    logger.info(f"Learned {learning_result['total_facts_learned']} facts from user message")

    # Generate conversation summary for new conversation
    if payload.get("messages") and len(payload["messages"]) > 0:
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
    """Extract factual key points from conversation"""
    messages = conversation.get("messages", [])
    key_points = []

    # Look for important information in responses, but filter out non-factual content
    for msg in messages:
        if msg.get("role") == "them" and len(msg.get("text", "")) > 20:
            text = msg["text"]
            
            # Skip responses that appear to be rephrasing or transformation tasks
            skip_indicators = [
                "rephrased as follows",
                "assuming the given context",
                "can be rephrased",
                "here are the revised responses",
                "here's how you can use this information",
                "user asked:",
                "allie responded:"
            ]
            
            if any(indicator in text.lower() for indicator in skip_indicators):
                continue
            
            # Extract sentences that might contain facts, but filter out obvious non-facts
            sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 10]
            
            for sentence in sentences[:2]:  # Take first 2 substantial sentences
                # Skip sentences that are clearly not factual
                non_factual_indicators = [
                    "here are the",
                    "i can help",
                    "let me",
                    "certainly",
                    "sure thing",
                    "as you mentioned",
                    "that's great",
                    "i understand",
                    "would you like",
                    "can you",
                    "do you"
                ]
                
                if not any(indicator in sentence.lower() for indicator in non_factual_indicators):
                    # Only include sentences that look like they contain factual information
                    if any(word in sentence.lower() for word in ["is", "are", "was", "were", "has", "have", "located", "born", "created", "developed", "invented"]):
                        key_points.append(sentence)

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

# pronoun resolution lives in backend/context_utils.py

# Keep the old generate endpoint as /api/generate
@app.post("/api/generate")
async def generate_response(payload: Dict[str, Any] = Body(...)):
    global tokenizer, model  # Declare global variables
    
    prompt = payload.get("prompt", "")
    max_tokens = payload.get("max_tokens", 512)  # Increased from 200 to allow fuller responses
    conversation_context = payload.get("conversation_context", [])  # New parameter

    # Input validation
    if not prompt or not isinstance(prompt, str):
        raise HTTPException(status_code=400, detail="Prompt is required and must be a string")

    if len(prompt) > 2000:
        raise HTTPException(status_code=400, detail="Prompt too long (maximum 2000 characters)")

    if len(prompt.strip()) == 0:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty or only whitespace")

    # Basic content filtering - reject potentially harmful prompts
    harmful_patterns = [
        "how to hack", "how to exploit", "how to crack", "password cracking",
        "sql injection", "xss attack", "ddos", "malware", "virus", "trojan",
        "illegal", "drugs", "weapons", "bomb", "explosive"
    ]

    prompt_lower = prompt.lower()
    for pattern in harmful_patterns:
        if pattern in prompt_lower:
            raise HTTPException(status_code=400, detail="Request contains inappropriate content")

    if not isinstance(max_tokens, int) or max_tokens < 1 or max_tokens > 1000:
        raise HTTPException(status_code=400, detail="max_tokens must be an integer between 1 and 1000")

    # Check simple facts database first for common questions
    prompt_lower = prompt.lower().strip()
    
    # Handle "show memory timeline" command
    if "show memory timeline" in prompt_lower or "memory timeline" in prompt_lower:
        timeline = hybrid_memory.get_timeline(include_outdated=True)
        if not timeline:
            return {"text": "My memory timeline is empty. I haven't learned any facts yet!"}
        
        response_parts = ["Here's my complete memory timeline:\n"]
        for i, fact_dict in enumerate(timeline, 1):
            timestamp = fact_dict["timestamp"][:16].replace("T", " ")  # Format: YYYY-MM-DD HH:MM
            outdated_marker = " [OUTDATED]" if fact_dict["is_outdated"] else ""
            response_parts.append(f"{i}. [{timestamp}] [{fact_dict['category']}] {fact_dict['fact']}{outdated_marker}")
        
        stats = hybrid_memory.get_statistics()
        response_parts.append(f"\n📊 Total: {stats['total_facts']} facts, {stats['active_facts']} active, {stats['outdated_facts']} outdated")
        
        return {"text": "\n".join(response_parts)}
    
    # Handle "memory statistics" command
    if "memory statistics" in prompt_lower or "memory stats" in prompt_lower:
        stats = hybrid_memory.get_statistics()
        response_parts = [
            "📊 Hybrid Memory Statistics:",
            f"Total Facts: {stats['total_facts']}",
            f"Active Facts: {stats['active_facts']}",
            f"Outdated Facts: {stats['outdated_facts']}",
            f"Indexed Keywords: {stats['indexed_keywords']}",
            "\nCategories:"
        ]
        for category, count in stats['categories'].items():
            response_parts.append(f"  • {category}: {count}")
        response_parts.append("\nSources:")
        for source, count in stats['sources'].items():
            response_parts.append(f"  • {source}: {count}")
        
        return {"text": "\n".join(response_parts)}
    
    for key, response in SIMPLE_FACTS.items():
        if key in prompt_lower:
            return {"text": response}
    
    # Handle simple greetings
    greetings = ["hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening"]
    if prompt_lower in greetings or any(prompt_lower.startswith(g + " ") for g in greetings):
        return {"text": "Hello! I'm Allie, your AI assistant. How can I help you today?"}
    
    # Handle simple math questions
    math_patterns = [
        (r"what\s+(?:is|are|does)\s+(\d+\.?\d*)\s*\+\s*(\d+\.?\d*)", lambda m: float(m.group(1)) + float(m.group(2))),
        (r"(\d+\.?\d*)\s*\+\s*(\d+\.?\d*)", lambda m: float(m.group(1)) + float(m.group(2))),
        (r"what\s+(?:is|are|does)\s+(\d+\.?\d*)\s*-\s*(\d+\.?\d*)", lambda m: float(m.group(1)) - float(m.group(2))),
        (r"(\d+\.?\d*)\s*-\s*(\d+\.?\d*)", lambda m: float(m.group(1)) - float(m.group(2))),
        (r"what\s+(?:is|are|does)\s+(\d+\.?\d*)\s*(?:\*|x)\s*(\d+\.?\d*)", lambda m: float(m.group(1)) * float(m.group(2))),
        (r"(\d+\.?\d*)\s*(?:\*|x)\s*(\d+\.?\d*)", lambda m: float(m.group(1)) * float(m.group(2))),
        (r"what\s+(?:is|are|does)\s+(\d+\.?\d*)\s*/\s*(\d+\.?\d*)", lambda m: float(m.group(1)) / float(m.group(2))),
        (r"(\d+\.?\d*)\s*/\s*(\d+\.?\d*)", lambda m: float(m.group(1)) / float(m.group(2))),
    ]
    
    import re
    for pattern, calculator in math_patterns:
        match = re.search(pattern, prompt_lower)
        if match:
            try:
                result = calculator(match)
                # Format the result nicely (remove unnecessary decimals)
                if result == int(result):
                    return {"text": str(int(result))}
                else:
                    return {"text": f"{result:.4f}".rstrip('0').rstrip('.')}
            except:
                pass

    # Step 1: Process user input for automatic learning
    learning_result = auto_learner.process_message(prompt, "user")
    learning_confirmation = auto_learner.generate_learning_response(learning_result["learning_actions"])

    # Step 2: Check memory for relevant facts (use hybrid memory first, fallback to legacy)
    # First, enhance the search query with conversation context if pronouns are detected
    enhanced_query = enhance_query_with_context(prompt, conversation_context)
    logger.debug(f"Augmented memory search query: '{enhanced_query}' (original: '{prompt}')")
    hybrid_search_result = hybrid_memory.search(enhanced_query, limit=5, include_disambiguation=True)
    hybrid_results = hybrid_search_result.get("results", [])
    disambiguation = hybrid_search_result.get("disambiguation")
    fact_check_warnings = hybrid_search_result.get("fact_check_warnings", [])

    # Only use hybrid memory - no fallback to legacy memory to avoid pollution
    relevant_facts = [result["fact"] for result in hybrid_results] if hybrid_results else []
    recent_context = None  # Disabled to avoid pollution
    
    # Log hybrid memory usage
    if hybrid_results:
        logger.info(f"Hybrid memory: Found {len(hybrid_results)} relevant facts")
        # Log top hits for debugging
        try:
            top_facts = [res.get('fact','')[:200] for res in hybrid_results[:3]]
            logger.debug(f"Hybrid memory top hits: {top_facts}")
        except Exception:
            pass
        for result in hybrid_results:
            logger.debug(f"  - [{result['category']}] {result['fact']} (confidence: {result['confidence']})")
    
    logger.info(f"relevant_facts list has {len(relevant_facts)} items")

    # Step 2.1: Check for self-referential questions that shouldn't trigger external searches
    self_referential_patterns = [
        "what is your name", "who are you", "what are you", "tell me about yourself",
        "what's your name", "who is this", "introduce yourself", "what do you do",
        "what is your purpose", "what are you called", "what should i call you",
        "did you learn", "have you learned", "what did you learn", "did you remember",
        "do you remember", "what do you remember", "can you learn", "are you learning"
    ]
    
    is_self_referential = any(pattern in prompt.lower() for pattern in self_referential_patterns)

    # Step 2.5: Validate memory facts against Wikipedia if we have stored facts
    # DISABLED - This was causing unnecessary web searches even when memory had good coverage
    memory_validation_updates = []
    # if relevant_facts and not is_self_referential:
    #     # Search Wikipedia to validate stored facts
    #     validation_wiki = await search_wikipedia(prompt)
    #     if validation_wiki and validation_wiki.get("success") and validation_wiki.get("summary"):
    #         wiki_text = validation_wiki["summary"]
    #         
    #         # Compare each memory fact with Wikipedia content
    #         for fact in relevant_facts:
    #             # Simple conflict detection: if fact contradicts Wikipedia
    #             fact_lower = fact.lower()
    #             wiki_lower = wiki_text.lower()
    #             
    #             # Check for obvious contradictions (this is a simplified approach)
    #             conflicting_indicators = [
    #                 ("was born in", "born in"),
    #                 ("died in", "death"),
    #                 ("located in", "located"),
    #                 ("founded in", "founded"),
    #                 ("created in", "created"),
    #                 ("president", "president"),
    #                 ("capital", "capital"),
    #                 ("population", "population")
    #             ]
    #             
    #             needs_update = False
    #             conflict_type = None
    #             
    #             for indicator1, indicator2 in conflicting_indicators:
    #                 if indicator1 in fact_lower and indicator2 in wiki_lower:
    #                     # Extract the conflicting information
    #                     if indicator1 == "president":
    #                         fact_value = extract_president_name(fact)
    #                         wiki_value = extract_president_name(wiki_text)
    #                     elif indicator1 == "population":
    #                         fact_value = extract_number_after_keyword(fact, "population")
    #                         wiki_value = extract_number_after_keyword(wiki_text, "population")
    #                     else:
    #                         fact_value = extract_info_after_keyword(fact, indicator1.split()[0])
    #                         wiki_value = extract_info_after_keyword(wiki_text, indicator2.split()[0])
    #                     
    #                     if fact_value and wiki_value and fact_value != wiki_value:
    #                         needs_update = True
    #                         conflict_type = indicator1
    #                         break
    #             
    #             if needs_update:
    #                 # Remove from legacy memory and update hybrid memory
    #                 allie_memory.remove_fact(fact)
    #                 memory_validation_updates.append(f"Updated {conflict_type} fact: '{fact}' → validated against Wikipedia")
    #                 
    #                 # Extract new facts from Wikipedia content
    #                 wiki_learning = auto_learner.process_message(wiki_text, "wikipedia_validation")
    #                 if wiki_learning["learning_actions"]:
    #                     validation_confirmations = auto_learner.generate_learning_response(wiki_learning["learning_actions"])
    #                     memory_validation_updates.extend(validation_confirmations)
    #                     
    #                     # Add validated facts to hybrid memory
    #                     for action in wiki_learning["learning_actions"]:
    #                         if action.get("action") == "stored_fact":
    #                             hybrid_memory.add_fact(
    #                                 action["fact"],
    #                                 category=action.get("category", "general"),
    #                                 confidence=0.9,  # High confidence from Wikipedia
    #                                 source="wikipedia_validation"
    #                             )

    # Step 3: Determine if external search is needed
    if is_self_referential:
        # Handle self-referential questions directly without external searches
        needs_web_search = False
        needs_wikipedia = False
    else:
        # Improved search logic - avoid searching for meta-comments, corrections, or conversational phrases
        meta_indicators = [
            "your answer", "you said", "you mentioned", "that's not", "that's wrong", "incorrect",
            "sorry", "apologize", "confused", "clarify", "explain", "what do you mean",
            "i don't understand", "can you", "could you", "would you", "please",
            "thank you", "thanks", "okay", "sure", "yes", "no", "maybe", "perhaps",
            "actually", "well", "hmm", "oh", "ah", "um", "like", "you know"
        ]
        
        is_meta_query = any(indicator in prompt.lower() for indicator in meta_indicators)
        
        if is_meta_query:
            needs_web_search = False
            needs_wikipedia = False
        else:
            search_keywords = ["current", "today", "latest", "news", "weather", "price", "stock", "score", "result", "update", "now", "what", "who", "where", "when", "how", "why"]
            needs_web_search = any(keyword in prompt.lower() for keyword in search_keywords)
            needs_wikipedia = any(word in prompt.lower() for word in ["history", "biography", "science", "geography", "technology", "definition", "explain", "president", "politics", "government", "election", "political"])

    # Check if we have sufficient memory coverage
    # Memory is used as SUPPLEMENTARY context, not the primary source
    has_good_memory_coverage = len(relevant_facts) >= 1
    
    logger.info(f"Memory coverage check: {len(relevant_facts)} facts, coverage={has_good_memory_coverage}, needs_web={needs_web_search if not is_self_referential else 'N/A'}")

    # Step 4: Perform external searches using new multi-source retrieval
    # ALWAYS search external sources for factual queries to ensure accuracy
    web_results = None
    wiki_results = None
    multi_source_results = None

    # For factual queries, ALWAYS check external sources (don't rely solely on memory)
    if (needs_web_search or needs_wikipedia) and not is_self_referential:
        logger.info(f"Triggering multi-source search for: '{prompt}' (memory has {len(relevant_facts)} facts)")
        
        # Use new retrieval system that searches all sources
        multi_source_results = await search_all_sources(
            query=prompt,
            memory_results=relevant_facts,
            max_results_per_source=3
        )
        
        # Store facts from multi-source results
        facts_stored = []
        if multi_source_results.get("success") and multi_source_results.get("facts_to_store"):
            for fact_data in multi_source_results["facts_to_store"]:
                # Validate that fact is a string
                fact = fact_data.get("fact", "")
                if not isinstance(fact, str):
                    logger.error(f"Skipping non-string fact from {fact_data.get('source')}: type={type(fact)}, value={fact}")
                    continue
                
                if not fact.strip():
                    logger.warning(f"Skipping empty fact from {fact_data.get('source')}")
                    continue
                
                # Add to hybrid memory
                result = hybrid_memory.add_fact(
                    fact,
                    category=fact_data.get("category", "general"),
                    confidence=fact_data.get("confidence", 0.8),
                    source=fact_data.get("source", "external")
                )
                
                # Track stored facts for transparency
                if result.get("status") == "stored":
                    facts_stored.append({
                        "fact": fact[:100] + "..." if len(fact) > 100 else fact,
                        "source": fact_data.get("source")
                    })
                
                logger.info(f"Stored fact from {fact_data.get('source')}: {fact[:100]}...")
        
        # Store facts_stored for later transparency message
        if facts_stored:
            multi_source_results["facts_stored_this_query"] = facts_stored
        
        # Check for disambiguation with external source results
        if multi_source_results.get("success"):
            # Build search results for disambiguation from all sources
            search_results_for_disambiguation = []
            
            # Add results from all sources (handle dicts and lists returned by different connectors)
            all_results = multi_source_results.get("all_results", {})

            def _iter_source_results(sd):
                """Yield result dicts from a source payload that may be a dict or a list."""
                if sd is None:
                    return
                # If source returns a dict with 'results' or 'items'
                if isinstance(sd, dict):
                    if sd.get("results") and isinstance(sd.get("results"), list):
                        for r in sd.get("results"):
                            yield r
                        return
                    if sd.get("items") and isinstance(sd.get("items"), list):
                        for r in sd.get("items"):
                            yield r
                        return
                    # Single result represented as dict
                    yield sd
                    return
                # If source returns a list of result dicts
                if isinstance(sd, list):
                    for item in sd:
                        if isinstance(item, dict):
                            # If each item wraps results
                            if item.get("results") and isinstance(item.get("results"), list):
                                for r in item.get("results"):
                                    yield r
                            else:
                                yield item
                    return

            for source_name, source_data in all_results.items():
                for result in _iter_source_results(source_data):
                    # Normalize possible fields to common names
                    text = result.get("text") or result.get("snippet") or result.get("excerpt") or result.get("description") or result.get("summary") or ""
                    title = result.get("title") or result.get("name") or ""
                    confidence = result.get("confidence", 0.8)
                    # Only include meaningful results
                    if not text and not title:
                        continue
                    search_results_for_disambiguation.append({
                        "text": text,
                        "source": source_name,
                        "title": title,
                        "confidence": confidence
                    })
            
            # Use disambiguation engine if available
            if hybrid_memory.disambiguation_engine and search_results_for_disambiguation:
                disambiguation = hybrid_memory.disambiguation_engine.detect_ambiguity(
                    prompt, 
                    search_results_for_disambiguation
                )
                logger.info(f"Disambiguation check: is_ambiguous={disambiguation.get('is_ambiguous')}, interpretations={len(disambiguation.get('interpretations', []))}")
        
        # For backward compatibility, set web_results if DuckDuckGo succeeded
        if multi_source_results and "duckduckgo" in multi_source_results.get("sources_used", []):
            ddg_data = multi_source_results["all_results"].get("duckduckgo", {})
            if ddg_data.get("success"):
                web_results = ddg_data

    # Step 5: Synthesize information from all sources
    # PRIORITY: External sources > Memory (for accuracy and freshness)
    context_parts = []

    # For self-referential questions, don't include external memory/context
    if not is_self_referential:
        # Add multi-source synthesized results FIRST (highest priority for accuracy)
        if multi_source_results and multi_source_results.get("success"):
            sources_used = ", ".join(multi_source_results.get("sources_used", []))
            synthesized_text = multi_source_results.get("synthesized_text", "")
            
            if synthesized_text:
                context_parts.append(f"From external sources ({sources_used}):\n{synthesized_text}")
        
        # Add memory information ONLY as supplementary context, after external sources
        if relevant_facts and not multi_source_results:
            # Only use memory if no external sources returned results
            context_parts.append("From my memory:\n" + "\n".join(f"- {fact}" for fact in relevant_facts[:3]))
        elif relevant_facts and multi_source_results:
            # If we have both, mention memory briefly for context
            context_parts.append(f"(I also have {len(relevant_facts)} related fact(s) in memory)")

    # Fallback to old web results format
    if web_results and web_results.get("success") and web_results.get("results"):
        web_info = []
        for result in web_results["results"][:3]:  # Limit to top 3
            web_info.append(f"- {result.get('text', '')} (Source: {result.get('source', 'Web')})")
        if web_info:
            context_parts.append("Current information from web search:\n" + "\n".join(web_info))

    # Note: Wikipedia results now come through multi_source_results, but keep this for compatibility
    if wiki_results and wiki_results.get("success") and wiki_results.get("summary"):
        context_parts.append(f"Background: {wiki_results['summary'][:500]}...")

    # Step 6: External learning handled above in multi-source section
    external_learning_confirmations = []
    
    # Legacy: Process any remaining web/wiki results not handled by multi-source
    if web_results and web_results.get("success") and not multi_source_results:
        for result in web_results.get("results", []):
            text = result.get("text", "")
            if text and len(text) > 10:
                web_learning = auto_learner.process_message(text, "external_web")
                if web_learning["learning_actions"]:
                    external_learning_confirmations.extend(auto_learner.generate_learning_response(web_learning["learning_actions"]))

    # Build enhanced prompt
    if is_self_referential:
        # For self-referential questions, use a direct prompt without external context
        enhanced_prompt = f"""Answer this question about yourself as Allie: {prompt}

You are Allie, a helpful AI assistant."""
    else:
        # PRIORITY: External sources first, memory as supplement only
        enhanced_prompt = prompt

        # Always prefer external sources for factual accuracy
        if multi_source_results and multi_source_results.get("success"):
            sources_used = ", ".join(multi_source_results.get("sources_used", []))
            synthesized_text = multi_source_results.get("synthesized_text", "")
            if synthesized_text and len(synthesized_text.strip()) > 10:
                # Use external sources as the primary information
                enhanced_prompt = f"""Question: {prompt}

Current information from {sources_used}:
{synthesized_text}

Answer the question using this information."""
            else:
                # No good external results, use memory sparingly
                if relevant_facts and len(relevant_facts) > 0:
                    # Only use the most relevant memory fact
                    enhanced_prompt = f"""Question: {prompt}

What I know: {relevant_facts[0]}"""
        elif relevant_facts and len(relevant_facts) > 0:
            # No external sources available, use memory with caution
            enhanced_prompt = f"""Question: {prompt}

From memory: {relevant_facts[0]}"""

    # Step 7: Generate response using TinyLlama (moved to thread pool for async performance)
    from datetime import datetime
    current_date = datetime.now().strftime("%B %d, %Y")
    
    system_content = f"""You are Allie, a helpful AI assistant. Today is {current_date}.

Give clear, direct answers. Be conversational and natural."""

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": enhanced_prompt}
    ]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def generate_model_response():
        """Generate model response synchronously in thread pool"""
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs, 
            max_length=inputs['input_ids'].shape[1] + max_tokens, 
            do_sample=True, 
            temperature=0.7,   # Balanced temperature for natural responses
            top_p=0.9,         # Higher top_p for more diverse vocabulary  
            top_k=50,          # Standard top_k value
            repetition_penalty=1.2,  # Moderate penalty to avoid repetition
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_return_sequences=1
        )
        raw_response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        # Clean up the response - remove any system prompt content that might have leaked through
        response_lines = raw_response.split('\n')
        cleaned_lines = []
        skip_rest = False
        
        for line in response_lines:
            line_lower = line.lower().strip()
            
            # Skip empty lines
            if not line_lower:
                continue
            
            # Skip lines that contain system prompt indicators or instructions
            if any(indicator in line_lower for indicator in [
                "you are allie", "helpful and friendly ai assistant", "today's date is",
                "you have access to", "your primary role", "always respond in clear",
                "synthesize information", "you are designed to respond",
                "instructions:", "- answer", "- do not", "- keep", "- be conversational",
                "do not repeat or include these instructions", "do not mention your capabilities"
            ]):
                skip_rest = True  # Once we hit instructions, skip everything after
                continue
            
            # If we're in skip mode (hit instructions), don't add any more lines
            if skip_rest:
                continue
                
            cleaned_lines.append(line)
        
        cleaned_response = '\n'.join(cleaned_lines).strip()
        
        # Additional check: if the response starts with "Instructions:" or similar, return empty
        if any(cleaned_response.lower().startswith(x) for x in ["instructions:", "here's how", "remember:", "note:"]):
            return ""
            
        return cleaned_response
    
    reply = await asyncio.to_thread(generate_model_response)
    
    # If reply is empty or very short, provide a fallback
    if not reply or len(reply.strip()) < 3:
        reply = "I apologize, but I'm having trouble generating a response. Could you please rephrase your question?"

    # Step 8: Process assistant response for additional learning
    assistant_learning = auto_learner.process_message(reply, "assistant")
    if assistant_learning["learning_actions"]:
        assistant_confirmation = auto_learner.generate_learning_response(assistant_learning["learning_actions"])
        reply += "\n\n" + assistant_confirmation

    # Step 9: Add all learning confirmations and memory updates
    final_reply = reply
    if learning_confirmation:
        final_reply += "\n\n" + learning_confirmation
    if external_learning_confirmations:
        final_reply += "\n\n" + "\n".join(external_learning_confirmations)
    if memory_validation_updates:
        final_reply += "\n\n" + "\n".join(memory_validation_updates)

    # Step 9.5: Handle disambiguation results
    if disambiguation and disambiguation.get("is_ambiguous"):
        # Format disambiguation results for UI display
        interpretations = disambiguation.get("interpretations", [])
        if interpretations:
            final_reply += "\n\n**Multiple Interpretations Detected:**\n"
            for i, interp in enumerate(interpretations[:3], 1):  # Show up to 3
                status_emoji = {
                    "true": "✅",
                    "not_verified": "⚠️",
                    "false": "❌"
                }.get(interp.get("status", "not_verified"), "❓")

                confidence_pct = int(interp.get("confidence_score", 0) * 100)
                final_reply += f"{i}. **{interp['meaning_label']}** {status_emoji} ({confidence_pct}% confidence)\n"
                final_reply += f"   {interp['summary']}\n"
                final_reply += f"   *Sources: {', '.join(interp.get('sources_consulted', []))}*\n\n"

            if disambiguation.get("needs_clarification"):
                clarifying_question = "Could you clarify which interpretation you meant?"
                final_reply += f"**{clarifying_question}**\n\n"

        # Log disambiguation events and fact-check warnings
        if disambiguation:
            # Log disambiguation event
            logger.info(f"Disambiguation: query='{prompt}', ambiguous={disambiguation.get('is_ambiguous')}, interpretations={len(disambiguation.get('interpretations', []))}, confidence={disambiguation.get('confidence', 0):.2f}")

            # Log to learning_log if available (for now, just log to console)
            # TODO: Store in learning_log table when database integration is complete

        if fact_check_warnings:
            logger.info(f"Fact-check warnings: {len(fact_check_warnings)} warnings for query '{prompt}'")
            for warning in fact_check_warnings:
                logger.info(f"  - {warning}")

    # Step 9.6: Add fact-check warnings to response
    if fact_check_warnings:
        final_reply += "\n\n**Fact-Check Warnings:**\n"
        for warning in fact_check_warnings[:3]:  # Limit to 3 warnings
            final_reply += f"⚠️ {warning}\n"
        final_reply += "\n"

    # Step 10: Always add source URL and confidence score to every response
    source_info = []
    confidence_score = 0.0
    primary_source = "model"

    # Determine primary source and confidence
    if not is_self_referential and multi_source_results and multi_source_results.get("success"):
        # External sources were used - high confidence
        primary_source = "external_sources"
        confidence_score = 0.85  # High confidence from external verification

        # Collect URLs from all sources
        all_results = multi_source_results.get("all_results", {})

        # Wikipedia URLs
        if "wikipedia" in multi_source_results.get("sources_used", []):
            wiki_data = all_results.get("wikipedia", {})
            if wiki_data.get("success") and wiki_data.get("results"):
                for result in wiki_data["results"][:1]:  # Top result
                    url = result.get("url", "")
                    if url:
                        source_info.append(f"📖 Wikipedia: {url}")

        # DuckDuckGo URLs (use the search results)
        if "duckduckgo" in multi_source_results.get("sources_used", []):
            ddg_data = all_results.get("duckduckgo", {})
            if ddg_data.get("success") and ddg_data.get("results"):
                for idx, result in enumerate(ddg_data["results"][:2], 1):  # Top 2 results
                    if result.get("url"):
                        source_name = result.get("source", f"Source {idx}")
                        source_info.append(f"🔍 {source_name}: {result['url']}")

        # Wikidata URLs
        if "wikidata" in multi_source_results.get("sources_used", []):
            wikidata_data = all_results.get("wikidata", {})
            if wikidata_data.get("success") and wikidata_data.get("results"):
                for result in wikidata_data["results"][:1]:  # Top result
                    url = result.get("url", "")
                    if url:
                        source_info.append(f"🗂️ Wikidata: {url}")

        # OpenLibrary URLs
        if "openlibrary" in multi_source_results.get("sources_used", []):
            openlibrary_data = all_results.get("openlibrary", {})
            if openlibrary_data.get("success") and openlibrary_data.get("results"):
                for result in openlibrary_data["results"][:1]:  # Top result
                    url = result.get("url", "")
                    if url:
                        source_info.append(f"� OpenLibrary: {url}")

        # DBpedia URLs
        if "dbpedia" in multi_source_results.get("sources_used", []):
            dbpedia_data = all_results.get("dbpedia", {})
            if dbpedia_data.get("success") and dbpedia_data.get("results"):
                for result in dbpedia_data["results"][:1]:  # Top result
                    url = result.get("url", "")
                    if url:
                        source_info.append(f"🔗 DBpedia: {url}")

    elif relevant_facts and len(relevant_facts) > 0:
        # Memory-based response - medium confidence
        primary_source = "memory"
        confidence_score = 0.70  # Medium confidence from stored knowledge
        source_info.append("💾 Internal Memory: Stored knowledge base")

        # If we have specific memory facts with confidence scores, use the highest
        if hybrid_results:
            max_confidence = max((result.get('confidence', 0.5) for result in hybrid_results), default=0.5)
            confidence_score = min(max_confidence, 0.75)  # Cap at 0.75 for memory-based

    else:
        # Pure model generation - lower confidence
        primary_source = "model"
        confidence_score = 0.60  # Base confidence for model-generated responses
        source_info.append("🤖 AI Model: Generated response")

    # Add sources and confidence section to every response
    final_reply += f"\n\n---\n**Source:** {primary_source.title()}\n**Confidence:** {confidence_score:.0%}"

    if source_info:
        final_reply += "\n**URLs:**\n" + "\n".join(source_info)

    return {"text": final_reply}

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

    # Process through advanced memory pipeline
    # Extract keyword from fact
    words = fact.split()[:5]
    keyword = ' '.join(words).strip('.,!?;:')
    
    result = learning_pipeline.process_fact(
        keyword=keyword,
        fact=fact,
        source="user",
        base_confidence=importance,
        category=category,
        auto_resolve=True
    )
    
    # Also add to legacy memory for backward compatibility
    allie_memory.add_fact(fact, importance, category)
    
    return {
        "status": "fact_added", 
        "fact": fact,
        "pipeline_status": result.get("final_status"),
        "confidence": result.get("confidence")
    }

@app.get("/api/memory/recall")
async def recall_memory(query: str = "", limit: int = 5):
    """Recall relevant facts from advanced memory"""
    if not query:
        # Return recent facts if no query
        timeline = advanced_memory.timeline(limit=limit)
        return {"facts": [f["fact"] for f in timeline]}

    # Search advanced memory
    results = advanced_memory.search_facts(query, limit=limit)
    facts = [f["fact"] for f in results]
    
    return {"query": query, "facts": facts, "count": len(facts)}

@app.delete("/api/memory/fact")
async def remove_memory_fact(fact: str):
    """Remove a specific fact from Allie's memory"""
    if not fact:
        raise HTTPException(status_code=400, detail="Fact text is required")
    
    removed = allie_memory.remove_fact(fact)
    if removed:
        return {"status": "fact_removed", "fact": fact}
    else:
        raise HTTPException(status_code=404, detail="Fact not found in memory")

# ======================================
# HYBRID MEMORY API ENDPOINTS
# ======================================

@app.post("/api/hybrid-memory/add")
async def add_to_hybrid_memory(
    fact: str = Body(...),
    category: str = Body("general"),
    confidence: float = Body(1.0),
    source: str = Body("user")
):
    """Add a fact to the hybrid memory system"""
    try:
        hybrid_memory.add_fact(fact, category=category, confidence=confidence, source=source)
        return {
            "status": "success",
            "message": "Fact added to hybrid memory",
            "fact": fact,
            "category": category
        }
    except Exception as e:
        logger.error(f"Error adding to hybrid memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/feedback/mark_false")
async def mark_response_false(payload: Dict[str, Any] = Body(...)):
    """User reports that a specific assistant response is wrong.
    Expected payload: { conv_id: str, message_index: int, correction?: str }
    This will:
    - mark the message with is_false and user_correction
    - add a flagged negative fact to hybrid memory
    - write a feedback report to data/feedback_reports.jsonl
    - append to negative_examples.jsonl if a correction is provided
    - trigger auto-learning check in background
    """
    try:
        conv_id = payload.get("conv_id")
        msg_idx = int(payload.get("message_index"))
        correction = payload.get("correction")

        if conv_id is None or msg_idx is None:
            raise HTTPException(status_code=400, detail="conv_id and message_index required")

        # Find conversation
        conv = next((c for c in conversation_history if c.get("id") == conv_id), None)
        if conv is None:
            raise HTTPException(status_code=404, detail="Conversation not found")

        messages = conv.get("messages", [])
        if msg_idx < 0 or msg_idx >= len(messages):
            raise HTTPException(status_code=400, detail="message_index out of range")

        msg = messages[msg_idx]
        if msg.get("role") != "them":
            raise HTTPException(status_code=400, detail="Can only mark assistant responses as false")

        # Mark message
        msg["is_false"] = True
        if correction:
            msg["user_correction"] = correction

        # Save conversation backup
        with open(DATA_DIR / "backup.json", "w", encoding="utf-8") as f:
            json.dump(conversation_history, f, indent=2)

        # Add flagged fact to hybrid memory for later processing
        try:
            if hybrid_memory:
                hybrid_memory.add_fact(
                    fact=msg.get("text", ""),
                    category="reported_false",
                    confidence=0.0,
                    source="user_feedback",
                    metadata={"flagged_false": True, "conv_id": conv_id, "message_index": msg_idx}



                )
        except Exception as e:
            logger.warning(f"Failed to add flagged fact to hybrid memory: {e}")

        # Write a feedback report to file for auditing and training
        try:
            report = {
                "timestamp": datetime.now().isoformat(),
                "conv_id": conv_id,
                "message_index": msg_idx,
                "assistant_text": msg.get("text", ""),
                "correction": correction
            }
            with open(DATA_DIR / "feedback_reports.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(report) + "\n")
        except Exception as e:
            logger.warning(f"Failed to write feedback report: {e}")

        # If we have a prior user prompt and the reporter provided a correction, add a negative example
        try:
            prompt_text = None
            if msg_idx > 0 and messages[msg_idx - 1].get("role") == "me":
                prompt_text = messages[msg_idx - 1].get("text")

            if correction and prompt_text:
                neg_example = {"prompt": prompt_text, "completion": correction}
                with open(DATA_DIR / "negative_examples.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(neg_example) + "\n")
        except Exception as e:
            logger.warning(f"Failed to append negative example: {e}")

        # Trigger auto-learning check asynchronously
        try:
            asyncio.create_task(check_and_trigger_auto_learning())
        except Exception:
            pass

        return {"status": "reported", "conv_id": conv_id, "message_index": msg_idx}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing feedback mark_false: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/feedback/request_better")
async def request_better_answer(payload: Dict[str, Any] = Body(...)):
    """User requests that Allie give a better answer for a specific assistant message.
    Expected payload: { conv_id: str, message_index: int }
    This will call the normal generation flow with context and append the new reply to the conversation.
    """
    try:
        conv_id = payload.get("conv_id")
        msg_idx = int(payload.get("message_index"))

        if conv_id is None:
            raise HTTPException(status_code=400, detail="conv_id required")

        conv = next((c for c in conversation_history if c.get("id") == conv_id), None)
        if conv is None:
            raise HTTPException(status_code=404, detail="Conversation not found")

        messages = conv.get("messages", [])
        if msg_idx < 0 or msg_idx >= len(messages):
            raise HTTPException(status_code=400, detail="message_index out of range")

        orig_msg = messages[msg_idx]
        if orig_msg.get("role") != "them":
            raise HTTPException(status_code=400, detail="Can only request better answers for assistant messages")

        # Build an improvement prompt using surrounding context
        # Prefer the user prompt immediately before the assistant reply
        user_prompt = None
        if msg_idx > 0 and messages[msg_idx - 1].get("role") == "me":
            user_prompt = messages[msg_idx - 1].get("text")

        improvement_instructions = "The previous assistant response was reported as incorrect. Please provide a better, corrected answer to the user's original prompt. If available, reference facts from memory or reputable sources and avoid hallucinations."

        if user_prompt:
            gen_prompt = f"User asked: {user_prompt}\n\nPrevious assistant answer: {orig_msg.get('text', '')}\n\n{improvement_instructions}\n\nProvide an improved answer:" 
        else:
            gen_prompt = f"Previous assistant answer: {orig_msg.get('text', '')}\n\n{improvement_instructions}\n\nProvide an improved answer:" 

        # Call the existing generation flow directly
        gen_payload = {"prompt": gen_prompt, "conversation_context": messages, "max_tokens": 512}
        try:
            gen_result = await generate_response(gen_payload)
        except Exception as e:
            logger.error(f"Error generating improved answer: {e}")
            raise HTTPException(status_code=500, detail=str(e))

        new_reply = gen_result.get("text") or gen_result.get("content") or str(gen_result)

        # Append new assistant reply to conversation
        messages.append({"role": "them", "text": new_reply, "timestamp": datetime.now().timestamp()})

        # Persist conversations
        try:
            with open(DATA_DIR / "backup.json", "w", encoding="utf-8") as f:
                json.dump(conversation_history, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save conversation after better-answer request: {e}")

        # Optionally process assistant reply for automatic learning
        try:
            assistant_learning = auto_learner.process_message(new_reply, "assistant")
            if assistant_learning.get("learning_actions"):
                logger.info(f"Auto-learner extracted {assistant_learning.get('total_facts_learned',0)} facts from improved assistant reply")
        except Exception:
            # Non-fatal if learning processing fails
            logger.debug("Auto-learner failed to process improved reply")

        return {"status": "ok", "new_reply": new_reply}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing request_better: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/hybrid-memory/search")
async def search_hybrid_memory(query: str, limit: int = 10):
    """Search the hybrid memory system"""
    try:
        search_result = hybrid_memory.search(query, limit=limit)
        results = search_result.get("results", [])
        return {
            "query": query,
            "count": len(results),
            "facts": results  # Already in dict format
        }
    except Exception as e:
        logger.error(f"Error searching hybrid memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/kb")
async def list_kb(status: str = None, limit: int = 100, offset: int = 0):
    """List Knowledge Base facts with optional status filter"""
    try:
        # If status provided, filter; otherwise return paginated list
        if status:
            # Using advanced_memory DB connector
            cursor = advanced_memory.connection.cursor(dictionary=True)
            cursor.execute("SELECT id, keyword, fact, source, status, confidence_score, provenance, updated_at FROM knowledge_base WHERE status = %s ORDER BY updated_at DESC LIMIT %s OFFSET %s", (status, limit, offset))
            rows = cursor.fetchall()
            cursor.close()
            return {"count": len(rows), "facts": rows}
        else:
            cursor = advanced_memory.connection.cursor(dictionary=True)
            cursor.execute("SELECT id, keyword, fact, source, status, confidence_score, provenance, updated_at FROM knowledge_base ORDER BY updated_at DESC LIMIT %s OFFSET %s", (limit, offset))
            rows = cursor.fetchall()
            cursor.close()
            return {"count": len(rows), "facts": rows}
    except Exception as e:
        logger.error(f"Error listing KB facts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/kb")
async def create_kb_fact(payload: dict):
    """Create a new KB fact (admin)"""
    try:
        keyword = payload.get('keyword')
        fact = payload.get('fact')
        source = payload.get('source', 'admin')
        status = payload.get('status', 'true')
        confidence_score = int(payload.get('confidence_score', 90))
        provenance = payload.get('provenance')
        res = advanced_memory.add_kb_fact(keyword, fact, source=source, confidence_score=confidence_score, provenance=provenance, status=status)
        return res
    except Exception as e:
        logger.error(f"Error creating KB fact: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.patch("/api/kb/{kb_id}")
async def patch_kb_fact(kb_id: int, payload: dict):
    try:
        new_fact = payload.get('fact')
        status = payload.get('status')
        confidence = payload.get('confidence_score')
        reviewer = payload.get('reviewer')
        reason = payload.get('reason')
        res = advanced_memory.update_kb_fact(kb_id, new_fact=new_fact, status=status, confidence_score=confidence, reviewer=reviewer, reason=reason)
        return res
    except Exception as e:
        logger.error(f"Error updating KB fact: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/kb/{kb_id}")
async def delete_kb_fact(kb_id: int, reviewer: str = None, reason: str = None):
    try:
        res = advanced_memory.delete_kb_fact(kb_id, reviewer=reviewer, reason=reason)
        return res
    except Exception as e:
        logger.error(f"Error deleting KB fact: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Role check helper (very simple header-based RBAC)
def _is_admin(request) -> bool:
    # Accept header 'X-User-Role: admin' for admin operations
    try:
        role = request.headers.get('X-User-Role', '')
        return role.lower() == 'admin'
    except Exception:
        return False


@app.get("/api/knowledge-base")
async def api_kb_list(request, status: str = None, limit: int = 100, offset: int = 0):
    """Return KB facts as JSON (paginated)."""
    try:
        rows = advanced_memory.get_all_kb_facts(status=status, limit=limit, offset=offset)
        return {"count": len(rows), "facts": rows}
    except Exception as e:
        logger.error(f"Error listing KB: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/knowledge-base/{kb_id}")
async def api_kb_get(request, kb_id: int):
    try:
        cursor = advanced_memory.connection.cursor(dictionary=True)
        cursor.execute("SELECT id, keyword, fact, source, status, confidence_score, provenance, updated_at FROM knowledge_base WHERE id = %s", (kb_id,))
        row = cursor.fetchone()
        cursor.close()
        if not row:
            raise HTTPException(status_code=404, detail="KB fact not found")
        return row
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting KB fact: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/knowledge-base")
async def api_kb_create(request, payload: dict):
    if not _is_admin(request):
        raise HTTPException(status_code=403, detail="admin role required")
    try:
        keyword = payload.get('keyword')
        fact = payload.get('fact')
        source = payload.get('source', 'admin')
        status = payload.get('status', 'true')
        confidence_score = int(payload.get('confidence_score', 90))
        provenance = payload.get('provenance')
        res = advanced_memory.add_kb_fact(keyword, fact, source=source, confidence_score=confidence_score, provenance=provenance, status=status)
        return res
    except Exception as e:
        logger.error(f"Error creating KB fact: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.patch("/api/knowledge-base/{kb_id}")
async def api_kb_patch(request, kb_id: int, payload: dict):
    if not _is_admin(request):
        raise HTTPException(status_code=403, detail="admin role required")
    try:
        new_fact = payload.get('fact')
        status_val = payload.get('status')
        confidence = payload.get('confidence_score')
        reviewer = payload.get('reviewer')
        reason = payload.get('reason')
        res = advanced_memory.update_kb_fact(kb_id, new_fact=new_fact, status=status_val, confidence_score=confidence, reviewer=reviewer, reason=reason)
        return res
    except Exception as e:
        logger.error(f"Error updating KB fact: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/knowledge-base/{kb_id}")
async def api_kb_delete(request, kb_id: int):
    if not _is_admin(request):
        raise HTTPException(status_code=403, detail="admin role required")
    try:
        res = advanced_memory.delete_kb_fact(kb_id, reviewer=request.headers.get('X-User', None), reason='api_delete')
        return res
    except Exception as e:
        logger.error(f"Error deleting KB fact: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/hybrid-memory/timeline")
async def get_memory_timeline(include_outdated: bool = False):
    """Get chronological timeline of all memories"""
    try:
        timeline = hybrid_memory.get_timeline(include_outdated=include_outdated)
        return {
            "count": len(timeline),
            "timeline": timeline  # Already in dict format
        }
    except Exception as e:
        logger.error(f"Error getting timeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/hybrid-memory/update")
async def update_hybrid_memory(
    old_fact: str = Body(...),
    new_fact: str = Body(...),
    source: str = Body("correction")
):
    """Update a fact in hybrid memory"""
    try:
        success = hybrid_memory.update_fact(old_fact, new_fact, source=source)
        if success:
            return {
                "status": "success",
                "message": "Fact updated",
                "old_fact": old_fact,
                "new_fact": new_fact
            }
        else:
            raise HTTPException(status_code=404, detail="Original fact not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating hybrid memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/hybrid-memory/reconcile")
async def reconcile_with_external(payload: Dict[str, Any] = Body(...)):
    """Reconcile hybrid memory with external facts"""
    external_facts = payload.get("external_facts", [])
    query = payload.get("query", "")
    source = payload.get("source", "external")

    if not isinstance(external_facts, list):
        raise HTTPException(status_code=400, detail="external_facts must be a list")

    try:
        report = hybrid_memory.reconcile_with_external(query, external_facts, source)
        return {
            "status": "success",
            "reconciliation_report": report
        }
    except Exception as e:
        logger.error(f"Error reconciling memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/hybrid-memory/statistics")
async def get_hybrid_memory_stats():
    """Get statistics about the hybrid memory system"""
    try:
        stats = hybrid_memory.get_statistics()
        
        # Add KB-specific statistics for learning score calculation
        kb_stats = advanced_memory.get_kb_statistics()
        stats.update(kb_stats)
        
        return {
            "status": "success",
            "statistics": stats
        }
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------
# Facts API Endpoints (NEW FOR UI COMPATIBILITY)
# -------------------------

@app.get("/api/facts")
async def get_facts(limit: int = 50, offset: int = 0):
    """Get facts from hybrid memory for UI compatibility"""
    try:
        # Get facts from hybrid memory timeline
        timeline = hybrid_memory.get_timeline(include_outdated=True)
        
        # Apply pagination
        start_idx = offset
        end_idx = offset + limit
        paginated_facts = timeline[start_idx:end_idx]
        
        # Convert to expected format
        facts = []
        current_time = datetime.now()
        for i, fact_data in enumerate(paginated_facts, start=offset + 1):
            # Check if fact was added within last 24 hours
            fact_timestamp = datetime.fromisoformat(fact_data["timestamp"].replace('Z', '+00:00'))
            newly_added = (current_time - fact_timestamp).total_seconds() < 86400  # 24 hours in seconds
            
            facts.append({
                "id": i,
                "fact": fact_data["fact"],
                "category": fact_data["category"],
                "confidence": fact_data.get("confidence", 0.8),
                "source": fact_data.get("source", "unknown"),
                "timestamp": fact_data["timestamp"],
                "updated_at": fact_data["timestamp"],  # Add for frontend compatibility
                "status": "true",  # Default status
                "is_outdated": fact_data.get("is_outdated", False),
                "newly_added": newly_added
            })
        
        return {
            "status": "success",  # Add status field for frontend
            "facts": facts,
            "total": len(timeline),
            "limit": limit,
            "offset": offset
        }
    except Exception as e:
        logger.error(f"Error getting facts: {e}")
        return {
            "status": "error",
            "message": str(e),
            "facts": [],
            "total": 0
        }

@app.get("/api/facts/{fact_id}")
async def get_fact(fact_id: int):
    """Get a specific fact by ID"""
    try:
        timeline = hybrid_memory.get_timeline(include_outdated=True)
        if fact_id < 1 or fact_id > len(timeline):
            raise HTTPException(status_code=404, detail="Fact not found")
        
        fact_data = timeline[fact_id - 1]
        return {
            "id": fact_id,
            "fact": fact_data["fact"],
            "category": fact_data["category"],
            "confidence": fact_data.get("confidence", 0.8),
            "source": fact_data.get("source", "unknown"),
            "timestamp": fact_data["timestamp"],
            "is_outdated": fact_data.get("is_outdated", False)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting fact {fact_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/facts/{fact_id}")
async def update_fact_status(fact_id: int, payload: Dict[str, Any] = Body(...)):
    """Update a fact's status and/or confidence in the database"""
    try:
        status = payload.get("status")
        confidence_score = payload.get("confidence_score")
        
        # Validate inputs
        if not status and confidence_score is None:
            raise HTTPException(status_code=400, detail="Must provide status or confidence_score")
        
        # Validate status value
        if status and status not in ['true', 'false', 'not_verified', 'needs_review', 'experimental']:
            raise HTTPException(status_code=400, detail=f"Invalid status: {status}. Must be one of: true, false, not_verified, needs_review, experimental")
        
        # Validate confidence_score range
        if confidence_score is not None:
            confidence_score = int(confidence_score)
            if confidence_score < 0 or confidence_score > 100:
                raise HTTPException(status_code=400, detail="Confidence score must be between 0 and 100")
        
        # Update fact status in database
        result = advanced_memory.update_fact_status(fact_id, status or 'not_verified', confidence_score)
        
        if result.get("status") == "not_found":
            raise HTTPException(status_code=404, detail=f"Fact {fact_id} not found")
        elif result.get("status") == "error":
            raise HTTPException(status_code=500, detail=result.get("message", "Database error"))
        elif result.get("status") == "updated":
            return {
                "status": "success",
                "fact_id": fact_id,
                "new_status": result.get("new_status"),
                "new_confidence_score": result.get("new_confidence_score"),
                "message": f"Fact {fact_id} updated successfully"
            }
        else:
            raise HTTPException(status_code=500, detail="Unknown error updating fact")
            
    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Validation error updating fact {fact_id}: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error updating fact {fact_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------
# Learning Status Endpoints (NEW FOR UI COMPATIBILITY)
# -------------------------

async def query_duckduckgo(http_client: httpx.AsyncClient, query: str) -> Dict[str, Any]:
    """Query DuckDuckGo for fact verification"""
    try:
        await asyncio.sleep(0.5)  # Rate limiting

        url = f"https://api.duckduckgo.com/?q={query}&format=json&no_html=1&skip_disambig=1"
        response = await http_client.get(url)

        if response.status_code == 200:
            data = response.json()
            results = []

            # Extract instant answer
            if data.get("Answer") and data["Answer"].strip():
                results.append({
                    "text": data["Answer"].strip(),
                    "source": "duckduckgo_instant",
                    "confidence": 0.8
                })

            # Extract abstract
            if data.get("AbstractText") and data["AbstractText"].strip():
                results.append({
                    "text": data["AbstractText"].strip(),
                    "source": "duckduckgo_abstract",
                    "confidence": 0.75
                })

            # Extract from RelatedTopics if available
            if data.get("RelatedTopics"):
                for topic in data["RelatedTopics"][:2]:  # Limit to 2
                    text = topic.get("Text", "").strip()
                    if text and len(text) > 20:  # Minimum useful length
                        results.append({
                            "text": text,
                            "source": "duckduckgo_related",
                            "confidence": 0.6
                        })

            # Extract from Results if available
            if data.get("Results"):
                for result in data["Results"][:2]:  # Limit to 2
                    text = result.get("Text", "").strip()
                    if text and len(text) > 20:  # Minimum useful length
                        results.append({
                            "text": text,
                            "source": "duckduckgo_results",
                            "confidence": 0.7
                        })

            return {
                "success": True,
                "results": results[:3],  # Limit to top 3 total
                "query": query
            }
        else:
            return {
                "success": False,
                "error": f"HTTP {response.status_code}",
                "query": query
            }

    except Exception as e:
        logger.warning(f"DuckDuckGo query failed for '{query}': {e}")
        return {
            "success": False,
            "error": str(e),
            "query": query
        }

async def query_wikidata(http_client: httpx.AsyncClient, query: str) -> Dict[str, Any]:
    """Query Wikidata for structured facts"""
    try:
        await asyncio.sleep(0.5)  # Rate limiting

        # First search for entities
        search_url = f"https://www.wikidata.org/w/api.php?action=wbsearchentities&search={query}&language=en&format=json&limit=3"
        response = await http_client.get(search_url)

        if response.status_code == 200:
            data = response.json()
            results = []

            if data.get("search"):
                for entity in data["search"][:2]:  # Limit to top 2
                    results.append({
                        "text": f"{entity.get('label', '')}: {entity.get('description', '')}",
                        "source": "wikidata",
                        "entity_id": entity.get("id"),
                        "confidence": 0.9
                    })

            return {
                "success": True,
                "results": results,
                "query": query
            }
        else:
            return {
                "success": False,
                "error": f"HTTP {response.status_code}",
                "query": query
            }

    except Exception as e:
        logger.warning(f"Wikidata query failed for '{query}': {e}")
        return {
            "success": False,
            "error": str(e),
            "query": query
        }

async def query_dbpedia(http_client: httpx.AsyncClient, query: str) -> Dict[str, Any]:
    """Query DBpedia for facts"""
    try:
        await asyncio.sleep(0.5)  # Rate limiting

        # Use DBpedia Spotlight for entity extraction
        spotlight_url = f"https://api.dbpedia-spotlight.org/en/annotate?text={query}&confidence=0.5&support=20"
        response = await http_client.get(spotlight_url, headers={"Accept": "application/json"})

        if response.status_code == 200:
            data = response.json()
            results = []

            # Handle the actual DBpedia Spotlight response format
            # It returns a single annotation object, not a Resources array
            if "@text" in data and data["@text"].strip():
                annotated_text = data["@text"].strip()
                confidence = float(data.get("@confidence", 0.5))

                # If the annotated text is longer than the query, it found entities
                if len(annotated_text) > len(query) and confidence > 0.3:
                    results.append({
                        "text": f"DBpedia Spotlight identified: {annotated_text}",
                        "source": "dbpedia_spotlight",
                        "confidence": confidence,
                        "uri": data.get("@URI", ""),
                        "types": data.get("@types", "")
                    })

            # Alternative: Try a different DBpedia endpoint for direct facts
            try:
                # Try to get facts about the topic directly from DBpedia
                lookup_url = f"https://lookup.dbpedia.org/api/search?query={query}&format=json&maxResults=3"
                lookup_response = await http_client.get(lookup_url, headers={"Accept": "application/json"})

                if lookup_response.status_code == 200:
                    lookup_data = lookup_response.json()
                    if lookup_data.get("docs"):
                        for doc in lookup_data["docs"][:2]:  # Limit to 2
                            label = doc.get("label", [""])[0] if isinstance(doc.get("label"), list) else doc.get("label", "")
                            comment = doc.get("comment", [""])[0] if isinstance(doc.get("comment"), list) else doc.get("comment", "")

                            if label and comment:
                                fact_text = f"{label}: {comment}"
                                results.append({
                                    "text": fact_text,
                                    "source": "dbpedia_lookup",
                                    "confidence": 0.8,
                                    "uri": doc.get("resource", [""])[0] if isinstance(doc.get("resource"), list) else doc.get("resource", "")
                                })
            except Exception as e:
                logger.debug(f"DBpedia lookup failed: {e}")

            return {
                "success": True,
                "results": results,
                "query": query
            }
        else:
            return {
                "success": False,
                "error": f"HTTP {response.status_code}",
                "query": query
            }

    except Exception as e:
        logger.warning(f"DBpedia query failed for '{query}': {e}")
        return {
            "success": False,
            "error": str(e),
            "query": query
        }

async def query_wikipedia(http_client: httpx.AsyncClient, query: str) -> Dict[str, Any]:
    """Query Wikipedia for facts"""
    try:
        await asyncio.sleep(0.5)  # Rate limiting

        # Use Wikipedia API for search and extract
        search_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{query.replace(' ', '_')}"
        response = await http_client.get(search_url, headers={"User-Agent": "Allie-AI/1.0"})

        if response.status_code == 200:
            data = response.json()
            results = []

            # Extract the summary/description
            if data.get("extract") and data["extract"].strip():
                results.append({
                    "text": data["extract"].strip(),
                    "source": "wikipedia",
                    "confidence": 0.85,
                    "title": data.get("title", ""),
                    "url": data.get("content_urls", {}).get("desktop", {}).get("page", "")
                })

            return {
                "success": True,
                "results": results,
                "query": query
            }
        else:
            return {
                "success": False,
                "error": f"HTTP {response.status_code}",
                "query": query
            }

    except Exception as e:
        logger.warning(f"Wikipedia query failed for '{query}': {e}")
        return {
            "success": False,
            "error": str(e),
            "query": query
        }

async def query_openlibrary(http_client: httpx.AsyncClient, query: str) -> Dict[str, Any]:
    """Query Open Library for book/author information"""
    try:
        await asyncio.sleep(0.5)  # Rate limiting

        # Search for books/authors
        search_url = f"https://openlibrary.org/search.json?q={query}&limit=3"
        response = await http_client.get(search_url)

        if response.status_code == 200:
            data = response.json()
            results = []

            # Extract book information
            if data.get("docs"):
                for doc in data["docs"][:2]:  # Limit to 2
                    title = doc.get("title", "")
                    author = doc.get("author_name", ["Unknown"])[0] if doc.get("author_name") else "Unknown"
                    first_publish_year = doc.get("first_publish_year", "")
                    subject = doc.get("subject", [""])[0] if doc.get("subject") else ""

                    if title:
                        fact_text = f"'{title}' by {author}"
                        if first_publish_year:
                            fact_text += f" (published {first_publish_year})"
                        if subject:
                            fact_text += f" - {subject}"

                        results.append({
                            "text": fact_text,
                            "source": "openlibrary",
                            "confidence": 0.8,
                            "key": doc.get("key", ""),
                            "isbn": doc.get("isbn", [""])[0] if doc.get("isbn") else ""
                        })

            return {
                "success": True,
                "results": results,
                "query": query
            }
        else:
            return {
                "success": False,
                "error": f"HTTP {response.status_code}",
                "query": query
            }

    except Exception as e:
        logger.warning(f"Open Library query failed for '{query}': {e}")
        return {
            "success": False,
            "error": str(e),
            "query": query
        }

async def query_musicbrainz(http_client: httpx.AsyncClient, query: str) -> Dict[str, Any]:
    """Query MusicBrainz for music/artist information"""
    try:
        await asyncio.sleep(0.5)  # Rate limiting

        # Search for artists/albums
        search_url = f"https://musicbrainz.org/ws/2/artist?query={query}&limit=3&fmt=json"
        response = await http_client.get(search_url, headers={"User-Agent": "Allie-AI/1.0"})

        if response.status_code == 200:
            data = response.json()
            results = []

            # Extract artist information
            if data.get("artists"):
                for artist in data["artists"][:2]:  # Limit to 2
                    name = artist.get("name", "")
                    country = artist.get("country", "")
                    type = artist.get("type", "")
                    disambiguation = artist.get("disambiguation", "")

                    if name:
                        fact_text = f"{name}"
                        if type:
                            fact_text += f" ({type})"
                        if country:
                            fact_text += f" from {country}"
                        if disambiguation:
                            fact_text += f" - {disambiguation}"

                        results.append({
                            "text": fact_text,
                            "source": "musicbrainz",
                            "confidence": 0.8,
                            "id": artist.get("id", ""),
                            "type": artist.get("type", "")
                        })

            return {
                "success": True,
                "results": results,
                "query": query
            }
        else:
            return {
                "success": False,
                "error": f"HTTP {response.status_code}",
                "query": query
            }

    except Exception as e:
        logger.warning(f"MusicBrainz query failed for '{query}': {e}")
        return {
            "success": False,
            "error": str(e),
            "query": query
        }

async def query_restcountries(http_client: httpx.AsyncClient, query: str) -> Dict[str, Any]:
    """Query REST Countries for country information"""
    try:
        await asyncio.sleep(0.5)  # Rate limiting

        # Search for countries
        search_url = f"https://restcountries.com/v3.1/name/{query}?fullText=false"
        response = await http_client.get(search_url)

        if response.status_code == 200:
            data = response.json()
            results = []

            # Extract country information
            for country in data[:2]:  # Limit to 2
                name = country.get("name", {}).get("common", "")
                capital = country.get("capital", [""])[0] if country.get("capital") else ""
                region = country.get("region", "")
                population = country.get("population", 0)
                languages = list(country.get("languages", {}).values()) if country.get("languages") else []

                if name:
                    fact_text = f"{name}"
                    if capital:
                        fact_text += f", capital: {capital}"
                    if region:
                        fact_text += f", region: {region}"
                    if population > 0:
                        fact_text += f", population: {population:,}"
                    if languages:
                        fact_text += f", languages: {', '.join(languages)}"

                    results.append({
                        "text": fact_text,
                        "source": "restcountries",
                        "confidence": 0.9,
                        "code": country.get("cca3", ""),
                        "flag": country.get("flag", "")
                    })

            return {
                "success": True,
                "results": results,
                "query": query
            }
        else:
            return {
                "success": False,
                "error": f"HTTP {response.status_code}",
                "query": query
            }

    except Exception as e:
        logger.warning(f"REST Countries query failed for '{query}': {e}")
        return {
            "success": False,
            "error": str(e),
            "query": query
        }

async def query_arxiv(http_client: httpx.AsyncClient, query: str) -> Dict[str, Any]:
    """Query ArXiv for scientific papers"""
    try:
        await asyncio.sleep(0.5)  # Rate limiting

        # Search for papers
        search_url = f"https://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results=3"
        response = await http_client.get(search_url)

        if response.status_code == 200:
            # Parse XML response (simple approach)
            import xml.etree.ElementTree as ET
            root = ET.fromstring(response.text)
            results = []

            # Extract paper information
            for entry in root.findall("{http://www.w3.org/2005/Atom}entry")[:2]:  # Limit to 2
                title_elem = entry.find("{http://www.w3.org/2005/Atom}title")
                summary_elem = entry.find("{http://www.w3.org/2005/Atom}summary")
                authors = entry.findall("{http://www.w3.org/2005/Atom}author")

                title = title_elem.text.strip() if title_elem is not None else ""
                summary = summary_elem.text.strip() if summary_elem is not None else ""
                author_names = [author.find("{http://www.w3.org/2005/Atom}name").text
                              for author in authors if author.find("{http://www.w3.org/2005/Atom}name") is not None]

                if title and summary:
                    fact_text = f"Paper: '{title}' by {', '.join(author_names[:3])}"
                    if len(summary) > 100:
                        fact_text += f" - {summary[:200]}..."

                    results.append({
                        "text": fact_text,
                        "source": "arxiv",
                        "confidence": 0.8,
                        "title": title,
                        "authors": author_names
                    })

            return {
                "success": True,
                "results": results,
                "query": query
            }
        else:
            return {
                "success": False,
                "error": f"HTTP {response.status_code}",
                "query": query
            }

    except Exception as e:
        logger.warning(f"ArXiv query failed for '{query}': {e}")
        return {
            "success": False,
            "error": str(e),
            "query": query
        }

async def query_pubmed(http_client: httpx.AsyncClient, query: str) -> Dict[str, Any]:
    """Query PubMed for medical/scientific research"""
    try:
        await asyncio.sleep(0.5)  # Rate limiting

        # Use PubMed E-utilities (NCBI)
        search_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={query}&retmax=3&retmode=json"
        response = await http_client.get(search_url)

        if response.status_code == 200:
            data = response.json()
            results = []

            # Get paper IDs
            id_list = data.get("esearchresult", {}).get("idlist", [])

            if id_list:
                # Fetch summaries for the papers
                ids = ",".join(id_list[:2])  # Limit to 2
                summary_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id={ids}&retmode=json"
                summary_response = await http_client.get(summary_url)

                if summary_response.status_code == 200:
                    summary_data = summary_response.json()
                    result = summary_data.get("result", {})

                    for paper_id in id_list[:2]:
                        if paper_id in result:
                            paper = result[paper_id]
                            title = paper.get("title", "")
                            authors = paper.get("authors", [])
                            pubdate = paper.get("pubdate", "")

                            if title:
                                author_names = [author.get("name", "") for author in authors if author.get("name")]
                                fact_text = f"Research: '{title}'"
                                if author_names:
                                    fact_text += f" by {', '.join(author_names[:3])}"
                                if pubdate:
                                    fact_text += f" ({pubdate})"

                                results.append({
                                    "text": fact_text,
                                    "source": "pubmed",
                                    "confidence": 0.8,
                                    "pmid": paper_id,
                                    "title": title
                                })

            return {
                "success": True,
                "results": results,
                "query": query
            }
        else:
            return {
                "success": False,
                "error": f"HTTP {response.status_code}",
                "query": query
            }

    except Exception as e:
        logger.warning(f"PubMed query failed for '{query}': {e}")
        return {
            "success": False,
            "error": str(e),
            "query": query
        }

@app.get("/api/learning/status")
async def get_learning_status():
    """Get learning system status for UI"""
    try:
        # Get memory statistics
        stats = hybrid_memory.get_statistics()
        
        # Calculate learning readiness based on data availability
        total_facts = stats.get("total_facts", 0)
        active_facts = stats.get("active_facts", 0)
        
        # Simple heuristics for learning readiness
        should_learn = total_facts >= 10 and active_facts >= 5
        auto_learning = total_facts >= 50  # Enable auto-learning with more data
        
        # Mock learning status for UI compatibility
        status = {
            "enabled": True,
            "is_active": False,  # No active learning episodes currently
            "should_learn": should_learn,
            "auto_learning": auto_learning,
            "reason": "Ready to learn" if should_learn else f"Need more facts (have {total_facts}, need 10)",
            "current_episode": None,
            "data_stats": {
                "total_conversations": len(conversation_history),
                "quality_conversations": len([c for c in conversation_history if len(c.get("messages", [])) >= 4]),
                "total_facts": total_facts,
                "active_facts": active_facts
            },
            "system_resources": {
                "cpu_percent": 25.0,  # Mock values
                "memory_percent": 45.0
            }
        }
        
        return status
        
    except Exception as e:
        logger.error(f"Error getting learning status: {e}")
        return {
            "enabled": False,
            "is_active": False,
            "should_learn": False,
            "auto_learning": False,
            "reason": f"Error: {str(e)}",
            "current_episode": None
        }

@app.post("/api/learning/start")
async def start_learning():
    """Start a learning episode (mock implementation)"""
    try:
        import uuid
        episode_id = str(uuid.uuid4())[:8]
        
        logger.info(f"Mock learning episode started: {episode_id}")
        
        return {
            "status": "started",
            "episode_id": episode_id,
            "message": "Learning episode started successfully"
        }
        
    except Exception as e:
        logger.error(f"Error starting learning: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/learning-log")
async def get_learning_log(limit: int = 100, offset: int = 0, change_type: str = None):
    """Get learning log entries for UI"""
    try:
        cursor = None
        connection = None
        try:
            import mysql.connector
            connection = mysql.connector.connect(
                host='localhost',
                user='allie',
                password='StrongPassword123!',
                database='allie_memory'
            )
            cursor = connection.cursor(dictionary=True)

            # Build query
            query = """
                SELECT id, fact_id, keyword, old_fact, new_fact, source, confidence, 
                       change_type, reviewer, reason, changed_at
                FROM learning_log
                WHERE 1=1
            """
            params = []

            if change_type:
                query += " AND change_type = %s"
                params.append(change_type)

            query += " ORDER BY changed_at DESC LIMIT %s OFFSET %s"
            params.extend([limit, offset])

            cursor.execute(query, params)
            entries = cursor.fetchall()

            # Get total count
            count_query = "SELECT COUNT(*) as total FROM learning_log WHERE 1=1"
            count_params = []
            if change_type:
                count_query += " AND change_type = %s"
                count_params.append(change_type)

            cursor.execute(count_query, count_params)
            total = cursor.fetchone()['total']

            return {
                "status": "success",
                "entries": entries,
                "total": total,
                "limit": limit,
                "offset": offset
            }

        except mysql.connector.Error as e:
            logger.error(f"Database error getting learning log: {e}")
            return {
                "status": "error",
                "message": str(e),
                "entries": [],
                "total": 0
            }
        finally:
            if cursor:
                cursor.close()
            if connection:
                connection.close()

    except Exception as e:
        logger.error(f"Error getting learning log: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/learning/quick-topics")
async def quick_topics_research(payload: Dict[str, Any] = Body(...)):
    """Research multiple topics in parallel and learn facts from external sources"""
    topics = payload.get("topics", [])

    if not topics or not isinstance(topics, list):
        raise HTTPException(status_code=400, detail="topics must be a non-empty list")

    if len(topics) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 topics allowed at once")

    try:
        # Import required modules for research
        import httpx
        import asyncio
        from typing import Dict, List, Any

        # Create HTTP client for external requests
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(10.0, connect=10.0),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
        ) as http_client:

            async def research_topic(topic: str) -> Dict[str, Any]:
                """Research a single topic using multiple external sources"""
                try:
                    logger.info(f"Researching topic: {topic}")

                    # Query multiple sources in parallel
                    queries = await asyncio.gather(
                        query_duckduckgo(http_client, topic),
                        query_wikidata(http_client, topic),
                        query_dbpedia(http_client, topic),
                        query_wikipedia(http_client, topic),
                        query_openlibrary(http_client, topic),
                        query_musicbrainz(http_client, topic),
                        query_restcountries(http_client, topic),
                        query_arxiv(http_client, topic),
                        query_pubmed(http_client, topic),
                        return_exceptions=True
                    )

                    # Collect successful results
                    external_results = []
                    for query_result in queries:
                        if isinstance(query_result, Exception):
                            logger.warning(f"Query failed for {topic}: {query_result}")
                            continue
                        if query_result.get("success") and query_result.get("results"):
                            external_results.extend(query_result["results"])

                    # Extract facts from results and store them
                    facts_learned = 0
                    stored_facts = []

                    for result in external_results:
                        text = result.get("text", "").strip()
                        if text and len(text) > 10:  # Minimum length check
                            try:
                                # Store fact in hybrid memory
                                fact_result = hybrid_memory.add_fact(
                                    fact=text,
                                    category=topic,  # Use topic as category instead of keyword
                                    confidence=result.get("confidence", 0.7),
                                    source=result.get("source", "external_research")
                                )
                                facts_learned += 1
                                stored_facts.append({
                                    "fact": text,
                                    "source": result.get("source", "unknown"),
                                    "confidence": result.get("confidence", 0.7)
                                })
                            except Exception as e:
                                logger.warning(f"Failed to store fact for {topic}: {e}")

                    return {
                        "topic": topic,
                        "success": facts_learned > 0,
                        "facts_learned": facts_learned,
                        "sources_queried": len([q for q in queries if not isinstance(q, Exception)]),
                        "results": stored_facts
                    }

                except Exception as e:
                    logger.error(f"Error researching topic '{topic}': {e}")
                    return {
                        "topic": topic,
                        "success": False,
                        "facts_learned": 0,
                        "error": str(e)
                    }

            # Research topics in batches of 5 to avoid overwhelming external APIs
            batch_size = 5
            all_results = []

            for i in range(0, len(topics), batch_size):
                batch = topics[i:i + batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}: {batch}")

                # Research batch in parallel
                batch_results = await asyncio.gather(*[research_topic(topic) for topic in batch])
                all_results.extend(batch_results)

                # Small delay between batches to be respectful to APIs
                if i + batch_size < len(topics):
                    await asyncio.sleep(1.0)

            # Calculate summary
            topics_processed = len(all_results)
            successful = sum(1 for r in all_results if r["success"])
            total_facts_learned = sum(r["facts_learned"] for r in all_results)

            logger.info(f"Quick topics research complete: {topics_processed} topics, {successful} successful, {total_facts_learned} facts learned")

            return {
                "topics_processed": topics_processed,
                "successful": successful,
                "total_facts_learned": total_facts_learned,
                "results": all_results
            }

    except Exception as e:
        logger.error(f"Error in quick topics research: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------
# Learning Queue Endpoints (NEW FOR UPGRADE)
# -------------------------

@app.post("/api/learning_queue")
async def enqueue_fact(payload: Dict[str, Any] = Body(...)):
    """Enqueue a new fact for processing (called by ingestion pipeline)"""
    keyword = payload.get("keyword", "").strip()
    fact = payload.get("fact", "").strip()
    source = payload.get("source", "unknown")
    provenance = payload.get("provenance", {})

    if not keyword or not fact:
        raise HTTPException(status_code=400, detail="keyword and fact are required")

    try:
        # Add to learning queue instead of directly to facts
        result = advanced_memory.add_to_learning_queue(keyword, fact, source, provenance=provenance)
        return {
            "status": "success",
            "result": result
        }
    except Exception as e:
        logger.error(f"Error enqueueing fact: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/learning_queue")
async def get_learning_queue(processed: bool = None, limit: int = 50, offset: int = 0):
    """List learning queue items"""
    try:
        # Convert processed boolean to status string
        status_filter = None
        if processed is not None:
            status_filter = "processed" if processed else "pending"

        queue_items = advanced_memory.get_learning_queue(status=status_filter, limit=limit)
        return {
            "status": "success",
            "queue": queue_items,
            "count": len(queue_items),
            "filters": {
                "processed": processed,
                "limit": limit,
                "offset": offset
            }
        }
    except Exception as e:
        logger.error(f"Error getting learning queue: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------
# Reconciliation Endpoints (NEW FOR UPGRADE)
# -------------------------

@app.post("/api/reconcile/{queue_id}/approve")
async def approve_reconciliation(queue_id: int, payload: Dict[str, Any] = Body(...)):
    """Admin approves suggested_action and applies the change"""
    reviewer = payload.get("reviewer", "admin")
    reason = payload.get("reason", "Manual approval")

    try:
        # Get the queued item
        queue_items = advanced_memory.get_learning_queue(status="pending", limit=1)
        item = next((i for i in queue_items if i["id"] == queue_id), None)

        if not item:
            raise HTTPException(status_code=404, detail=f"Queue item {queue_id} not found")

        suggested_action = item.get("suggested_action")
        if not suggested_action:
            raise HTTPException(status_code=400, detail="No suggested action available for approval")

        action_type = suggested_action.get("action")
        if action_type not in ["promote", "mark_needs_review", "auto_update", "ignore"]:
            raise HTTPException(status_code=400, detail=f"Unsupported action type: {action_type}")

        result = {"status": "approved", "action": action_type}

        if action_type == "promote":
            # Add fact to facts table with high confidence
            fact_result = advanced_memory.add_fact(
                item["fact"],
                category=item["keyword"],
                source=item["source"],
                confidence=0.9,
                status="true",
                confidence_score=85
            )
            result["fact_added"] = fact_result

        elif action_type == "mark_needs_review":
            # Add fact with needs_review status
            fact_result = advanced_memory.add_fact(
                item["fact"],
                category=item["keyword"],
                source=item["source"],
                confidence=0.5,
                status="needs_review",
                confidence_score=50
            )
            result["fact_added"] = fact_result

        elif action_type == "auto_update":
            # Update existing fact
            update_result = advanced_memory.update_fact(
                item["keyword"],
                suggested_action.get("new_fact", item["fact"]),
                item["source"],
                confidence=0.8
            )
            result["fact_updated"] = update_result

        # Mark queue item as processed
        advanced_memory.process_queue_item(queue_id, "processed")

        # Log the approval
        learning_log_entry = {
            "fact_id": result.get("fact_added", {}).get("fact_id") or result.get("fact_updated", {}).get("fact_id"),
            "old_fact": item["fact"],
            "new_fact": suggested_action.get("new_fact", item["fact"]),
            "action": f"reconcile_approve_{action_type}",
            "reviewer": reviewer,
            "reason": reason,
            "meta": {"queue_id": queue_id, "suggested_action": suggested_action}
        }

        # Log to learning_log table (would need to be implemented in AllieMemoryDB)
        # For now, we'll just return the result

        return {
            "status": "success",
            "result": result,
            "reviewer": reviewer,
            "reason": reason
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error approving reconciliation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------
# Feature Flags Endpoints (NEW FOR UPGRADE)
# -------------------------

@app.get("/api/feature_flags")
async def get_feature_flags():
    """Get all feature flags and their current status"""
    try:
        # For now, return hardcoded flags since we don't have DB integration yet
        flags = {
            "AUTO_APPLY_UPDATES": {"enabled": False, "description": "Automatically apply suggested updates from reconciliation worker"},
            "READ_ONLY_MEMORY": {"enabled": False, "description": "Disable all memory modifications"},
            "WRITE_DIRECT": {"enabled": False, "description": "Allow direct writes to facts table bypassing queue"}
        }
        return {
            "status": "success",
            "flags": flags
        }
    except Exception as e:
        logger.error(f"Error getting feature flags: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/feature_flags")
async def update_feature_flag(payload: Dict[str, Any] = Body(...)):
    """Update a feature flag (admin only)"""
    flag_name = payload.get("flag_name")
    enabled = payload.get("enabled")

    if not flag_name:
        raise HTTPException(status_code=400, detail="flag_name is required")

    valid_flags = ["AUTO_APPLY_UPDATES", "READ_ONLY_MEMORY", "WRITE_DIRECT"]
    if flag_name not in valid_flags:
        raise HTTPException(status_code=400, detail=f"Invalid flag name. Must be one of: {', '.join(valid_flags)}")

    if not isinstance(enabled, bool):
        raise HTTPException(status_code=400, detail="enabled must be a boolean")

    try:
        # For now, just log the change since we don't have DB integration yet
        logger.info(f"Feature flag {flag_name} set to {enabled}")

        return {
            "status": "success",
            "flag_name": flag_name,
            "enabled": enabled,
            "message": f"Feature flag {flag_name} updated successfully"
        }
    except Exception as e:
        logger.error(f"Error updating feature flag: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------
# Auth Middleware Stub (NEW FOR UPGRADE)
# -------------------------

def get_current_user_role():
    """Stub for authentication - returns admin role for now"""
    # TODO: Implement proper authentication
    return "admin"

def require_role(required_role: str):
    """Decorator to check user role"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            user_role = get_current_user_role()
            # Simple role hierarchy: admin > reviewer > viewer
            role_hierarchy = {"admin": 3, "reviewer": 2, "viewer": 1}

            if role_hierarchy.get(user_role, 0) < role_hierarchy.get(required_role, 999):
                raise HTTPException(status_code=403, detail=f"Insufficient permissions. Required: {required_role}")

            return await func(*args, **kwargs)
        return wrapper
    return decorator

def check_feature_flag(flag_name: str):
    """Check if a feature flag is enabled"""
    # TODO: Implement actual flag checking from database
    # For now, return False for safety (no auto-updates)
    flag_defaults = {
        "AUTO_APPLY_UPDATES": False,
        "READ_ONLY_MEMORY": False,
        "WRITE_DIRECT": False
    }
    return flag_defaults.get(flag_name, False)

# Apply role requirements to sensitive endpoints
@app.patch("/api/facts/{fact_id}")
@require_role("reviewer")
async def update_fact_status_protected(fact_id: int, payload: Dict[str, Any] = Body(...)):
    """Protected version of update_fact_status"""
    return await update_fact_status(fact_id, payload)

@app.post("/api/reconcile/{queue_id}/approve")
@require_role("reviewer")
async def approve_reconciliation_protected(queue_id: int, payload: Dict[str, Any] = Body(...)):
    """Protected version of approve_reconciliation"""
    return await approve_reconciliation(queue_id, payload)

@app.post("/api/feature_flags")
@require_role("admin")
async def update_feature_flag_protected(payload: Dict[str, Any] = Body(...)):
    """Protected version of update_feature_flag"""
    return await update_feature_flag(payload)

# -------------------------


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

# Run backup asynchronously during startup instead of at import time
# backup_conversations()  # Commented out - will be called in lifespan if needed


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
