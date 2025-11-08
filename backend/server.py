import os
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
import asyncio

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
        local_files_only=False  # Allow downloading if not cached locally
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

from automatic_learner import AutomaticLearner

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

# Initialize memory system
MEMORY_FILE = DATA_DIR / "allie_memory.json"
allie_memory = AllieMemory(MEMORY_FILE)

# Initialize automatic learning system
auto_learner = AutomaticLearner(allie_memory)

# Initial cleanup on startup
cleanup_all_folders()
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

async def search_wikipedia(query: str) -> Dict[str, Any]:
    """Search Wikipedia for authoritative background information"""
    # Check cache first
    cache_key = _get_cache_key(query, "wikipedia")
    cached_result = _get_cached_response(cache_key)
    if cached_result:
        logger.info(f"Wikipedia search cache hit for: {query}")
        return cached_result

    logger.info(f"Wikipedia search cache miss for: {query}")
    try:
        # Use Wikipedia API for summary
        # First, search for the page title
        search_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{query.replace(' ', '_')}"
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(search_url)
            if response.status_code == 200:
                data = response.json()
                result = {
                    "query": query,
                    "title": data.get("title", query),
                    "summary": data.get("extract", ""),
                    "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
                    "success": True
                }
                _set_cached_response(cache_key, result)
                return result
            else:
                # Try search endpoint if direct page lookup fails
                search_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={query}&format=json&utf8=1"
                response = await client.get(search_url)
                if response.status_code == 200:
                    data = response.json()
                    if data.get("query", {}).get("search"):
                        # Get the top result
                        top_result = data["query"]["search"][0]
                        page_title = top_result["title"]
                        
                        # Get summary for the top result
                        summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{page_title.replace(' ', '_')}"
                        summary_response = await client.get(summary_url)
                        if summary_response.status_code == 200:
                            summary_data = summary_response.json()
                            result = {
                                "query": query,
                                "title": summary_data.get("title", page_title),
                                "summary": summary_data.get("extract", ""),
                                "url": summary_data.get("content_urls", {}).get("desktop", {}).get("page", ""),
                                "success": True
                            }
                            _set_cached_response(cache_key, result)
                            return result
                
                result = {
                    "query": query,
                    "title": query,
                    "summary": "",
                    "url": "",
                    "success": False,
                    "error": "No Wikipedia page found"
                }
                _set_cached_response(cache_key, result)
                return result
    except Exception as e:
        logger.warning(f"Wikipedia search failed: {e}")
        result = {
            "query": query,
            "title": query,
            "summary": "",
            "url": "",
            "success": False,
            "error": str(e)
        }
        _set_cached_response(cache_key, result)
        return result

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

# Keep the old generate endpoint as /api/generate
@app.post("/api/generate")
async def generate_response(payload: Dict[str, Any] = Body(...)):
    prompt = payload.get("prompt", "")
    max_tokens = payload.get("max_tokens", 200)

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

    # Step 1: Process user input for automatic learning
    learning_result = auto_learner.process_message(prompt, "user")
    learning_confirmation = auto_learner.generate_learning_response(learning_result["learning_actions"])

    # Step 2: Check memory for relevant facts
    relevant_facts = allie_memory.recall_facts(prompt)
    recent_context = allie_memory.get_recent_context()

    # Step 2.1: Check for self-referential questions that shouldn't trigger external searches
    self_referential_patterns = [
        "what is your name", "who are you", "what are you", "tell me about yourself",
        "what's your name", "who is this", "introduce yourself", "what do you do",
        "what is your purpose", "what are you called", "what should i call you"
    ]
    
    is_self_referential = any(pattern in prompt.lower() for pattern in self_referential_patterns)

    # Step 2.5: Validate memory facts against Wikipedia if we have stored facts
    memory_validation_updates = []
    if relevant_facts and not is_self_referential:
        # Search Wikipedia to validate stored facts
        validation_wiki = await search_wikipedia(prompt)
        if validation_wiki and validation_wiki.get("success") and validation_wiki.get("summary"):
            wiki_text = validation_wiki["summary"]
            
            # Compare each memory fact with Wikipedia content
            for fact in relevant_facts:
                # Simple conflict detection: if fact contradicts Wikipedia
                fact_lower = fact.lower()
                wiki_lower = wiki_text.lower()
                
                # Check for obvious contradictions (this is a simplified approach)
                conflicting_indicators = [
                    ("was born in", "born in"),
                    ("died in", "death"),
                    ("located in", "located"),
                    ("founded in", "founded"),
                    ("created in", "created"),
                    ("president", "president"),
                    ("capital", "capital"),
                    ("population", "population")
                ]
                
                needs_update = False
                conflict_type = None
                
                for indicator1, indicator2 in conflicting_indicators:
                    if indicator1 in fact_lower and indicator2 in wiki_lower:
                        # Extract the conflicting information
                        if indicator1 == "president":
                            fact_value = extract_president_name(fact)
                            wiki_value = extract_president_name(wiki_text)
                        elif indicator1 == "population":
                            fact_value = extract_number_after_keyword(fact, "population")
                            wiki_value = extract_number_after_keyword(wiki_text, "population")
                        else:
                            fact_value = extract_info_after_keyword(fact, indicator1.split()[0])
                            wiki_value = extract_info_after_keyword(wiki_text, indicator2.split()[0])
                        
                        if fact_value and wiki_value and fact_value != wiki_value:
                            needs_update = True
                            conflict_type = indicator1
                            break
                
                if needs_update:
                    # Remove the old fact and extract new facts from Wikipedia
                    allie_memory.remove_fact(fact)
                    memory_validation_updates.append(f"Updated {conflict_type} fact: '{fact}' → validated against Wikipedia")
                    
                    # Extract new facts from Wikipedia content
                    wiki_learning = auto_learner.process_message(wiki_text, "wikipedia_validation")
                    if wiki_learning["learning_actions"]:
                        validation_confirmations = auto_learner.generate_learning_response(wiki_learning["learning_actions"])
                        memory_validation_updates.extend(validation_confirmations)

    # Step 3: Determine if external search is needed
    if is_self_referential:
        # Handle self-referential questions directly without external searches
        needs_web_search = False
        needs_wikipedia = False
    else:
        search_keywords = ["current", "today", "latest", "news", "weather", "price", "stock", "score", "result", "update", "now", "what", "who", "where", "when", "how", "why"]
        needs_web_search = any(keyword in prompt.lower() for keyword in search_keywords)
        needs_wikipedia = any(word in prompt.lower() for word in ["history", "biography", "science", "geography", "technology", "definition", "explain", "president", "politics", "government", "election", "political"])

    # Check if we have sufficient memory coverage
    has_good_memory_coverage = len(relevant_facts) >= 3

    # Step 4: Perform external searches if needed
    web_results = None
    wiki_results = None

    if needs_web_search and not has_good_memory_coverage:
        web_results = await search_web(prompt)

    if needs_wikipedia:
        wiki_results = await search_wikipedia(prompt)

    # Step 5: Synthesize information from all sources
    context_parts = []

    # For self-referential questions, don't include external memory/context
    if not is_self_referential:
        # Add memory information
        if relevant_facts:
            context_parts.append("From my memory:\n" + "\n".join(f"- {fact}" for fact in relevant_facts))

        # Add recent context
        if recent_context:
            context_parts.append(f"Recent conversation context: {recent_context}")

    # Add web search results
    if web_results and web_results.get("success") and web_results.get("results"):
        web_info = []
        for result in web_results["results"][:3]:  # Limit to top 3
            web_info.append(f"- {result.get('text', '')} (Source: {result.get('source', 'Web')})")
        if web_info:
            context_parts.append("Current information from web search:\n" + "\n".join(web_info))

    # Add Wikipedia results
    if wiki_results and wiki_results.get("success") and wiki_results.get("summary"):
        context_parts.append(f"Wikipedia background: {wiki_results['summary'][:500]}...")

    # Step 6: Store new facts from external sources
    external_learning_confirmations = []
    if web_results and web_results.get("success"):
        for result in web_results.get("results", []):
            text = result.get("text", "")
            if text and len(text) > 10:  # Only process substantial facts
                web_learning = auto_learner.process_message(text, "external_web")
                if web_learning["learning_actions"]:
                    external_learning_confirmations.extend(auto_learner.generate_learning_response(web_learning["learning_actions"]))

    if wiki_results and wiki_results.get("success") and wiki_results.get("summary"):
        wiki_learning = auto_learner.process_message(wiki_results["summary"], "external_wikipedia")
        if wiki_learning["learning_actions"]:
            external_learning_confirmations.extend(auto_learner.generate_learning_response(wiki_learning["learning_actions"]))

    # Build enhanced prompt
    if is_self_referential:
        # For self-referential questions, use a direct prompt without external context
        enhanced_prompt = f"""Please answer this question directly about yourself as Allie the AI assistant: {prompt}

Remember: You are Allie, a helpful and friendly AI assistant created to answer questions and engage in natural conversation. Answer directly without referencing external sources or memory."""
    else:
        enhanced_prompt = prompt
        if context_parts:
            enhanced_prompt = f"{prompt}\n\nContext from multiple sources:\n" + "\n\n".join(context_parts)

    # Step 7: Generate response using TinyLlama (moved to thread pool for async performance)
    from datetime import datetime
    current_date = datetime.now().strftime("%B %d, %Y")
    
    system_content = f"""You are Allie, a helpful and friendly AI assistant. Today's date is {current_date}.

You are designed to respond naturally and helpfully in English. Always provide direct, clear answers to questions.

You have access to:
- Your long-term memory of important facts and information
- Recent conversation context
- Current web search results from DuckDuckGo
- Authoritative background information from Wikipedia

Your primary role is to answer questions and engage in natural conversation. When someone asks you a question, provide a direct, helpful answer based on all available information.

IMPORTANT: Always respond in clear, natural English. Do not respond in other languages unless specifically asked. Be conversational but informative.

I automatically validate my stored knowledge against authoritative sources like Wikipedia. If I find conflicting information, I update my memory to ensure accuracy.

Synthesize information from all available sources to provide comprehensive, accurate responses. If you learn something new and important, acknowledge that you're storing it for future conversations."""

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": enhanced_prompt}
    ]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def generate_model_response():
        """Generate model response synchronously in thread pool"""
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_length=inputs['input_ids'].shape[1] + max_tokens, do_sample=True, temperature=0.3, top_p=0.7, pad_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    reply = await asyncio.to_thread(generate_model_response)

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
