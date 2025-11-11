# Progress & TODO

This document records the current progress on the memory/pronoun-resolution work and lists remaining TODOs we can work from.

## How to run tests (local dev)

Run tests in the workspace venv (Windows PowerShell):

```powershell
# Activate your venv then:
C:/Users/brock/allieai/.venv/Scripts/python.exe -m pytest -q
```

Test summary (local dev):
- Unit tests: 69 passed ✅
- Integration tests: 19 passed ✅
- API endpoint tests: 31 passed ✅
- API integration tests: 12 passed ✅

## Files of interest (changes made so far)

- `backend/context_utils.py` — pronoun-resolution helper (enhance_query_with_context)
- `backend/server.py` — integrated the helper into the /api/generate / memory lookup flow; added debug logging for augmented queries
- `frontend/static/ui.html` — now sends `conversation_context` in the /api/generate request body
- `backend/simulate_conversation.py` — small script used for manual verification
- `memory/hybrid.py` (backend implementation) — test-friendly HybridMemory implementation used by tests
- `memory/__init__.py` — exports for `HybridMemory`
- `backend/memory/linked_list_impl.py` — migrated linked list implementation (moved from `linked_list.py`)
- `backend/memory/index.py` — keyword index (unchanged logic but used by memory)
- `scripts/test_search.py` — async search test adapted to run reliably in our pytest environment
- `pytest.ini` — enables asyncio support for pytest where available
- `docs/HYBRID_MEMORY_GUIDE.md` and `docs/memory/README.md` — updated references to the migrated linked-list filename

## Recent fixes and outcomes

- ✅ **FIXED: Quick-topics research functionality fully working**. Root cause was incorrect parameter (`keyword=...`) passed to `hybrid_memory.add_fact()`; corrected to use `fact=` and `category=`. Updated query functions to properly parse API responses (DuckDuckGo, DBpedia). Verified: 2 topics processed successfully, 7 facts learned and stored in hybrid memory. Backend functionality confirmed working via direct function testing.
- Fixed: UI popup "facts not loaded". Root cause: intermittent premature server shutdown in some dev environments; added `start_server.py` for more robust startup and `test_server_manual.py` to validate endpoints. Verified `/api/facts` formatting and timeline retrieval.


## Remaining TODOs

These are actionable items we can pick from next. Each item is short and has a suggested next step.

- Add query-level logging for memory searches
  - Next step: implement structured JSON logging at `backend/server.py` for `/api/generate` with fields: `request_id`, `user_query`, `augmented_query`, `top_memory_hits` (list of top N fact summaries). Add a small unit test that calls the endpoint and asserts the logger was invoked (or logs to a test handler).

- Add spaCy/coref fallback (optional)
  - Next step: decide whether to add `spacy` + `coref` (or `coreferee` / other lightweight package) to `requirements.txt`. If yes, I can wire it in `backend/context_utils.py` as an optional path activated when the package is importable.

- Install dev/runtime dependencies (optional / environment specific)
  - Suggested packages (only install if you need full model runs): `mysql-connector-python`, `transformers`, `peft`, `accelerate`, `datasets`.
  - Next step: add a `requirements-dev.txt` or `pyproject.toml` with optional extras and document how to install.

- Run full end-to-end integration tests
  - Next step: define what "end-to-end" means for this repo (does it include starting the FastAPI server, model invocation, MySQL integration?). I can add a small `run_e2e.sh` / `run_e2e.ps1` that spins up local dependencies (or mocks) and validates critical flows.

- Replace linked list with list-backed storage (bigger refactor) — optional
  - If we truly don't want linked lists, we can refactor `FactLinkedList` to be a thin wrapper around a Python list and update the index to work with list nodes or dicts.
  - Next step: I can prepare an automated refactor in a feature branch and run the tests.


---

## DONE items (bottom — completed work so far)

- Update `/api/generate` endpoint to accept conversation context
- Modify frontend to send conversation history (UI now posts `conversation_context`)
- Implement pronoun resolution in memory search (`enhance_query_with_context` in `backend/context_utils.py`)
- Test conversational context fix (Rocky Mountains / Eiffel Tower scenario validated)
- Add unit tests for pronoun-resolution helper (`tests/test_context_utils.py`)
- Create test-friendly `HybridMemory` implementation and export (`memory/hybrid.py`, `memory/__init__.py`)
- Move linked list implementation to `backend/memory/linked_list_impl.py` and update imports
- Fix hybrid memory bugs (preserve timestamps on load, always include `fact` and `is_outdated` keys in returned dicts, merge DB/in-memory statistics)
- Add a synchronous wrapper / plugin support for async tests to ensure `scripts/test_search.py` runs in CI-local environments
- Add `pytest.ini` to help pytest async mode
- Run the full pytest suite and iterate to green (local run: unit + integration + API tests passing; see Test summary above)
- Update documentation references to the migrated linked-list filename
- ✅ **COMPLETED: Fix quick-topics research "Unknown error" issue**. Corrected `hybrid_memory.add_fact()` parameter usage, updated API query parsing logic, verified fact storage working (7 facts learned from 2 topics in testing)


If you'd like, I can (pick one):
- Implement the structured JSON query-level logging now and add tests (recommended next small task),
- Wire spaCy/coref fallback and update `requirements.txt`, or
- Start the larger refactor to remove the linked-list data structure entirely.

Tell me which to do next and I'll start (I'll also update this progress file as I make changes).