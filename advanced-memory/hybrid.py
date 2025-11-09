import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime


class HybridMemory:
	"""A lightweight in-repo HybridMemory used by tests.

	Features implemented (minimal but sufficient for tests):
	- load/save facts from data/hybrid_memory.json
	- add_fact(...)
	- search(query)
	- get_timeline(include_outdated=False)
	- get_statistics()
	- reconcile_with_external(...) (simple behavior)
	"""

	def __init__(self, data_path: Optional[str] = None):
		if data_path:
			self.data_file = Path(data_path)
		else:
			self.data_file = Path(__file__).parent.parent.joinpath("data", "hybrid_memory.json")

		self.facts: List[Dict[str, Any]] = []
		self._load()

	def _load(self):
		if self.data_file.exists():
			try:
				with open(self.data_file, "r", encoding="utf-8") as f:
					payload = json.load(f)
					raw = payload.get("facts", [])
					# Normalize entries
					out = []
					for item in raw:
						obj = dict(item)
						obj.setdefault("timestamp", datetime.now().isoformat())
						obj.setdefault("category", obj.get("category", "general"))
						obj.setdefault("source", obj.get("source", "external"))
						obj.setdefault("confidence", float(obj.get("confidence", 0.9)))
						obj.setdefault("status", obj.get("status", "not_verified"))
						obj.setdefault("confidence_score", obj.get("confidence_score", int(obj.get("confidence", 0.9) * 100)))
						obj.setdefault("is_outdated", bool(obj.get("is_outdated", False)))
						out.append(obj)
					self.facts = out
			except Exception:
				self.facts = []
		else:
			self.facts = []

	def _save(self):
		try:
			payload = {"version": "1.0", "saved_at": datetime.now().isoformat(), "fact_count": len(self.facts), "facts": self.facts}
			with open(self.data_file, "w", encoding="utf-8") as f:
				json.dump(payload, f, indent=2, ensure_ascii=False)
		except Exception:
			pass

	def add_fact(self, fact: str, category: str = "general", confidence: float = 0.9, source: str = "user", status: str = "not_verified", confidence_score: int = None) -> Dict[str, Any]:
		"""Add a fact. If a closely matching fact exists, mark it outdated and
		return an update report.
		"""
		now = datetime.now().isoformat()
		normalized = fact.strip()

		# Calculate confidence_score from confidence if not provided
		if confidence_score is None:
			confidence_score = int(confidence * 100)

		# Detect potential duplicates / corrections by comparing normalized prefixes
		def prefix_key(s: str):
			return " ".join(s.lower().split()[:6])

		new_prefix = prefix_key(normalized)
		updated = False
		old_fact = None

		for entry in list(self.facts):
			if not entry.get("is_outdated", False) and prefix_key(entry.get("fact", "")) == new_prefix and entry.get("fact", "") != normalized:
				# Mark older as outdated
				entry["is_outdated"] = True
				updated = True
				old_fact = entry.get("fact")

		new_entry = {
			"fact": normalized,
			"timestamp": now,
			"category": category,
			"confidence": float(confidence),
			"source": source,
			"status": status,
			"confidence_score": confidence_score,
			"is_outdated": False,
		}

		# Prepend (recent first)
		self.facts.insert(0, new_entry)
		self._save()

		result = {"status": "stored", "message": "stored", "fact": normalized}
		if updated:
			result.update({"updated": True, "old_fact": old_fact})
		return result

	def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
		q = (query or "").lower()
		if not q:
			return []

		hits: List[Dict[str, Any]] = []
		for entry in self.facts:
			text = (entry.get("fact") or "").lower()
			if q in text and not entry.get("is_outdated", False):
				hits.append(entry)

		# Sort by status priority and confidence
		status_priority = {
			'true': 5,           # Highest priority - verified true facts
			'experimental': 4,   # Experimental facts
			'not_verified': 3,   # Default status
			'needs_review': 2,   # Needs human review
			'false': 1           # Lowest priority - avoid if possible
		}

		hits.sort(key=lambda x: (
			status_priority.get(x.get('status', 'not_verified'), 3),  # Status priority
			x.get('confidence_score', x.get('confidence', 0.5) * 100),  # Confidence score (0-100)
			x.get('timestamp', ''),  # Most recent first
		), reverse=True)

		return hits[:limit]

	def get_timeline(self, include_outdated: bool = False) -> List[Dict[str, Any]]:
		if include_outdated:
			return list(self.facts)
		return [e for e in self.facts if not e.get("is_outdated", False)]

	def get_statistics(self) -> Dict[str, Any]:
		total = len(self.facts)
		outdated = sum(1 for e in self.facts if e.get("is_outdated", False))
		active = total - outdated
		categories: Dict[str, int] = {}
		sources: Dict[str, int] = {}
		keywords = set()

		for e in self.facts:
			categories[e.get("category", "general")] = categories.get(e.get("category", "general"), 0) + 1
			sources[e.get("source", "unknown")] = sources.get(e.get("source", "unknown"), 0) + 1
			for w in (e.get("fact", "").lower().split()):
				keywords.add(w.strip('.,'))

		stats = {
			"total_facts": total,
			"active_facts": active,
			"outdated_facts": outdated,
			"indexed_keywords": len(keywords),
			"categories": categories,
			"sources": sources,
		}
		return stats

	def reconcile_with_external(self, query: str, external_facts: List[str], source: str = "external") -> Dict[str, Any]:
		# Very simple reconciliation: mark conflicts where prefixes match but facts differ
		conflicts = []
		facts_added = []
		memory_confirmed = []

		for ext in external_facts:
			ext_norm = ext.strip()
			matched = False
			for e in list(self.facts):
				if e.get("is_outdated", False):
					continue
				if ext_norm.lower() == e.get("fact", "").lower():
					matched = True
					memory_confirmed.append(ext_norm)
					break
				# simple prefix conflict
				if " ".join(ext_norm.lower().split()[:6]) == " ".join(e.get("fact", "").lower().split()[:6]) and ext_norm.lower() != e.get("fact", "").lower():
					# mark existing outdated and add new
					e["is_outdated"] = True
					self.add_fact(ext_norm, category=e.get("category", "general"), source=source)
					conflicts.append({"old": e.get("fact"), "new": ext_norm})
					matched = True
					break
			if not matched:
				# add new fact
				self.add_fact(ext_norm, source=source)
				facts_added.append(ext_norm)

		self._save()
		return {
			"conflicts_found": len(conflicts),
			"facts_updated": conflicts,
			"facts_added": facts_added,
			"memory_confirmed": memory_confirmed,
		}

