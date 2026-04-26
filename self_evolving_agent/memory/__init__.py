"""
memory/__init__.py — Unified Memory System
──────────────────────────────────────────
Short-term : Python deque (recent N interactions, survives one session)
Long-term  : SQLite   (persists across restarts, similarity search via TF-IDF)
"""
from __future__ import annotations

import json
import math
import sqlite3
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import Config

config = Config()


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tfidf_similarity(query: str, doc: str) -> float:
    """
    Lightweight cosine-like similarity using term-frequency vectors.
    No external deps — good enough for episodic memory retrieval.
    """
    def term_freq(text: str) -> Dict[str, float]:
        words = text.lower().split()
        freq: Dict[str, float] = {}
        for w in words:
            freq[w] = freq.get(w, 0) + 1
        total = len(words) or 1
        return {w: c / total for w, c in freq.items()}

    q_tf = term_freq(query)
    d_tf = term_freq(doc)
    common = set(q_tf) & set(d_tf)
    if not common:
        return 0.0
    dot = sum(q_tf[w] * d_tf[w] for w in common)
    norm_q = math.sqrt(sum(v ** 2 for v in q_tf.values()))
    norm_d = math.sqrt(sum(v ** 2 for v in d_tf.values()))
    return dot / (norm_q * norm_d + 1e-9)


# ──────────────────────────────────────────────────────────────────────────────
# Short-term Memory
# ──────────────────────────────────────────────────────────────────────────────

class ShortTermMemory:
    """Ring-buffer of the most recent N interactions."""

    def __init__(self, maxlen: int = config.max_short_term):
        self._buffer: deque[Dict[str, Any]] = deque(maxlen=maxlen)

    def push(self, entry: Dict[str, Any]) -> None:
        entry.setdefault("timestamp", _utc_now())
        self._buffer.append(entry)

    def recent(self, n: int = 5) -> List[Dict[str, Any]]:
        items = list(self._buffer)
        return items[-n:]

    def clear(self) -> None:
        self._buffer.clear()

    def as_context_string(self, n: int = 5) -> str:
        entries = self.recent(n)
        if not entries:
            return "(no recent interactions)"
        lines = []
        for e in entries:
            ts = e.get("timestamp", "?")
            q  = e.get("query", "")
            sc = e.get("score", "n/a")
            lines.append(f"[{ts[:19]}] Q: {q!r}  score={sc}")
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Long-term Memory (SQLite)
# ──────────────────────────────────────────────────────────────────────────────

class LongTermMemory:
    """Persistent episodic memory stored in SQLite."""

    DDL = """
    CREATE TABLE IF NOT EXISTS episodes (
        id          TEXT PRIMARY KEY,
        query       TEXT NOT NULL,
        answer      TEXT,
        reflection  TEXT,
        score       REAL,
        tags        TEXT,        -- JSON list
        playbook    TEXT,        -- JSON snapshot
        timestamp   TEXT
    );
    CREATE TABLE IF NOT EXISTS performance_log (
        id          TEXT PRIMARY KEY,
        episode_id  TEXT,
        metric      TEXT,
        value       REAL,
        timestamp   TEXT
    );
    """

    def __init__(self, db_path: str = config.db_path):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(self.DDL)
        self._conn.commit()

    # ── Write ────────────────────────────────────────────────────────────

    def store_episode(
        self,
        query: str,
        answer: str,
        reflection: Optional[Dict] = None,
        score: float = 0.0,
        tags: Optional[List[str]] = None,
        playbook_snapshot: Optional[Dict] = None,
    ) -> str:
        eid = str(uuid.uuid4())
        self._conn.execute(
            "INSERT INTO episodes VALUES (?,?,?,?,?,?,?,?)",
            (
                eid,
                query,
                answer,
                json.dumps(reflection) if reflection else None,
                score,
                json.dumps(tags or []),
                json.dumps(playbook_snapshot) if playbook_snapshot else None,
                _utc_now(),
            ),
        )
        self._conn.commit()
        return eid

    def log_performance(self, episode_id: str, metric: str, value: float) -> None:
        self._conn.execute(
            "INSERT INTO performance_log VALUES (?,?,?,?,?)",
            (str(uuid.uuid4()), episode_id, metric, value, _utc_now()),
        )
        self._conn.commit()

    # ── Read ─────────────────────────────────────────────────────────────

    def retrieve_similar(self, query: str, top_k: int = config.long_term_top_k) -> List[Dict]:
        rows = self._conn.execute(
            "SELECT id, query, answer, score, tags FROM episodes ORDER BY timestamp DESC LIMIT 200"
        ).fetchall()
        scored = [
            (row, _tfidf_similarity(query, row["query"]))
            for row in rows
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [dict(r) for r, _ in scored[:top_k]]

    def performance_trend(self, last_n: int = 50) -> Dict[str, Any]:
        rows = self._conn.execute(
            "SELECT metric, value FROM performance_log ORDER BY timestamp DESC LIMIT ?",
            (last_n,),
        ).fetchall()
        by_metric: Dict[str, List[float]] = {}
        for r in rows:
            by_metric.setdefault(r["metric"], []).append(r["value"])
        return {
            m: {
                "mean": sum(vs) / len(vs),
                "min":  min(vs),
                "max":  max(vs),
                "n":    len(vs),
            }
            for m, vs in by_metric.items()
        }

    def all_episodes(self, limit: int = 500) -> List[Dict]:
        rows = self._conn.execute(
            "SELECT * FROM episodes ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        self._conn.close()


# ──────────────────────────────────────────────────────────────────────────────
# Unified Memory Facade
# ──────────────────────────────────────────────────────────────────────────────

class AgentMemory:
    """Single entry-point for all memory operations."""

    def __init__(self):
        self.short = ShortTermMemory()
        self.long  = LongTermMemory()

    def record(
        self,
        query: str,
        answer: str,
        score: float,
        reflection: Optional[Dict] = None,
        tags: Optional[List[str]] = None,
        playbook_snapshot: Optional[Dict] = None,
    ) -> str:
        """Push to both short and long-term memory."""
        self.short.push({"query": query, "answer": answer, "score": score})
        eid = self.long.store_episode(
            query=query,
            answer=answer,
            reflection=reflection,
            score=score,
            tags=tags,
            playbook_snapshot=playbook_snapshot,
        )
        self.long.log_performance(eid, "cycle_score", score)
        return eid

    def context_for_query(self, query: str) -> str:
        """Build a context string from similar past episodes + recent history."""
        similar = self.long.retrieve_similar(query, top_k=5)
        lines = ["=== Relevant Past Episodes ==="]
        for ep in similar:
            lines.append(f"  Q: {ep['query']!r}  score={ep.get('score', '?'):.2f}")
        lines.append("")
        lines.append("=== Recent Short-term Context ===")
        lines.append(self.short.as_context_string(n=5))
        return "\n".join(lines)


# Singleton used across the system
agent_memory = AgentMemory()
