"""
data_handler/__init__.py — Data Ingestion & Preprocessing
──────────────────────────────────────────────────────────
Provides utilities to:
  - Load tasks from JSON/CSV/plain-text files
  - Stream tasks into the planner queue
  - Export performance data for the dashboard
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterator, List, Optional

from planner import Task, planner
from memory import agent_memory


# ──────────────────────────────────────────────────────────────────────────────
# Loaders
# ──────────────────────────────────────────────────────────────────────────────

def load_tasks_from_json(path: str | Path) -> List[Task]:
    """
    Expects a JSON array like:
      [{"query": "...", "category": "math"}, ...]
    """
    data = json.loads(Path(path).read_text())
    return [Task(query=d["query"], category=d.get("category", "general")) for d in data]


def load_tasks_from_csv(path: str | Path) -> List[Task]:
    """
    Expects columns: query, [category]
    """
    tasks = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tasks.append(Task(
                query=row["query"],
                category=row.get("category", "general"),
            ))
    return tasks


def load_tasks_from_text(path: str | Path) -> List[Task]:
    """One query per line."""
    lines = Path(path).read_text().splitlines()
    return [Task(query=line.strip()) for line in lines if line.strip()]


def enqueue_from_file(path: str | Path) -> int:
    """Auto-detect format and push all tasks into the planner."""
    p = Path(path)
    if p.suffix == ".json":
        tasks = load_tasks_from_json(p)
    elif p.suffix == ".csv":
        tasks = load_tasks_from_csv(p)
    else:
        tasks = load_tasks_from_text(p)
    for t in tasks:
        planner.add_task(t)
    return len(tasks)


# ──────────────────────────────────────────────────────────────────────────────
# Export helpers (for dashboard)
# ──────────────────────────────────────────────────────────────────────────────

def export_episodes_json(out_path: str | Path = "logs/episodes.json") -> None:
    episodes = agent_memory.long.all_episodes()
    Path(out_path).write_text(json.dumps(episodes, indent=2))


def export_performance_csv(out_path: str | Path = "logs/performance.csv") -> None:
    conn = agent_memory.long._conn
    rows = conn.execute(
        "SELECT episode_id, metric, value, timestamp FROM performance_log ORDER BY timestamp"
    ).fetchall()
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode_id", "metric", "value", "timestamp"])
        for r in rows:
            writer.writerow([r["episode_id"], r["metric"], r["value"], r["timestamp"]])
