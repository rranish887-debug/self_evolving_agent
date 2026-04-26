"""
planner/__init__.py — Goal-Driven Planner & Task Prioritizer
─────────────────────────────────────────────────────────────
Decides:
  - What to learn next (exploration vs exploitation)
  - Which model/strategy to use
  - Task priority ordering

Uses a simple Q-table (ε-greedy RL) over task categories.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config import Config
from evaluator import evaluator

config = Config()

_QTABLE_PATH = Path("db/qtable.json")

# ──────────────────────────────────────────────────────────────────────────────
# Task representation
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Task:
    query: str
    category: str = "general"
    priority: float = 0.5       # higher = more urgent
    estimated_value: float = 0.5
    attempts: int = 0
    last_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "query": self.query,
            "category": self.category,
            "priority": self.priority,
            "estimated_value": self.estimated_value,
            "attempts": self.attempts,
            "last_score": self.last_score,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Simple Q-Table (ε-greedy RL)
# ──────────────────────────────────────────────────────────────────────────────

class QTable:
    """
    State  = task_category  (str)
    Action = strategy       ("exploit_playbook" | "explore_new" | "retry_low_score")
    """

    ACTIONS = ["exploit_playbook", "explore_new", "retry_low_score"]
    LEARNING_RATE = 0.1
    DISCOUNT = 0.9

    def __init__(self):
        self._table: Dict[str, Dict[str, float]] = {}
        self._load()

    def _default_q(self) -> Dict[str, float]:
        return {a: 0.0 for a in self.ACTIONS}

    def _load(self) -> None:
        if _QTABLE_PATH.exists():
            with open(_QTABLE_PATH) as f:
                self._table = json.load(f)

    def _save(self) -> None:
        _QTABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_QTABLE_PATH, "w") as f:
            json.dump(self._table, f, indent=2)

    def best_action(self, state: str) -> str:
        q = self._table.get(state, self._default_q())
        return max(q, key=lambda a: q[a])

    def choose_action(self, state: str, epsilon: float) -> str:
        if random.random() < epsilon:
            return random.choice(self.ACTIONS)
        return self.best_action(state)

    def update(self, state: str, action: str, reward: float, next_state: str) -> None:
        q = self._table.setdefault(state, self._default_q())
        q_next = self._table.get(next_state, self._default_q())
        best_next = max(q_next.values())
        q[action] = q[action] + self.LEARNING_RATE * (
            reward + self.DISCOUNT * best_next - q[action]
        )
        self._save()

    def q_values(self, state: str) -> Dict[str, float]:
        return self._table.get(state, self._default_q())


# ──────────────────────────────────────────────────────────────────────────────
# Planner
# ──────────────────────────────────────────────────────────────────────────────

class Planner:
    """
    Orchestrates the agent's learning agenda:
      1. Maintain a priority queue of pending tasks
      2. Select next task using ε-greedy RL
      3. Recommend strategy (exploit/explore/retry)
      4. Update Q-table after each cycle
    """

    def __init__(self):
        self._queue: List[Task] = []
        self._qtable = QTable()
        self._last_action: Optional[str] = None
        self._last_state: Optional[str] = None

    # ── Queue management ─────────────────────────────────────────────────

    def add_task(self, task: Task) -> None:
        task.priority = self._compute_priority(task)
        self._queue.append(task)
        self._queue.sort(key=lambda t: t.priority, reverse=True)

    def _compute_priority(self, task: Task) -> float:
        w = config.task_priority_weights
        if task.attempts == 0:
            return w["unseen"]
        if task.last_score < 0.4:
            return w["low_score"]
        if task.estimated_value > 0.7:
            return w["high_value"]
        return w["routine"]

    def next_task(self) -> Optional[Task]:
        if not self._queue:
            return None
        return self._queue[0]

    def pop_task(self) -> Optional[Task]:
        if not self._queue:
            return None
        return self._queue.pop(0)

    # ── RL strategy selection ────────────────────────────────────────────

    def choose_strategy(self, task: Task) -> str:
        epsilon = (
            config.exploration_rate * 2
            if evaluator.needs_exploration
            else config.exploration_rate
        )
        state = task.category
        action = self._qtable.choose_action(state, epsilon)
        self._last_state  = state
        self._last_action = action
        return action

    def record_outcome(self, task: Task, score: float) -> None:
        """Call after a cycle completes to update the Q-table."""
        if self._last_state is None or self._last_action is None:
            return
        reward = score - 0.5   # centred around 0
        next_state = task.category
        self._qtable.update(self._last_state, self._last_action, reward, next_state)

    # ── Planning context ─────────────────────────────────────────────────

    def plan_steps(self, task: Task, strategy: str) -> List[str]:
        """
        Return a human-readable plan given a task and chosen strategy.
        """
        steps = [
            f"[1] Retrieve memory context for: {task.query!r}",
            f"[2] Run Generator (strategy={strategy})",
            "[3] Run Reflector -> tag playbook bullets",
            "[4] Run Curator   -> apply delta to playbook",
            "[5] Evaluate cycle -> update EMA score",
            "[6] Update Q-table with reward signal",
            "[7] Decide next task based on priority & EMA",
        ]
        if strategy == "explore_new":
            steps.insert(2, "[1b] Inject exploration prompt to Generator")
        elif strategy == "retry_low_score":
            steps.insert(2, "[1b] Prepend past error analysis to Generator")
        return steps[:config.max_plan_depth]

    def status(self) -> Dict:
        return {
            "queue_depth": len(self._queue),
            "exploration_mode": evaluator.needs_exploration,
            "last_action": self._last_action,
            "q_table_states": list(self._qtable._table.keys()),
        }


# Singleton
planner = Planner()
