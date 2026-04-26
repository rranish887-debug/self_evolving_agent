"""
config.py — Central configuration for the Self-Evolving ACE Agent.
Extends the original Config with memory, RL, and planner settings.
"""
import os
from pydantic import BaseModel, Field


class Config(BaseModel):
    # ── Original ACE settings (preserved) ──────────────────────────────
    agent_dir: str = os.path.dirname(os.path.abspath(__file__)) + "/agents"
    serve_web_interface: bool = True
    reload_agents: bool = True

    generator_model: str = Field(default="openrouter/google/gemini-2.0-flash-001")
    reflector_model: str = Field(default="openrouter/google/gemini-2.0-flash-001")
    curator_model:   str = Field(default="openrouter/google/gemini-2.0-flash-001")

    # ── Memory settings ─────────────────────────────────────────────────
    db_path: str = "db/agent_memory.sqlite"
    max_short_term: int = 20          # recent interactions kept in RAM
    long_term_top_k: int = 10         # top-k similar experiences to retrieve

    # ── Self-improvement ────────────────────────────────────────────────
    improvement_threshold: float = 0.6   # below this → force exploration
    reward_decay: float = 0.9            # exponential moving average decay

    # ── Planner / RL ────────────────────────────────────────────────────
    exploration_rate: float = 0.15       # ε-greedy: chance to explore
    max_plan_depth: int = 5              # max number of planned steps
    task_priority_weights: dict = Field(default_factory=lambda: {
        "unseen": 1.0,
        "low_score": 0.8,
        "high_value": 0.6,
        "routine": 0.2,
    })

    # ── Logging / Dashboard ─────────────────────────────────────────────
    log_dir: str = "logs"
    dashboard_port: int = 8501
