"""
model/__init__.py — Model Strategy Selector
─────────────────────────────────────────────
Implements dynamic model selection based on:
  - Task category
  - Current EMA performance
  - Exploration flag from evaluator

This is the "which model/strategy to use" part of the Decision-Making Engine.
"""
from __future__ import annotations

from typing import Dict, Tuple

from config import Config
from evaluator import evaluator

config = Config()


# ──────────────────────────────────────────────────────────────────────────────
# Strategy registry
# ──────────────────────────────────────────────────────────────────────────────

STRATEGY_PROFILES: Dict[str, Dict] = {
    "exploit_playbook": {
        "description": "Use full playbook context; trust accumulated knowledge.",
        "generator_temp": 0.3,
        "reflector_depth": "standard",
        "curator_ops_limit": 2,
        "prompt_prefix": "Use the playbook carefully. Prioritise proven strategies.",
    },
    "explore_new": {
        "description": "Push generator to try novel approaches; higher temperature.",
        "generator_temp": 0.9,
        "reflector_depth": "deep",
        "curator_ops_limit": 3,
        "prompt_prefix": "Explore creative solutions. Don't rely solely on the playbook.",
    },
    "retry_low_score": {
        "description": "Focus on previous failure points. Patch known gaps.",
        "generator_temp": 0.5,
        "reflector_depth": "deep",
        "curator_ops_limit": 3,
        "prompt_prefix": "Previous attempts scored low. Analyse past errors carefully.",
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# Model selector
# ──────────────────────────────────────────────────────────────────────────────

class ModelSelector:
    """
    Selects the concrete model names and strategy profile to use
    for a given task. Can auto-downgrade to cheaper models when
    the agent is performing well (cost efficiency).
    """

    def select(self, strategy: str, category: str = "general") -> Tuple[str, str, str, Dict]:
        """
        Returns: (generator_model, reflector_model, curator_model, strategy_profile)
        """
        profile = STRATEGY_PROFILES.get(strategy, STRATEGY_PROFILES["exploit_playbook"])

        # Auto-select model tier
        if evaluator.ema_score > 0.8:
            # Performing well → use efficient/cheaper model
            gen_model = ref_model = cur_model = config.generator_model
        elif evaluator.needs_exploration:
            # Struggling → use best available model
            gen_model = ref_model = cur_model = config.generator_model
        else:
            gen_model = config.generator_model
            ref_model = config.reflector_model
            cur_model = config.curator_model

        return gen_model, ref_model, cur_model, profile

    def describe(self, strategy: str) -> str:
        return STRATEGY_PROFILES.get(strategy, {}).get("description", "unknown strategy")


model_selector = ModelSelector()
