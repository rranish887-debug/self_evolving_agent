"""
evaluator/__init__.py — Self-Evaluation & Scoring Engine
─────────────────────────────────────────────────────────
Scores each ACE cycle and tracks an exponential-moving-average (EMA)
performance signal used by the planner for meta-learning decisions.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from config import Config

config = Config()


# ──────────────────────────────────────────────────────────────────────────────
# Score components
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class CycleScore:
    """Aggregate quality score for one Generator→Reflector→Curator cycle."""

    # Sub-scores in [0, 1]
    answer_completeness: float = 0.0    # How complete the final_answer is
    reflection_depth:    float = 0.0    # How deep the error analysis was
    playbook_delta:      float = 0.0    # How many net beneficial changes were made
    harmful_bullet_rate: float = 0.0    # Penalise high harmful:helpful ratio

    weights: Dict[str, float] = field(default_factory=lambda: {
        "answer_completeness": 0.35,
        "reflection_depth":    0.30,
        "playbook_delta":      0.20,
        "harmful_bullet_rate": 0.15,   # inverted (lower harm = higher score)
    })

    @property
    def composite(self) -> float:
        raw = (
            self.weights["answer_completeness"] * self.answer_completeness
            + self.weights["reflection_depth"]    * self.reflection_depth
            + self.weights["playbook_delta"]       * self.playbook_delta
            + self.weights["harmful_bullet_rate"]  * (1.0 - self.harmful_bullet_rate)
        )
        return round(min(max(raw, 0.0), 1.0), 4)

    def to_dict(self) -> Dict:
        return {
            "answer_completeness": self.answer_completeness,
            "reflection_depth":    self.reflection_depth,
            "playbook_delta":      self.playbook_delta,
            "harmful_bullet_rate": self.harmful_bullet_rate,
            "composite":           self.composite,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Evaluator
# ──────────────────────────────────────────────────────────────────────────────

class Evaluator:
    """
    Stateful evaluator that:
    1. Scores each cycle based on ACE state outputs
    2. Maintains an EMA performance signal
    3. Exposes a 'needs_exploration' flag for the planner
    """

    def __init__(self):
        self._ema: float = 0.5   # start neutral
        self._history: List[float] = []

    # ── Scoring ──────────────────────────────────────────────────────────

    def score_cycle(
        self,
        generator_output: Optional[Dict],
        reflector_output: Optional[Dict],
        curator_output:   Optional[Dict],
        playbook_before:  Optional[Dict],
        playbook_after:   Optional[Dict],
    ) -> CycleScore:
        cs = CycleScore()

        # 1. Answer completeness: words in final_answer (soft proxy)
        if generator_output and isinstance(generator_output, dict):
            ans = generator_output.get("final_answer", "")
            word_count = len(ans.split())
            cs.answer_completeness = min(word_count / 80, 1.0)   # 80 words = full score

        # 2. Reflection depth: length of error_identification + root_cause
        if reflector_output and isinstance(reflector_output, dict):
            depth_text = (
                reflector_output.get("error_identification", "")
                + reflector_output.get("root_cause_analysis", "")
                + reflector_output.get("key_insight", "")
            )
            cs.reflection_depth = min(len(depth_text) / 600, 1.0)

        # 3. Playbook delta quality
        if curator_output and isinstance(curator_output, dict):
            ops = curator_output.get("operations", [])
            add_upd = sum(1 for op in ops if op.get("type") in ("ADD", "UPDATE"))
            cs.playbook_delta = min(add_upd / 3, 1.0)
            
            # --- Advanced: Consistency Bonus ---
            # Reward Curator for addressing errors identified by Reflector
            if reflector_output and reflector_output.get("error_identification"):
                if add_upd > 0:
                    cs.playbook_delta = min(cs.playbook_delta + 0.2, 1.0)
                else:
                    cs.playbook_delta *= 0.5

        # 4. Harmful bullet rate from playbook stats
        if playbook_after and isinstance(playbook_after, dict):
            bullets = playbook_after.get("bullets", {})
            if bullets:
                helpful = sum(b.get("helpful", 0) for b in bullets.values())
                harmful = sum(b.get("harmful", 0) for b in bullets.values())
                total   = helpful + harmful + 1e-9
                cs.harmful_bullet_rate = harmful / total

        self._update_ema(cs.composite)
        return cs

    def _update_ema(self, new_score: float) -> None:
        α = 1.0 - config.reward_decay
        self._ema = α * new_score + (1 - α) * self._ema
        self._history.append(new_score)

    # ── Status ───────────────────────────────────────────────────────────

    @property
    def ema_score(self) -> float:
        return round(self._ema, 4)

    @property
    def needs_exploration(self) -> bool:
        return self._ema < config.improvement_threshold

    def trend(self, last_n: int = 10) -> str:
        if len(self._history) < 2:
            return "insufficient data"
        recent = self._history[-last_n:]
        delta = recent[-1] - recent[0]
        if delta > 0.05:
            return "improving ^"
        if delta < -0.05:
            return "declining v"
        return "stable ->"

    def summary(self) -> Dict:
        return {
            "ema_score": self.ema_score,
            "trend": self.trend(),
            "needs_exploration": self.needs_exploration,
            "cycles_evaluated": len(self._history),
        }


# Singleton
evaluator = Evaluator()
