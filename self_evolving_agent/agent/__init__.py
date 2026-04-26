"""
agent/__init__.py — Autonomous Self-Evolving Agent Loop
────────────────────────────────────────────────────────
Wraps the original ACE pipeline inside an intelligent agent that:
  1. Accepts goals as input
  2. Plans actions (via Planner)
  3. Executes the ACE cycle (Generator → Reflector → Curator)
  4. Scores the result (Evaluator)
  5. Stores experiences (Memory)
  6. Decides what to learn next (RL Planner)
  7. Self-adjusts strategy based on EMA performance
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

# Suppress noisy Windows SSL cleanup error (cosmetic, not functional)
import warnings
warnings.filterwarnings("ignore", message=".*SSL.*")

from evaluator import evaluator
from memory import agent_memory
from planner import Task, planner
from config import Config

config = Config()

# ── Logging ──────────────────────────────────────────────────────────────────
Path(config.log_dir).mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(f"{config.log_dir}/agent.log"),
        logging.StreamHandler(),
    ],
)
# Suppress Windows asyncio SSL transport errors (cosmetic noise)
logging.getLogger("asyncio").setLevel(logging.CRITICAL)
log = logging.getLogger("SelfEvolvingAgent")

# ── Rate limiting ─────────────────────────────────────────────────────────────
_last_api_call: float = 0.0
_MIN_CALL_INTERVAL = 5.0   # seconds between API calls to avoid 429


# ──────────────────────────────────────────────────────────────────────────────
# ACE Pipeline runner (framework-agnostic, uses stored state dict)
# ──────────────────────────────────────────────────────────────────────────────

def _run_ace_cycle_sync(query: str, state: Dict) -> Dict:
    """
    Simulates one ACE cycle using stored playbook state.
    When the real Google ADK is available, swap this for the actual runner.
    This stub lets the self-evolving loop work in any Python env.
    """
    import sys, os
    sys.path.insert(0, str(Path(__file__).parent.parent))

    # Use InMemoryRunner to execute the real ACE components
    async def run_ace_async():
        from google.adk.runners import InMemoryRunner
        from google.genai import types
        from agents.ace_agent.agent import root_agent

        log.info(f"   [ACE] Executing real agent pipeline for: {query!r}")
        
        # Initialize runner with app_name
        import hashlib
        task_hash = hashlib.md5(query.encode()).hexdigest()[:8]
        user_id, session_id, app_name = "user", f"session_{task_hash}", "ace_agent"
        runner = InMemoryRunner(agent=root_agent, app_name=app_name)
        runner.auto_create_session = True
        
        # Setup session (using internal helper to ensure creation)
        session = await runner._get_or_create_session(user_id=user_id, session_id=session_id)
        session.state.update(state)
        
        # Prepare input message
        new_msg = types.Content(role="user", parts=[types.Part(text=query)])
        
        # Run the pipeline (using the async generator)
        event_count = 0
        async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=new_msg):
            event_count += 1
            # Log progress for visibility
            if event.author:
                log.info(f"      [ACE Event] {event.author} is running...")

        log.info(f"      [ACE] Pipeline finished. Total events: {event_count}")

        # RE-FETCH SESSION: The runner uses a copy internally; we need the latest one from storage.
        session = await runner._get_or_create_session(user_id=user_id, session_id=session_id)
        final_state = dict(session.state)
        
        # log.info(f"   [DEBUG] Session State Keys: {list(final_state.keys())}")
        if "generator_output" in final_state:
            g_out = str(final_state['generator_output']).encode('ascii', 'ignore').decode()
            log.info(f"   [DEBUG] generator_output content: {g_out}")

        return {
            "generator_output": final_state.get("generator_output", {}),
            "reflector_output": final_state.get("reflector_output", {}),
            "curator_output": final_state.get("curator_output", {}),
            "app:playbook": final_state.get("app:playbook", {}),
            "user_query": final_state.get("user_query", query),
        }

    try:
        import asyncio
        global _last_api_call
        # Enforce minimum interval between API calls to avoid quota errors
        elapsed = time.time() - _last_api_call
        if elapsed < _MIN_CALL_INTERVAL:
            wait = _MIN_CALL_INTERVAL - elapsed
            log.info(f"   [ACE] Rate limiting: waiting {wait:.1f}s before API call...")
            time.sleep(wait)
        _last_api_call = time.time()
        return asyncio.run(run_ace_async())

    except Exception as e:
        import traceback
        err_str = str(e)
        # Handle quota exhaustion specifically - wait and inform user
        if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower():
            wait_time = 30
            import re
            m = re.search(r"retryDelay.*?(\d+)s", err_str)
            if m:
                wait_time = int(m.group(1)) + 5
            log.warning(f"   [ACE] API quota exceeded. Waiting {wait_time}s before next attempt...")
            time.sleep(wait_time)
        else:
            log.warning(f"ACE execution failed ({type(e).__name__}: {e}); using minimal stub.\n{traceback.format_exc()}")
        return {
            "generator_output": {"final_answer": f"Stub: {query}", "reasoning": [], "bullet_ids": []},
            "reflector_output": {"reasoning": "", "error_identification": "", "root_cause_analysis": "",
                                 "correct_approach": "", "key_insight": "", "bullet_tags": []},
            "curator_output": {"reasoning": "", "operations": []},
            "app:playbook": state.get("app:playbook", {}),
            "user_query": query,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Self-Evolving Agent
# ──────────────────────────────────────────────────────────────────────────────

class SelfEvolvingAgent:
    """
    The autonomous wrapper around the ACE pipeline.

    Lifecycle per task:
        plan → inject_context → run_ace → evaluate → remember → rl_update → repeat
    """

    def __init__(self):
        self._state: Dict[str, Any] = {}   # persistent session state
        self._cycle_count = 0

    # ── Public API ───────────────────────────────────────────────────────

    def submit_goal(self, query: str, category: str = "general") -> None:
        """Add a goal/task to the planner queue."""
        task = Task(query=query, category=category)
        planner.add_task(task)
        log.info(f"Goal submitted: {query!r}  category={category}")

    def run_next(self) -> Optional[Dict[str, Any]]:
        """
        Execute one full self-evolving cycle for the highest-priority task.
        Returns a result dict or None if queue is empty.
        """
        task = planner.pop_task()
        if task is None:
            log.info("Task queue empty.")
            return None

        task.attempts += 1
        return self._execute_cycle(task)

    def run_all(self, max_cycles: int = 100) -> None:
        """Drain the task queue up to max_cycles."""
        for _ in range(max_cycles):
            result = self.run_next()
            if result is None:
                break
        log.info(f"Agent completed. EMA score={evaluator.ema_score:.3f}  trend={evaluator.trend()}")

    # ── Core cycle ───────────────────────────────────────────────────────

    def _execute_cycle(self, task: Task) -> Dict[str, Any]:
        self._cycle_count += 1
        log.info(f"== Cycle #{self._cycle_count}  task={task.query!r}  cat={task.category}")

        # 1. Plan
        strategy = planner.choose_strategy(task)
        steps = planner.plan_steps(task, strategy)
        log.info(f"   Strategy: {strategy}")
        for s in steps:
            log.info(f"   {s}")

        # 2. Inject memory context into state
        memory_ctx = agent_memory.context_for_query(task.query)
        self._state["memory_context"] = memory_ctx
        self._state["user_query"] = task.query
        if "app:playbook" not in self._state:
            self._state["app:playbook"] = {}

        # 3. Run ACE pipeline
        playbook_before = dict(self._state.get("app:playbook") or {})
        result = _run_ace_cycle_sync(task.query, self._state)
        self._state.update(result)
        playbook_after = dict(self._state.get("app:playbook") or {})

        # 4. Evaluate
        score_obj = evaluator.score_cycle(
            generator_output=result.get("generator_output"),
            reflector_output=result.get("reflector_output"),
            curator_output=result.get("curator_output"),
            playbook_before=playbook_before,
            playbook_after=playbook_after,
        )
        score = score_obj.composite
        task.last_score = score
        log.info(f"   Score: {score:.4f}  EMA: {evaluator.ema_score:.4f}  Trend: {evaluator.trend()}")

        # 5. Store in memory

        episode_id = agent_memory.record(
            query=task.query,
            answer=result.get("generator_output", {}).get("final_answer", ""),
            score=score,
            reflection=result.get("reflector_output"),
            tags=[task.category],
            playbook_snapshot=playbook_after,
        )

        # 6. RL update
        planner.record_outcome(task, score)

        # 7. Auto-enqueue follow-up if score is low (meta-learning signal)
        if score < config.improvement_threshold and task.attempts < 2:
            retry = Task(
                query=task.query,
                category=task.category,
                attempts=task.attempts,
                last_score=score,
                estimated_value=0.9,   # high value: we need to improve this
            )
            planner.add_task(retry)
            log.info(f"   Low score — re-queued for retry (attempt {task.attempts + 1})")

        return {
            "episode_id": episode_id,
            "cycle": self._cycle_count,
            "query": task.query,
            "strategy": strategy,
            "score": score_obj.to_dict(),
            "ema_score": evaluator.ema_score,
            "trend": evaluator.trend(),
            "answer": result["generator_output"].get("final_answer", ""),
            "planner_status": planner.status(),
            "evaluator_summary": evaluator.summary(),
        }

    # ── Introspection ────────────────────────────────────────────────────

    def status(self) -> Dict:
        return {
            "cycles_run": self._cycle_count,
            "evaluator": evaluator.summary(),
            "planner": planner.status(),
            "memory": {
                "short_term_size": len(agent_memory.short._buffer),
                "performance_trend": agent_memory.long.performance_trend(last_n=20),
            },
        }


# Singleton agent
agent = SelfEvolvingAgent()
