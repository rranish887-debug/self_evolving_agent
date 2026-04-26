"""
main.py — Self-Evolving ACE Agent — Entry Point
─────────────────────────────────────────────────
Modes:
  python main.py                 → interactive CLI
  python main.py --web           → start FastAPI + Google ADK web UI
  python main.py --demo          → run 5 demo tasks autonomously
  python main.py --dashboard     → launch Streamlit dashboard
  python main.py --status        → print agent status and exit
"""
from __future__ import annotations

import argparse
import json
import sys
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# Fix for Windows asyncio SSL transport error
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Load environment variables from .env file
load_dotenv()

# ── Ensure local modules are importable ──────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from agent import agent
from config import Config
from data_handler import enqueue_from_file, export_episodes_json, export_performance_csv
from evaluator import evaluator
from memory import agent_memory
from planner import Task, planner

config = Config()


# ──────────────────────────────────────────────────────────────────────────────
# Modes
# ──────────────────────────────────────────────────────────────────────────────

def run_interactive_cli():
    print("\n+----------------------------------------------+")
    print("|   Self-Evolving ACE Agent  -  Interactive    |")
    print("+----------------------------------------------+\n")
    print("Commands:  :status   :history   :export   :quit\n")

    while True:
        try:
            raw = input("Goal > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not raw:
            continue

        if raw == ":quit":
            break

        if raw == ":status":
            print(json.dumps(agent.status(), indent=2))
            continue

        if raw == ":history":
            episodes = agent_memory.long.all_episodes(limit=10)
            for ep in episodes:
                print(f"  [{ep['timestamp'][:19]}] score={ep['score']:.3f}  q={ep['query']!r}")
            continue

        if raw == ":export":
            export_episodes_json()
            export_performance_csv()
            print("Exported to logs/episodes.json and logs/performance.csv")
            continue

        # Parse optional category prefix:  "math: What is 2+2?"
        if ":" in raw and raw.index(":") < 20:
            cat, query = raw.split(":", 1)
            category = cat.strip()
            query = query.strip()
        else:
            query = raw
            category = "general"

        agent.submit_goal(query, category=category)
        result = agent.run_next()

        if result:
            print(f"\n  Answer  : {result['answer']}")
            print(f"  Score   : {result['score']['composite']:.4f}  (EMA: {result['ema_score']:.4f}  {result['trend']})")
            print(f"  Strategy: {result['strategy']}")
            print()


def run_demo():
    demo_tasks = [
        ("math",    "What is the sum of angles in a pentagon?"),
        ("logic",   "If all cats are mammals and Whiskers is a cat, is Whiskers a mammal?"),
        ("code",    "Write a Python function to compute Fibonacci numbers."),
        ("math",    "What is 17 * 23?"),
        ("general", "Explain the concept of entropy in thermodynamics."),
    ]
    print("\n[!] Demo Mode - running 5 autonomous tasks\n")
    for cat, q in demo_tasks:
        agent.submit_goal(q, category=cat)

    agent.run_all(max_cycles=10)

    print("\n-- Final Agent Status --------------------------")
    print(json.dumps(agent.status(), indent=2))

    export_episodes_json()
    export_performance_csv()
    print("\nExported logs to logs/")


def run_web():
    import uvicorn
    from fastapi import FastAPI
    from google.adk.cli.fast_api import get_fast_api_app

    app = get_fast_api_app(
        agents_dir=config.agent_dir,
        web=config.serve_web_interface,
        reload_agents=config.reload_agents,
    )
    print("\n🌐  Web UI: http://localhost:8080\n")
    uvicorn.run(app, host="0.0.0.0", port=8080)


def run_dashboard():
    """Launch the Streamlit dashboard."""
    import subprocess
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        str(Path(__file__).parent / "dashboard" / "app.py"),
        "--server.port", str(config.dashboard_port),
    ])


def print_status():
    print(json.dumps(agent.status(), indent=2))


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Self-Evolving ACE Agent")
    parser.add_argument("--web",       action="store_true", help="Start Google ADK web UI")
    parser.add_argument("--demo",      action="store_true", help="Run 5 demo tasks")
    parser.add_argument("--dashboard", action="store_true", help="Launch Streamlit dashboard")
    parser.add_argument("--status",    action="store_true", help="Print status and exit")
    parser.add_argument("--file",      type=str, default=None, help="Load tasks from file")
    args = parser.parse_args()

    if args.status:
        print_status()
    elif args.demo:
        run_demo()
    elif args.web:
        run_web()
    elif args.dashboard:
        run_dashboard()
    elif args.file:
        n = enqueue_from_file(args.file)
        print(f"Loaded {n} tasks. Running...")
        agent.run_all()
    else:
        run_interactive_cli()


if __name__ == "__main__":
    main()
