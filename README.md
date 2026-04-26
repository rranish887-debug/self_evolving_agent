# 🧠 Self-Evolving ACE Agent

> **Autonomous Learning System** built on top of the original ACE-ADK (Agentic Context Engineering) project.
> Transforms the static `Generator → Reflector → Curator` pipeline into a self-improving agent that learns from every cycle.

---

## 🗺️ Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                     Self-Evolving Agent Loop                         │
│                                                                      │
│  Goal ──▶ [Planner] ──▶ [Memory Context] ──▶ [ACE Pipeline]         │
│               ▲                                     │                │
│               │                                     ▼                │
│          [RL Update] ◀── [Evaluator] ◀── Generator → Reflector      │
│               │               │               → Curator             │
│               │               ▼                                      │
│          [Q-Table]     [EMA Score Signal]                            │
│                                    │                                 │
│                                    ▼                                 │
│                              [Memory Store]                          │
│                          Short-term (deque)                          │
│                          Long-term  (SQLite)                         │
└──────────────────────────────────────────────────────────────────────┘
```

### Modules

| Module | Role |
|---|---|
| `agents/ace_agent/` | **Original ACE pipeline** — Generator, Reflector, Curator (unchanged) |
| `agent/` | **Autonomous wrapper** — orchestrates the full self-evolving loop |
| `memory/` | **Two-tier memory** — short-term deque + SQLite long-term episodic store |
| `evaluator/` | **Self-scoring engine** — multi-metric CycleScore + EMA tracking |
| `planner/` | **Goal-driven planner** — ε-greedy Q-learning task selector |
| `model/` | **Model strategy selector** — picks generation temperature + model based on EMA |
| `data_handler/` | **Data ingestion** — JSON/CSV/text task loaders + export helpers |
| `dashboard/` | **Streamlit live dashboard** — score trends, Q-table, playbook health |
| `config.py` | **Unified config** — extends original Config with new settings |

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
# Original ADK way (requires uv):
uv sync

# Or with pip:
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Add your GOOGLE_API_KEY
```

### 3. Run modes

```bash
# Interactive CLI (best for exploration)
python main.py

# Run 5 demo tasks autonomously
python main.py --demo

# Load tasks from a file
python main.py --file tasks.json

# Start Google ADK web UI (requires google-adk)
python main.py --web

# Launch Streamlit dashboard
python main.py --dashboard

# Print current agent status
python main.py --status
```

### 4. CLI commands (interactive mode)

```
Goal > math: What is the derivative of x^2?
Goal > code: Write a binary search function
Goal > :status       → print agent + evaluator + planner status
Goal > :history      → show last 10 episodes
Goal > :export       → save episodes.json + performance.csv to logs/
Goal > :quit
```

---

## 🧠 How Self-Learning Works

### 1. Memory System

Every completed cycle is recorded in two places:

- **Short-term memory** (`memory.short`): a ring-buffer of the last 20 interactions kept in RAM. Used to build a "recent context" string injected into the Generator's state.
- **Long-term memory** (`memory.long`): SQLite database (`db/agent_memory.sqlite`) storing every episode with its query, answer, score, reflection JSON, and playbook snapshot. Retrieved via lightweight TF-IDF cosine similarity — no external vector DB needed.

### 2. Self-Evaluation (Evaluator)

After each cycle the `Evaluator` computes a `CycleScore` with four sub-metrics:

| Metric | What it measures | Weight |
|---|---|---|
| `answer_completeness` | Word count of final answer | 35% |
| `reflection_depth` | Length of error analysis + insight | 30% |
| `playbook_delta` | ADD/UPDATE operations in this cycle | 20% |
| `harmful_bullet_rate` | Ratio of harmful:helpful tags (inverted) | 15% |

The composite score feeds an **Exponential Moving Average (EMA)** with decay α = 0.1. When EMA drops below `improvement_threshold` (default 0.6), the agent switches to **exploration mode**.

### 3. Decision-Making (Planner + Q-Table)

The `Planner` maintains a priority queue of `Task` objects. Each task gets a priority based on:
- `unseen` (never tried): highest priority
- `low_score` (< 0.4 last attempt): high priority
- `high_value` (estimated value > 0.7): medium priority
- `routine`: lowest priority

Strategy selection uses **ε-greedy Q-learning**:
- **State** = task category (e.g., `"math"`, `"code"`)
- **Actions** = `exploit_playbook` | `explore_new` | `retry_low_score`
- **Reward** = cycle score − 0.5 (centred)
- Q-values are persisted to `db/qtable.json` and survive restarts

When the agent is in exploration mode (low EMA), ε is doubled — forcing it to try novel strategies.

### 4. Self-Improvement Loop

```
Task ──▶ Plan (RL) ──▶ Inject Memory ──▶ ACE Cycle
                                              │
                                   ┌──────────┴──────────┐
                                   │                     │
                            Score Cycle          Update Playbook
                                   │
                            Update EMA
                                   │
                     Score < threshold?
                        ├── YES → re-queue task (max 3 retries)
                        └── NO  → next task
```

### 5. Playbook Evolution

The original ACE `Playbook` (stored in `app:playbook` state) evolves with every cycle:
- **ADD**: new insights from high-quality reflections
- **UPDATE**: existing bullets improved by Curator
- **REMOVE**: harmful/duplicate bullets pruned
- **Tag statistics** (helpful/harmful/neutral) accumulate per bullet — the Evaluator uses these to penalise playbooks with high harmful ratios

---

## 📁 Project Structure

```
self_evolving_agent/
├── main.py                      # Entry point (CLI / web / demo / dashboard)
├── config.py                    # Unified config (extends original)
├── requirements.txt
├── .env.example
│
├── agents/ace_agent/            # ✅ ORIGINAL CODE (preserved)
│   ├── agent.py
│   ├── schemas/
│   │   ├── playbook.py
│   │   └── delta.py
│   └── sub_agents/
│       ├── generator.py
│       ├── reflector.py
│       └── curator.py
│
├── agent/                       # 🆕 Autonomous loop
│   └── __init__.py
├── memory/                      # 🆕 Two-tier memory
│   └── __init__.py
├── evaluator/                   # 🆕 Self-scoring
│   └── __init__.py
├── planner/                     # 🆕 RL task planner
│   └── __init__.py
├── model/                       # 🆕 Strategy selector
│   └── __init__.py
├── data_handler/                # 🆕 Data ingestion
│   └── __init__.py
├── dashboard/                   # 🆕 Streamlit UI
│   └── app.py
│
├── db/                          # Auto-created at runtime
│   ├── agent_memory.sqlite
│   └── qtable.json
└── logs/                        # Auto-created at runtime
    ├── agent.log
    ├── episodes.json
    └── performance.csv
```

---

## ⚙️ Configuration (`config.py`)

| Setting | Default | Description |
|---|---|---|
| `generator_model` | `gemini-2.5-flash` | LLM for answer generation |
| `reflector_model` | `gemini-2.5-flash` | LLM for error analysis |
| `curator_model` | `gemini-2.5-flash` | LLM for playbook curation |
| `max_short_term` | 20 | Short-term memory buffer size |
| `long_term_top_k` | 10 | Episodes to retrieve per query |
| `improvement_threshold` | 0.6 | EMA below this → force exploration |
| `reward_decay` | 0.9 | EMA decay factor (α = 1 − decay) |
| `exploration_rate` | 0.15 | ε-greedy base exploration rate |
| `max_plan_depth` | 5 | Max steps shown in plan |
| `dashboard_port` | 8501 | Streamlit port |

---

## 🎮 Task File Format

**JSON** (`tasks.json`):
```json
[
  {"query": "What is the sum of 1 to 100?", "category": "math"},
  {"query": "Write a quicksort in Python", "category": "code"}
]
```

**CSV** (`tasks.csv`):
```csv
query,category
What is entropy?,physics
Explain recursion,code
```

**Plain text** (`tasks.txt`):
```
What is 2 + 2?
Explain neural networks
Write a haiku about Python
```

---

## 🔮 Further Improvements (Advanced)

### Immediate wins
1. **FAISS vector store** — replace TF-IDF with sentence-transformer embeddings for richer similarity search
2. **Async pipeline** — run Reflector and Evaluator concurrently after Generator completes
3. **Ground-truth feedback** — pass known correct answers to Reflector for supervised scoring

### Medium-term
4. **Multi-agent routing** — add a Router agent that delegates tasks to specialised sub-agents (math, code, language)
5. **Curriculum learning** — auto-generate harder variants of low-scoring tasks
6. **Playbook compression** — periodically cluster similar bullets and merge them (prevent playbook bloat)

### Advanced
7. **PPO/A3C** — replace the Q-table with a neural policy for continuous state spaces
8. **Online fine-tuning** — when a local model is used, apply LoRA updates from high-confidence episodes
9. **Multi-session continuity** — expose a REST API so the agent persists across user sessions
10. **Adversarial self-play** — generate challenging counter-examples automatically to stress-test the playbook
