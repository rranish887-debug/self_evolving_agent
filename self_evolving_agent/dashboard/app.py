"""
dashboard/app.py — Self-Evolving Agent Dashboard
──────────────────────────────────────────────────
Live visualisation of:
  - EMA performance over time
  - Playbook health (helpful vs harmful bullets)
  - Task queue and completion stats
  - Recent episode log
  - Q-table heatmap

Run:  streamlit run dashboard/app.py
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

st.set_page_config(
    page_title="Self-Evolving Agent Dashboard",
    page_icon="🧠",
    layout="wide",
)

# ── Attempt to import live data ───────────────────────────────────────────────
try:
    from memory import agent_memory
    from evaluator import evaluator
    from planner import planner
    from config import Config

    config = Config()
    LIVE = True
except Exception:
    LIVE = False

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_episodes_from_db(db_path="db/agent_memory.sqlite"):
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM episodes ORDER BY timestamp DESC LIMIT 300"
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception:
        return []


def load_perf_log(db_path="db/agent_memory.sqlite"):
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM performance_log ORDER BY timestamp"
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception:
        return []


def load_qtable(path="db/qtable.json"):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return {}


# ──────────────────────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────────────────────

st.title("🧠 Self-Evolving ACE Agent — Live Dashboard")
st.caption("Auto-refreshes every 10 seconds when data is available.")

col_refresh, _ = st.columns([1, 5])
with col_refresh:
    if st.button("🔄 Refresh"):
        st.rerun()

# ── Top KPI row ───────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)

episodes = load_episodes_from_db()
perf_log = load_perf_log()

scores = [e["score"] for e in episodes if e.get("score") is not None]
ema_val = scores[-1] if scores else 0.0   # simple last value as proxy

with k1:
    st.metric("Total Episodes", len(episodes))
with k2:
    avg_score = round(sum(scores) / len(scores), 4) if scores else 0.0
    st.metric("Avg Score", avg_score)
with k3:
    recent_5 = scores[-5:] if len(scores) >= 5 else scores
    trend = "↑ improving" if (len(recent_5) > 1 and recent_5[-1] > recent_5[0]) else "→ stable"
    st.metric("Recent Trend", trend)
with k4:
    qtable = load_qtable()
    st.metric("Q-Table States", len(qtable))

st.divider()

# ── Score over time ───────────────────────────────────────────────────────────
st.subheader("📈 Performance Over Time (Cycle Score)")

if perf_log:
    import pandas as pd
    df_perf = pd.DataFrame(perf_log)
    df_perf["timestamp"] = pd.to_datetime(df_perf["timestamp"])
    df_cycle = df_perf[df_perf["metric"] == "cycle_score"].copy()
    if not df_cycle.empty:
        df_cycle = df_cycle.sort_values("timestamp").reset_index(drop=True)
        df_cycle["ema"] = df_cycle["value"].ewm(span=5).mean()
        st.line_chart(df_cycle[["value", "ema"]].rename(columns={"value": "raw_score", "ema": "EMA"}))
    else:
        st.info("No cycle_score entries yet.")
else:
    st.info("No performance log entries yet. Run some tasks first.")

st.divider()

# ── Episode log ───────────────────────────────────────────────────────────────
st.subheader("📋 Recent Episodes")
if episodes:
    import pandas as pd
    df_ep = pd.DataFrame(episodes)[["timestamp", "query", "score", "tags"]]
    df_ep["timestamp"] = df_ep["timestamp"].str[:19]
    df_ep["score"] = df_ep["score"].round(4)
    st.dataframe(df_ep.head(30), use_container_width=True)
else:
    st.info("No episodes recorded yet.")

st.divider()

# ── Q-Table heatmap ───────────────────────────────────────────────────────────
st.subheader("🎮 Q-Table (RL Strategy Values)")
if qtable:
    import pandas as pd
    rows = []
    for state, actions in qtable.items():
        row = {"state": state}
        row.update(actions)
        rows.append(row)
    df_q = pd.DataFrame(rows).set_index("state")
    st.dataframe(df_q.style.background_gradient(cmap="RdYlGn", axis=None), use_container_width=True)
else:
    st.info("Q-table empty — run tasks to populate it.")

st.divider()

# ── Playbook health ───────────────────────────────────────────────────────────
st.subheader("📚 Playbook Bullet Health")
if episodes:
    latest_pb_json = next(
        (e["playbook"] for e in episodes if e.get("playbook")), None
    )
    if latest_pb_json:
        try:
            pb_data = json.loads(latest_pb_json)
            bullets = pb_data.get("bullets", {})
            if bullets:
                import pandas as pd
                records = [
                    {
                        "id": b["id"],
                        "section": b["section"],
                        "content": b["content"][:60],
                        "helpful": b.get("helpful", 0),
                        "harmful": b.get("harmful", 0),
                        "neutral": b.get("neutral", 0),
                    }
                    for b in bullets.values()
                ]
                df_pb = pd.DataFrame(records)
                st.dataframe(df_pb, use_container_width=True)

                total_helpful = df_pb["helpful"].sum()
                total_harmful = df_pb["harmful"].sum()
                ph1, ph2 = st.columns(2)
                with ph1:
                    st.metric("Total Helpful Tags", int(total_helpful))
                with ph2:
                    st.metric("Total Harmful Tags", int(total_harmful))
            else:
                st.info("Playbook is empty.")
        except Exception as e:
            st.warning(f"Could not parse playbook snapshot: {e}")
    else:
        st.info("No playbook snapshot in episodes yet.")
else:
    st.info("Run tasks to populate playbook health data.")

# ── Auto-refresh hint ─────────────────────────────────────────────────────────
st.caption("Tip: Use `st.rerun()` or press Refresh to reload live data.")
