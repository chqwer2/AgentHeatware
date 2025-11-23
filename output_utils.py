from __future__ import annotations
from typing import List, Dict, Any
import csv
import os


def ensure_dir_for_file(path: str) -> None:
    """
    Create parent directory for a file path if it does not exist.
    """
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def save_agent_daily_log(
    agent_daily_log: List[Dict[str, Any]],
    path: str = "output/agent_daily_log.csv",
) -> None:
    """
    Save the full per-day, per-agent log (every move/decision)
    into a CSV file.

    Each row = (day, agent_id, actions, states, LLM info, disease info, etc.)
    """
    if not agent_daily_log:
        print("[OUTPUT] No agent_daily_log entries to save.")
        return

    ensure_dir_for_file(path)

    # Use keys from first row as header
    fieldnames = list(agent_daily_log[0].keys())

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(agent_daily_log)

    print(f"[OUTPUT] Saved agent_daily_log with {len(agent_daily_log)} rows to {path}")


def save_daily_summary(
    history: List[Dict[str, Any]],
    path: str = "output/daily_summary.csv",
) -> None:
    """
    Save day-level summary (avg_health, avg_mental) to CSV.
    """
    if not history:
        print("[OUTPUT] No daily history to save.")
        return

    ensure_dir_for_file(path)

    fieldnames = list(history[0].keys())

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)

    print(f"[OUTPUT] Saved daily summary with {len(history)} rows to {path}")
