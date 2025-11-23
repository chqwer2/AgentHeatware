from __future__ import annotations
import os

import pandas as pd
import matplotlib.pyplot as plt


OUTPUT_DIR = "output"
LOG_PATH = os.path.join(OUTPUT_DIR, "agent_daily_log.csv")
SUMMARY_PATH = os.path.join(OUTPUT_DIR, "daily_summary.csv")


def ensure_output_dir() -> None:
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    if not os.path.exists(LOG_PATH):
        raise FileNotFoundError(
            f"{LOG_PATH} not found. "
            "Run `python main.py` first to generate simulation outputs."
        )
    if not os.path.exists(SUMMARY_PATH):
        raise FileNotFoundError(
            f"{SUMMARY_PATH} not found. "
            "Run `python main.py` first to generate simulation outputs."
        )

    log_df = pd.read_csv(LOG_PATH)
    summary_df = pd.read_csv(SUMMARY_PATH)
    return log_df, summary_df


def plot_daily_averages(summary_df: pd.DataFrame) -> None:
    """
    Plot avg_health and avg_mental over days.
    """
    plt.figure(figsize=(8, 4))
    plt.plot(summary_df["day"], summary_df["avg_health"], marker="o", label="Avg Health")
    plt.plot(summary_df["day"], summary_df["avg_mental"], marker="s", label="Avg Mental")
    plt.xlabel("Day")
    plt.ylabel("Average level (0–1)")
    plt.title("Daily average health and mental state")
    plt.legend()
    plt.grid(True)

    out_path = os.path.join(OUTPUT_DIR, "plot_avg_health_mental.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[VIZ] Saved {out_path}")


def plot_final_health_by_age_group(log_df: pd.DataFrame) -> None:
    """
    Look at the final day's health distribution by age_group.
    """
    last_day = log_df["day"].max()
    last_df = log_df[log_df["day"] == last_day].copy()

    # One row per agent for final day (if multiple rows, we average by agent_id)
    last_df = (
        last_df.groupby(["agent_id", "age_group"], as_index=False)["health_after"]
        .mean()
    )

    plt.figure(figsize=(8, 4))
    # Boxplot of health by age_group
    # Prepare data grouped by age_group
    groups = sorted(last_df["age_group"].unique())
    data = [last_df[last_df["age_group"] == g]["health_after"] for g in groups]

    plt.boxplot(data, labels=groups, showfliers=False)
    plt.xlabel("Age group")
    plt.ylabel("Final health (0–1)")
    plt.title(f"Final-day health distribution by age group (day={last_day})")
    plt.xticks(rotation=30, ha="right")

    out_path = os.path.join(OUTPUT_DIR, "plot_final_health_by_age_group.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[VIZ] Saved {out_path}")


def plot_heat_conditions_by_age_group(log_df: pd.DataFrame) -> None:
    """
    Count how many agents ended up with each heat_condition,
    aggregated by age_group (final day).
    """
    last_day = log_df["day"].max()
    last_df = log_df[log_df["day"] == last_day].copy()

    # One row per agent
    last_df = (
        last_df.groupby(["agent_id", "age_group", "heat_condition"], as_index=False)
        .agg({"heat_condition_severity": "mean"})
    )

    # Filter out "none"
    cond_df = last_df[last_df["heat_condition"] != "none"].copy()
    if cond_df.empty:
        print("[VIZ] No agents with heat_condition != 'none' on final day; skipping condition plot.")
        return

    # Count per (age_group, heat_condition)
    count_df = (
        cond_df.groupby(["age_group", "heat_condition"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )

    # Pivot for stacked bar
    pivot = count_df.pivot(index="age_group", columns="heat_condition", values="count").fillna(0)
    pivot = pivot.sort_index()

    plt.figure(figsize=(8, 5))
    bottom = None
    for cond in pivot.columns:
        values = pivot[cond].values
        if bottom is None:
            bottom = values
            plt.bar(pivot.index, values, label=cond)
        else:
            plt.bar(pivot.index, values, bottom=bottom, label=cond)
            bottom = bottom + values

    plt.xlabel("Age group")
    plt.ylabel("Number of agents (final day)")
    plt.title("Heat-related conditions by age group (final day)")
    plt.xticks(rotation=30, ha="right")
    plt.legend(title="Condition")

    out_path = os.path.join(OUTPUT_DIR, "plot_heat_conditions_by_age_group.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[VIZ] Saved {out_path}")


def plot_trajectories_for_sample_agents(log_df: pd.DataFrame, n: int = 5) -> None:
    """
    Plot health/mental trajectories for a few sample agents.
    """
    # pick n distinct agent_ids
    agent_ids = sorted(log_df["agent_id"].unique())[:n]

    plt.figure(figsize=(9, 6))

    for i, aid in enumerate(agent_ids):
        sub = log_df[log_df["agent_id"] == aid].sort_values("day")
        # offset in y so multiple lines do not overlap? Let's keep raw to see actual health.
        plt.plot(sub["day"], sub["health_after"], marker="o", linestyle="-", label=f"Agent {aid} H")
        plt.plot(sub["day"], sub["mental_after"], marker="x", linestyle="--", label=f"Agent {aid} M")

    plt.xlabel("Day")
    plt.ylabel("Health/Mental (0–1)")
    plt.title(f"Health & mental trajectories for {len(agent_ids)} sample agents")
    plt.grid(True)
    plt.legend(fontsize=7, ncol=2)

    out_path = os.path.join(OUTPUT_DIR, "plot_sample_agent_trajectories.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[VIZ] Saved {out_path}")


def main():
    ensure_output_dir()
    log_df, summary_df = load_data()

    print("[VIZ] Data loaded. Creating plots...")

    plot_daily_averages(summary_df)
    plot_final_health_by_age_group(log_df)
    plot_heat_conditions_by_age_group(log_df)
    plot_trajectories_for_sample_agents(log_df, n=5)

    print("[VIZ] All plots generated.")


if __name__ == "__main__":
    main()
