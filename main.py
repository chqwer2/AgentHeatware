from __future__ import annotations

import argparse

from config import default_params
from data_loader import load_agents_from_csv
from environment import load_weather_from_csv, generate_dummy_weather
from simulation import Simulation
from output_utils import save_agent_daily_log, save_daily_summary
from llm_interfaces import set_global_llm_enabled


def parse_args():
    parser = argparse.ArgumentParser(
        description="Heatwave agent-based simulation with optional LLM influence."
    )

    parser.add_argument(
        "--agents-path",
        type=str,
        default="data/agents.csv",
        help="Path to agents CSV file (default: data/agents.csv)",
    )

    parser.add_argument(
        "--weather-path",
        type=str,
        default="data/weather.csv",
        help=(
            "Path to weather CSV file (default: data/weather.csv). "
            "If missing, synthetic dummy weather will be used."
        ),
    )

    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable ALL LLM features (weather, personal risk, disease, groups).",
    )

    parser.add_argument(
        "--silent",
        action="store_true",
        help="Reduce console output (sets verbose=False in Simulation).",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Single global switch: use_llm=True/False
    use_llm = not args.no_llm
    set_global_llm_enabled(use_llm)

    if use_llm:
        print("[MAIN] LLM usage ENABLED for all features.")
    else:
        print("[MAIN] LLM usage DISABLED. Using purely rule-based behavior.")

    # 1. Load agents + families + companies
    print(f"[MAIN] Loading agents from {args.agents_path}")
    agents, families, companies = load_agents_from_csv(args.agents_path)

    # 2. Load weather series
    try:
        print(f"[MAIN] Loading weather from {args.weather_path}")
        weather_series = load_weather_from_csv(args.weather_path)
    except FileNotFoundError:
        print(f"[MAIN] {args.weather_path} not found, using synthetic dummy weather instead.")
        weather_series = generate_dummy_weather(days=14)

    # 3. Parameters
    params = default_params()

    verbose = not args.silent

    # 4. Build simulation
    sim = Simulation(
        agents=agents,
        weather_series=weather_series,
        params=params,
        families=families,
        companies=companies,
        decision_mode="rule",   # if you ever set "llm", it will STILL be ignored when use_llm=False
        use_llm=use_llm,        # <--- SINGLE flag controlling ALL LLM usage inside Simulation
        ac_available_fraction=0.6,
        group_influence_prob=0.6,
        verbose=verbose,
    )

    # 5. Run
    result = sim.run()

    if verbose:
        print("\n=== Daily Averages ===")
        print("Day\tAvgHealth\tAvgMental")
        for day_info in result.history:
            print(f"{day_info['day']:3d}\t{day_info['avg_health']:.3f}\t\t{day_info['avg_mental']:.3f}")

        print("\n=== Sample agents at end ===")
        for a in result.agents[:5]:
            print(
                f"Agent {a.id} ({a.age}y, {a.job}, {a.heat_condition}) "
                f"H:{a.health:.2f}/{a.baseline_health:.2f} "
                f"M:{a.mental:.2f}/{a.baseline_mental:.2f}"
            )

        print("\n=== Log preview (first 5 entries) ===")
        for row in result.agent_daily_log[:5]:
            print(row)

    # 6. Save detailed outputs
    save_agent_daily_log(result.agent_daily_log, "output/agent_daily_log.csv")
    save_daily_summary(result.history, "output/daily_summary.csv")


if __name__ == "__main__":
    main()
