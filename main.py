from __future__ import annotations

from config import default_params
from data_loader import load_agents_from_csv
from environment import load_weather_from_csv, generate_dummy_weather
from simulation import Simulation


def main():
    import time
    start_time = time.time()

    agents, families, companies = load_agents_from_csv("data/agents.csv")

    try:
        weather_series = load_weather_from_csv("data/weather.csv")
    except FileNotFoundError:
        print("No data/weather.csv found, using synthetic dummy weather instead.")
        weather_series = generate_dummy_weather(days=14)

    params = default_params()

    sim = Simulation(
        agents=agents,
        weather_series=weather_series,
        params=params,
        families=families,
        companies=companies,
        decision_mode="rule",  # personal risk LLM still used regardless
        use_llm_weather=True,
        ac_available_fraction=0.6,
        group_influence_prob=0.6,
        verbose=True,
    )

    result = sim.run()

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

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\n[INFO] Simulation completed in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    main()
