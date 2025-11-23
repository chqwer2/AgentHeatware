from __future__ import annotations
from typing import Dict, Tuple, Optional
import random
import copy

from config import default_params
from data_loader import load_agents_from_csv
from environment import load_weather_from_csv, generate_dummy_weather
from simulation import Simulation, SimulationResult
from llm_interfaces import USE_REAL_LLM


def compute_death_rate(
    result: SimulationResult,
    death_health_threshold: float = 0.0,
    death_mental_threshold: float = 0.0,
) -> float:
    """
    Death is defined as:
      health <= death_health_threshold OR
      mental <= death_mental_threshold
    """
    dead = 0
    total = len(result.agents)
    for a in result.agents:
        if (a.health <= death_health_threshold) or (a.mental <= death_mental_threshold):
            dead += 1
    if total == 0:
        return 0.0
    return dead / total


def sample_params(base: Dict[str, float]) -> Dict[str, float]:
    """
    Randomly perturb some config weights around their base values.
    """
    p = base.copy()

    # Perception weights
    p["w_temp"] = random.uniform(0.03, 0.15)
    p["w_hum"] = random.uniform(1.0, 3.0)
    p["w_age"] = random.uniform(0.3, 1.5)
    p["w_health"] = random.uniform(0.3, 1.5)

    # Health / mental dynamics
    p["health_sensitivity"] = random.uniform(0.005, 0.02)
    p["mental_sensitivity"] = random.uniform(0.002, 0.02)
    p["mental_isolation_weight"] = random.uniform(0.003, 0.03)

    # Decision effect parameters
    p["drink_hydration_boost"] = random.uniform(0.2, 0.6)
    p["baseline_dehydration"] = random.uniform(0.05, 0.15)
    p["hospital_relief_factor"] = random.uniform(0.1, 0.6)
    p["hospital_mental_boost"] = random.uniform(0.01, 0.1)

    return p


def evaluate_params(
    params: Dict[str, float],
    agents_template,
    weather_series,
    target_death_rate: Optional[float] = None,
    random_seed: int = 123,
) -> Tuple[float, float]:
    """
    Run ONE simulation with given params and return:
      (objective_value, simulated_death_rate)
    """
    agents = copy.deepcopy(agents_template)
    random.seed(random_seed)

    # For optimization we disable LLM and groups to keep it fast/deterministic
    sim = Simulation(
        agents=agents,
        weather_series=weather_series,
        params=params,
        families=[],
        companies=[],
        decision_mode="rule",
        use_llm_weather=False,
        ac_available_fraction=0.6,
        group_influence_prob=0.0,
        verbose=False,
    )

    result = sim.run()
    death_rate = compute_death_rate(result)

    if target_death_rate is None:
        objective = death_rate
    else:
        objective = (death_rate - target_death_rate) ** 2

    return objective, death_rate


def random_search_optimization(
    agents_template,
    weather_series,
    iterations: int = 30,
    target_death_rate: Optional[float] = None,
) -> Tuple[Dict[str, float], float, float]:
    """
    Simple random search over parameter space.
    Returns: best_params, best_objective, best_death_rate
    """
    base = default_params()
    best_params: Dict[str, float] = base
    best_objective = float("inf")
    best_death_rate = 0.0

    print(f"[OPT] Starting random search for {iterations} iterations...")
    for i in range(iterations):
        params = sample_params(base)
        obj, dr = evaluate_params(
            params=params,
            agents_template=agents_template,
            weather_series=weather_series,
            target_death_rate=target_death_rate,
        )
        print(f"[OPT] Iter {i+1:02d}/{iterations} objective={obj:.5f}, death_rate={dr:.4f}")

        if obj < best_objective:
            best_objective = obj
            best_death_rate = dr
            best_params = params
            print(f"[OPT]  -> NEW BEST! obj={best_objective:.5f}, death_rate={best_death_rate:.4f}")

    print("[OPT] Random search finished.")
    return best_params, best_objective, best_death_rate


def main():
    if USE_REAL_LLM:
        print("[WARN] USE_REAL_LLM is True, but optimize.py is designed to run without LLM.")
        print("       For cheaper optimization, set PROVIDER='none' in llm_interfaces.py.")

    agents, families, companies = load_agents_from_csv("data/agents.csv")

    try:
        weather_series = load_weather_from_csv("data/weather.csv")
    except FileNotFoundError:
        print("No data/weather.csv found, using dummy weather.")
        weather_series = generate_dummy_weather(days=14)

    TARGET_DEATH_RATE = None  # or set e.g. 0.05 if you want to match observed rate

    best_params, best_obj, best_dr = random_search_optimization(
        agents_template=agents,
        weather_series=weather_series,
        iterations=20,
        target_death_rate=TARGET_DEATH_RATE,
    )

    print("\n=== Optimization finished ===")
    if TARGET_DEATH_RATE is None:
        print(f"Best simulated death rate: {best_dr:.4f}")
        print(f"(Objective was: minimize death rate directly)")
    else:
        print(f"Target death rate: {TARGET_DEATH_RATE:.4f}")
        print(f"Best simulated death rate: {best_dr:.4f}")
        print(f"Best objective (squared error): {best_obj:.6f}")

    print("\nBest parameters:")
    for k, v in best_params.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
