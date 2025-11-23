from __future__ import annotations

import random
from typing import Dict

from agents import Agent
from environment import Environment


def compute_heat_index(temperature_c: float, humidity: float) -> float:
    """
    Return a crude heat index (higher = worse) from temperature and humidity.

    This is intentionally simple; if you want physics-accurate WBGT or
    OSHA-style heat index, you can swap this out.

    We approximate in Celsius by:

        heat_index = T + 10 * humidity

    kept in one place for easy tuning.
    """
    return float(temperature_c) + 10.0 * float(humidity)


def age_vulnerability_multiplier(age_group: str) -> float:
    """
    Map the refined age_group label to a vulnerability multiplier.

    These values are *modeling assumptions*; tweak freely.
    """
    table: Dict[str, float] = {
        "infant": 1.9,
        "toddler": 1.6,
        "child": 1.3,
        "teen": 1.1,
        "young_adult": 1.0,
        "adult": 1.0,
        "middle_aged": 1.15,
        "elderly": 1.7,
        "very_elderly": 2.1,
    }
    return table.get(age_group, 1.0)


def disease_vulnerability_multiplier(agent: Agent) -> float:
    """
    Additional vulnerability if the agent already has chronic conditions.
    You can refine this to depend on the specific diseases present.
    """
    if not agent.diseases:
        return 1.0

    # Differentiated example: some diseases are worse under heat
    multiplier = 1.0
    for disease in agent.diseases:
        if disease in {"cardiovascular", "respiratory"}:
            multiplier *= 1.3
        elif disease in {"diabetes"}:
            multiplier *= 1.15
        elif disease in {"heatwave_disease", "heat_exhaustion"}:
            multiplier *= 1.25
        else:
            multiplier *= 1.05  # default mild extra risk

    return multiplier


def compute_daily_heat_risk_score(agent: Agent, env: Environment) -> float:
    """
    Compress all relevant factors into a single "risk score" for this day.

    Rough sketch:
      base_risk from weather (heat_index buckets)
      * age_vulnerability_multiplier
      * disease_vulnerability_multiplier

    Returns a scalar in [0, ~0.5+] that is then used to adjust health/sanity.
    """
    hi = compute_heat_index(env.weather.temperature_c, env.weather.humidity)

    if hi < 35:
        base = 0.0
    elif hi < 40:
        base = 0.04
    elif hi < 45:
        base = 0.09
    elif hi < 50:
        base = 0.16
    else:
        base = 0.24

    age_mult = age_vulnerability_multiplier(agent.age_group)
    disease_mult = disease_vulnerability_multiplier(agent)

    return base * age_mult * disease_mult


def update_agent_from_heat(agent: Agent, env: Environment) -> None:
    """
    Apply the effect of today's heat/humidity on the agent.

    Rules of thumb:
      - Physical health falls roughly linearly with risk score.
      - Sanity (mental state) also deteriorates, but slightly less.
      - When both are low, there is a rising probability of acquiring
        a "heatwave_disease" condition representing acute heat illness.
    """
    if not agent.is_alive:
        return

    risk = compute_daily_heat_risk_score(agent, env)

    # Physical and mental deltas, risk scaled down moderately
    physical_delta = -0.3 * risk
    mental_delta = -0.2 * risk

    agent.apply_health_delta(physical_delta, mental_delta)

    # Chance of acute heat disease if both are low
    if agent.health < 0.45 and agent.sanity < 0.45:
        # risk increases sharply as both approach 0
        distress = (0.45 - agent.health) + (0.45 - agent.sanity)
        # Normalise distress into [0, 1] and then map into probability
        p = min(0.85, 0.15 + 0.7 * distress)
        if random.random() < p:
            agent.add_disease("heatwave_disease")


def apply_nighttime_recovery(agent: Agent) -> None:
    """
    Simple overnight recovery rule: people partially recover each night,
    but diseases cap the maximum they can bounce back to.
    """
    if not agent.is_alive:
        return

    # Baseline recovery
    target_health = 0.9
    target_sanity = 0.95

    # If they have acute heat disease, cap the nightly recovery
    if "heatwave_disease" in agent.diseases:
        target_health = min(target_health, 0.7)
        target_sanity = min(target_sanity, 0.85)

    # Move 20% of the way toward the target each night
    agent.apply_health_delta(
        0.2 * (target_health - agent.health),
        0.2 * (target_sanity - agent.sanity),
    )
