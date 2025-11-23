from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
import random
import math

# ==========================
# Agent definition
# ==========================

@dataclass
class Agent:
    id: int
    age: int
    sex: str
    job: str
    baseline_health: float  # 0..1
    baseline_mental: float  # 0..1
    outdoor_hours: float    # typical daily outdoor exposure
    risk_sensitivity: float # 0..1: how responsive to perceived risk

    # dynamic state (will change during simulation)
    health: float = field(init=False)
    mental: float = field(init=False)
    hydration: float = field(init=False)

    def __post_init__(self):
        self.health = self.baseline_health
        self.mental = self.baseline_mental
        self.hydration = 1.0  # start fully hydrated

    def decide(
        self,
        env: EnvironmentState,
        params: Dict[str, float],
        mode: str = "rule"
    ) -> Dict[str, Any]:
        """
        Decide daily actions. Two modes:
        - 'rule': deterministic rules based on parameters.
        - 'llm' : placeholder that would use an LLM.
        """
        if mode == "llm":
            return self.decide_with_llm(env)

        # ---------- RULE-BASED DECISION ----------
        # Perceived risk is a function of temperature, humidity, age, and health.
        heat_excess = max(0.0, env.temperature - params["safe_temp"])
        hum_excess = max(0.0, env.humidity - params["safe_humidity"])

        perceived_risk = (
            params["w_temp"] * heat_excess +
            params["w_hum"] * hum_excess +
            params["w_age"] * (self.age / 100.0) +
            params["w_health"] * (1.0 - self.health)
        )

        perceived_risk *= (0.5 + 0.5 * self.risk_sensitivity)  # scale by personal trait

        # Decision thresholds
        # Simple: low risk -> keep outside, medium -> reduce a bit, high -> reduce a lot.
        if perceived_risk < params["risk_low"]:
            target_outdoor = self.outdoor_hours
            hydration_boost = 0.1
            use_cooling = False
        elif perceived_risk < params["risk_high"]:
            target_outdoor = self.outdoor_hours * 0.7
            hydration_boost = 0.3
            use_cooling = env.ac_available
        else:
            target_outdoor = self.outdoor_hours * 0.3
            hydration_boost = 0.5
            use_cooling = env.ac_available

        # Keep outdoor hours within [0, baseline]
        target_outdoor = max(0.0, min(self.outdoor_hours, target_outdoor))

        return {
            "target_outdoor_hours": target_outdoor,
            "hydration_boost": hydration_boost,
            "use_cooling": use_cooling,
        }

    def decide_with_llm(self, env: EnvironmentState) -> Dict[str, Any]:
        """
        Placeholder for LLM-based decision. You can plug in your LLM call here.
        For now, falls back to a simple rule to keep the simulation runnable.
        """
        # ----- Example prompt builder (for your future use) -----
        prompt = build_llm_prompt(self, env)

        # response = call_llm_api(prompt)  # You implement this
        # parsed = parse_llm_decision(response)
        # return parsed

        # Fallback: behave like rule-based but with slightly higher caution.
        heat_index = env.temperature + (env.humidity * 10.0)
        if heat_index > 40:
            target_outdoor = self.outdoor_hours * 0.4
            hydration_boost = 0.5
            use_cooling = env.ac_available
        else:
            target_outdoor = self.outdoor_hours * 0.8
            hydration_boost = 0.3
            use_cooling = env.ac_available

        return {
            "target_outdoor_hours": target_outdoor,
            "hydration_boost": hydration_boost,
            "use_cooling": use_cooling,
        }

    def update_state(
        self,
        env: EnvironmentState,
        action: Dict[str, Any],
        params: Dict[str, float]
    ) -> None:
        """
        Update health and mental state based on heat exposure and behavior.
        """
        # Effective outdoor exposure: if job requires being outside, don't let it go to 0.
        required_min_outdoor = 0.0
        if self.job in ("construction", "delivery", "farmer"):
            required_min_outdoor = min(4.0, self.outdoor_hours)

        outdoor_hours = max(action["target_outdoor_hours"], required_min_outdoor)

        # Update hydration
        self.hydration = max(0.0, min(1.0, self.hydration + action["hydration_boost"] - 0.1))
        if outdoor_hours > self.outdoor_hours * 0.8:  # more exposure -> more dehydration
            self.hydration = max(0.0, self.hydration - 0.1)

        # Compute raw heat stress
        temp_excess = max(0.0, env.temperature - params["base_safe_temp"])
        hum_factor = 1.0 + env.humidity   # 1.0..2.0
        exposure_factor = outdoor_hours / max(1.0, self.outdoor_hours)

        raw_heat_stress = temp_excess * hum_factor * exposure_factor

        # Cooling mitigation
        cooling_factor = 0.5 if action["use_cooling"] else 1.0
        # Hydration mitigation (0..1 hydration -> 1..0.5 damage multiplier)
        hydration_factor = 1.0 - 0.5 * self.hydration

        effective_stress = raw_heat_stress * cooling_factor * (0.5 + hydration_factor / 2.0)

        # Health decreases faster for older/less healthy agents
        age_factor = 0.5 + (self.age / 100.0)  # 0.5..1.5
        health_damage = params["health_sensitivity"] * effective_stress * age_factor

        self.health -= health_damage
        self.health = max(0.0, min(1.0, self.health))

        # Mental health: stress + isolation if staying home a lot
        isolation_factor = 0.0
        if outdoor_hours < self.outdoor_hours * 0.5:
            isolation_factor = (self.outdoor_hours * 0.5 - outdoor_hours) / max(self.outdoor_hours, 0.1)

        mental_damage = (
            params["mental_sensitivity"] * effective_stress +
            params["mental_isolation_weight"] * isolation_factor
        )

        self.mental -= mental_damage
        self.mental = max(0.0, min(1.0, self.mental))


# ==========================
# Environment / heatwave
# ==========================

@dataclass
class EnvironmentState:
    day: int
    temperature: float  # °C
    humidity: float     # 0..1
    ac_available: bool


def generate_heatwave_scenario(
    days: int = 14,
    base_temp: float = 28.0,
    heatwave_peak: float = 43.0,
    base_humidity: float = 0.4,
    peak_humidity: float = 0.7,
) -> List[Tuple[float, float]]:
    """
    Generate a simple scenario: gradually ramp up to a heatwave and then back down.
    Returns list of (temperature, humidity) for each day.
    """
    scenario: List[Tuple[float, float]] = []
    for d in range(days):
        # 0..1 ramp
        t = d / (days - 1)
        # Use a bump-shaped curve: low -> high -> low
        heat_factor = math.sin(math.pi * t)
        temp = base_temp + heat_factor * (heatwave_peak - base_temp)
        hum = base_humidity + heat_factor * (peak_humidity - base_humidity)
        scenario.append((temp, hum))
    return scenario


# ==========================
# Simulation
# ==========================

@dataclass
class SimulationResult:
    history: List[Dict[str, float]]
    agents: List[Agent]
    params: Dict[str, float]


class Simulation:
    def __init__(
        self,
        agents: List[Agent],
        scenario: List[Tuple[float, float]],
        params: Dict[str, float],
        ac_available_fraction: float = 0.5,
        decision_mode: str = "rule",
    ):
        self.agents = agents
        self.scenario = scenario
        self.params = params
        self.ac_available_fraction = ac_available_fraction
        self.decision_mode = decision_mode
        self.history: List[Dict[str, float]] = []

    def run(self) -> SimulationResult:
        for day, (temp, hum) in enumerate(self.scenario):
            env = EnvironmentState(
                day=day,
                temperature=temp,
                humidity=hum,
                ac_available=random.random() < self.ac_available_fraction,
            )

            # Each agent decides and updates
            for agent in self.agents:
                action = agent.decide(env, self.params, mode=self.decision_mode)
                agent.update_state(env, action, self.params)

            self._record_day(day)

        return SimulationResult(history=self.history, agents=self.agents, params=self.params)

    def _record_day(self, day: int) -> None:
        avg_health = sum(a.health for a in self.agents) / len(self.agents)
        avg_mental = sum(a.mental for a in self.agents) / len(self.agents)
        self.history.append(
            {
                "day": day,
                "avg_health": avg_health,
                "avg_mental": avg_mental,
            }
        )


# ==========================
# Optimization
# ==========================

def create_random_agent(id_: int) -> Agent:
    age = random.randint(18, 85)
    sex = random.choice(["M", "F"])
    job = random.choice(["office", "construction", "delivery", "retired", "teacher"])
    baseline_health = random.uniform(0.4, 1.0)
    baseline_mental = random.uniform(0.4, 1.0)
    if job in ("construction", "delivery", "farmer"):
        outdoor_hours = random.uniform(4.0, 8.0)
    elif job == "retired":
        outdoor_hours = random.uniform(0.0, 3.0)
    else:
        outdoor_hours = random.uniform(1.0, 5.0)

    risk_sensitivity = random.uniform(0.0, 1.0)

    return Agent(
        id=id_,
        age=age,
        sex=sex,
        job=job,
        baseline_health=baseline_health,
        baseline_mental=baseline_mental,
        outdoor_hours=outdoor_hours,
        risk_sensitivity=risk_sensitivity,
    )


def create_population(n: int) -> List[Agent]:
    return [create_random_agent(i) for i in range(n)]


def default_params() -> Dict[str, float]:
    """
    Parameter dict controlling influence. These are what we "learn" via optimization.
    """
    return {
        # perception
        "safe_temp": 32.0,
        "safe_humidity": 0.5,
        "w_temp": 0.08,
        "w_hum": 2.0,
        "w_age": 0.8,
        "w_health": 1.0,
        "risk_low": 1.5,
        "risk_high": 3.0,
        # health dynamics
        "base_safe_temp": 30.0,
        "health_sensitivity": 0.01,
        "mental_sensitivity": 0.005,
        "mental_isolation_weight": 0.01,
    }


def sample_params() -> Dict[str, float]:
    """
    Randomly sample parameters within reasonable ranges for optimization.
    """
    return {
        "safe_temp": random.uniform(30.0, 36.0),
        "safe_humidity": random.uniform(0.4, 0.7),
        "w_temp": random.uniform(0.03, 0.15),
        "w_hum": random.uniform(1.0, 3.0),
        "w_age": random.uniform(0.3, 1.5),
        "w_health": random.uniform(0.3, 1.5),
        "risk_low": random.uniform(1.0, 2.5),
        "risk_high": random.uniform(2.0, 4.5),
        "base_safe_temp": random.uniform(28.0, 32.0),
        "health_sensitivity": random.uniform(0.005, 0.02),
        "mental_sensitivity": random.uniform(0.002, 0.02),
        "mental_isolation_weight": random.uniform(0.003, 0.03),
    }


def evaluate_params(params: Dict[str, float], population_size: int = 100, days: int = 14) -> float:
    """
    Objective function for optimization.
    Lower is better.
    Metric: average loss of health & mental over the period.
    """
    scenario = generate_heatwave_scenario(days=days)
    agents = create_population(population_size)
    sim = Simulation(agents, scenario, params, ac_available_fraction=0.5, decision_mode="rule")
    result = sim.run()

    # Compare final vs initial
    health_losses = []
    mental_losses = []
    for a in result.agents:
        health_losses.append(a.baseline_health - a.health)
        mental_losses.append(a.baseline_mental - a.mental)

    avg_health_loss = sum(health_losses) / len(health_losses)
    avg_mental_loss = sum(mental_losses) / len(mental_losses)

    # Combine
    score = avg_health_loss + 0.8 * avg_mental_loss
    return score


def random_search_optimization(
    iterations: int = 30,
    population_size: int = 100,
    days: int = 14,
) -> Tuple[Dict[str, float], float]:
    """
    Super simple optimization: random search over parameter space.
    """
    best_params: Dict[str, float] | None = None
    best_score = float("inf")

    for i in range(iterations):
        params = sample_params()
        score = evaluate_params(params, population_size=population_size, days=days)
        print(f"[Iteration {i+1}/{iterations}] score={score:.4f}")

        if score < best_score:
            best_score = score
            best_params = params

    assert best_params is not None
    return best_params, best_score


# ==========================
# LLM interface (stubs)
# ==========================

def build_llm_prompt(agent: Agent, env: EnvironmentState) -> str:
    """
    Example prompt builder that you can send to an LLM.
    """
    prompt = f"""
You are an expert advising a person during a dangerous heatwave.

Person:
- Age: {agent.age}
- Sex: {agent.sex}
- Job: {agent.job}
- Baseline physical health (0-1): {agent.baseline_health:.2f}
- Current physical health (0-1): {agent.health:.2f}
- Current mental health (0-1): {agent.mental:.2f}
- Typical outdoor hours per day: {agent.outdoor_hours:.1f}

Today:
- Temperature: {env.temperature:.1f} °C
- Humidity: {env.humidity:.0%}
- Air conditioning available: {env.ac_available}

Suggest safe behavior for **today**. Respond in valid JSON with keys:
- "target_outdoor_hours": float (0-24)
- "hydration_boost": float (between 0 and 1, higher = drinks more)
- "use_cooling": bool

JSON only, no explanation.
"""
    return prompt.strip()


def call_llm_api(prompt: str) -> str:
    """
    Placeholder where you plug in your actual LLM call.
    For example with openai:
        client = OpenAI()
        response = client.responses.create(...)
    """
    raise NotImplementedError("Implement LLM call here.")


def parse_llm_decision(response: str) -> Dict[str, Any]:
    """
    Tiny helper to parse the JSON response from the model.
    """
    import json
    data = json.loads(response)
    return {
        "target_outdoor_hours": float(data["target_outdoor_hours"]),
        "hydration_boost": float(data["hydration_boost"]),
        "use_cooling": bool(data["use_cooling"]),
    }


# ==========================
# Demo / main
# ==========================

def run_demo():
    print("=== Heatwave simulation demo (no optimization) ===")
    agents = create_population(50)
    scenario = generate_heatwave_scenario(days=14)
    params = default_params()

    sim = Simulation(agents, scenario, params, ac_available_fraction=0.5, decision_mode="rule")
    result = sim.run()

    print("Day\tAvg Health\tAvg Mental")
    for day_info in result.history:
        print(f"{day_info['day']:2d}\t{day_info['avg_health']:.3f}\t\t{day_info['avg_mental']:.3f}")

    # Show a few agents
    print("\nSample agents at end of simulation:")
    for a in result.agents[:5]:
        print(
            f"Agent {a.id} ({a.age}y, {a.job}): "
            f"health {a.health:.2f} (start {a.baseline_health:.2f}), "
            f"mental {a.mental:.2f} (start {a.baseline_mental:.2f})"
        )


def run_optimization_demo():
    print("\n=== Optimizing influence parameters with random search ===")
    best_params, best_score = random_search_optimization(
        iterations=20,
        population_size=80,
        days=14,
    )
    print("\nBest score:", best_score)
    print("Best parameters:")
    for k, v in best_params.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    # Simple CLI toggle: change as you like.
    run_demo()
    run_optimization_demo()

