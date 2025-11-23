from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any

from environment import EnvironmentState


def age_to_group(age: int) -> str:
    """
    Map a numeric age into one of 7 refined age groups.
    """
    if age < 5:
        return "infant_0_4"
    elif age < 13:
        return "child_5_12"
    elif age < 18:
        return "teen_13_17"
    elif age < 40:
        return "young_adult_18_39"
    elif age < 60:
        return "middle_aged_40_59"
    elif age < 75:
        return "older_adult_60_74"
    else:
        return "senior_75_plus"


def disease_to_group(disease_name: str, has_chronic: bool) -> str:
    """
    Map free-text chronic disease into a coarse disease group.
    Groups:
      - none
      - cardio
      - diabetes
      - respiratory
      - renal
      - other
    """
    if not has_chronic:
        return "none"

    name = (disease_name or "").lower()
    if any(k in name for k in ["heart", "cardio", "hypertens", "stroke"]):
        return "cardio"
    if "diab" in name:
        return "diabetes"
    if any(k in name for k in ["asthma", "copd", "lung", "respirat"]):
        return "respiratory"
    if any(k in name for k in ["kidney", "renal"]):
        return "renal"
    return "other"


@dataclass
class Agent:
    id: int
    age: int
    sex: str
    job: str
    baseline_health: float  # 0..1
    baseline_mental: float  # 0..1
    outdoor_hours: float    # typical daily outdoor exposure
    risk_sensitivity: float # 0..1

    # Chronic disease (e.g. diabetes, heart disease)
    has_chronic_disease: bool = False
    disease_name: str = ""

    # Heat-related acute condition (assigned by LLM)
    heat_condition: str = "none"
    heat_condition_severity: float = 0.0

    health: float = field(init=False)
    mental: float = field(init=False)
    hydration: float = field(init=False)

    def __post_init__(self):
        self.health = self.baseline_health
        self.mental = self.baseline_mental
        self.hydration = 1.0  # fully hydrated at start

    # ------------ DECISION ------------

    def decide(
        self,
        env: EnvironmentState,
        params: Dict[str, float],
        mode: str = "rule",
        risk_hint: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Decide daily actions.

        Returns:
          - drink_water: bool
          - go_hospital: bool
          - use_ac: bool
          - target_outdoor_hours: float

        mode:
          - "rule": deterministic behavior (uses risk_hint)
          - "llm": per-agent LLM (via Gemini/OpenAI/Qwen) using build_llm_prompt_for_agent
        """
        if mode == "llm":
            return self.decide_with_llm(env)

        # ---- risk perception ----
        heat_excess = max(0.0, env.temperature - params["safe_temp"])
        hum_excess = max(0.0, env.humidity - params["safe_humidity"])

        perceived_risk = (
            params["w_temp"] * heat_excess
            + params["w_hum"] * hum_excess
            + params["w_age"] * (self.age / 100.0)
            + params["w_health"] * (1.0 - self.health)
        )

        perceived_risk *= (0.5 + 0.5 * self.risk_sensitivity)
        perceived_risk *= risk_hint  # external multiplier from LLM

        # ---- decision rules ----
        drink_water = True   # always recommend water
        use_ac = False
        go_hospital = False
        target_outdoor = self.outdoor_hours

        if perceived_risk < params["risk_low"]:
            use_ac = False
            go_hospital = False
            target_outdoor = self.outdoor_hours
        elif perceived_risk < params["risk_high"]:
            use_ac = env.ac_available
            go_hospital = False
            target_outdoor = self.outdoor_hours * 0.7
        else:
            use_ac = env.ac_available
            go_hospital = (self.health < 0.4) or (self.age > 70)
            target_outdoor = self.outdoor_hours * 0.3

        target_outdoor = max(0.0, min(self.outdoor_hours, target_outdoor))

        return {
            "drink_water": drink_water,
            "go_hospital": go_hospital,
            "use_ac": use_ac,
            "target_outdoor_hours": target_outdoor,
        }

    def decide_with_llm(self, env: EnvironmentState) -> Dict[str, Any]:
        """
        Optional per-agent direct LLM decision (NOT the personal risk prompt).
        This is only used if decision_mode="llm".
        """
        from llm_interfaces import (
            USE_REAL_LLM,
            build_llm_prompt_for_agent,
            call_llm_api,
            parse_agent_decision,
        )

        if USE_REAL_LLM:
            prompt = build_llm_prompt_for_agent(self, env)
            response = call_llm_api(prompt, tag=f"agent_{self.id}_day_{env.day}")
            return parse_agent_decision(response)

        # Fallback heuristic
        heat_index = env.temperature + env.humidity * 10.0
        if heat_index > 40:
            target_outdoor = self.outdoor_hours * 0.3
            drink_water = True
            use_ac = env.ac_available
            go_hospital = (self.health < 0.4) or (self.age > 70)
        else:
            target_outdoor = self.outdoor_hours * 0.8
            drink_water = True
            use_ac = env.ac_available
            go_hospital = False

        return {
            "drink_water": drink_water,
            "go_hospital": go_hospital,
            "use_ac": use_ac,
            "target_outdoor_hours": target_outdoor,
        }

    # ------------ STATE UPDATE ------------

    def update_state(
        self,
        env: EnvironmentState,
        params: Dict[str, float],
        action: Dict[str, Any],
    ) -> None:
        """
        Update physical + mental state using:
          - drink_water
          - go_hospital
          - use_ac
          - target_outdoor_hours
        """
        drink_water = bool(action.get("drink_water", False))
        go_hospital = bool(action.get("go_hospital", False))
        use_ac = bool(action.get("use_ac", False))
        target_outdoor = float(action.get("target_outdoor_hours", self.outdoor_hours))

        # Some jobs require minimum outdoor exposure
        required_min_outdoor = 0.0
        if self.job in ("construction", "delivery", "farmer"):
            required_min_outdoor = min(4.0, self.outdoor_hours)

        outdoor_hours = max(target_outdoor, required_min_outdoor)

        # Hydration dynamics
        if drink_water:
            self.hydration = min(1.0, self.hydration + params["drink_hydration_boost"])

        # baseline dehydration
        self.hydration = max(0.0, self.hydration - params["baseline_dehydration"])

        # Extra dehydration if outside a lot
        if outdoor_hours > self.outdoor_hours * 0.8:
            self.hydration = max(0.0, self.hydration - 0.1)

        # Heat stress
        temp_excess = max(0.0, env.temperature - params["base_safe_temp"])
        hum_factor = 1.0 + env.humidity
        exposure_factor = outdoor_hours / max(1.0, self.outdoor_hours)
        raw_heat_stress = temp_excess * hum_factor * exposure_factor

        # Cooling & hospital mitigation
        cooling_factor = 0.5 if use_ac else 1.0
        hospital_factor = params["hospital_relief_factor"] if go_hospital else 1.0
        hydration_factor = 1.0 - 0.5 * self.hydration  # 0..1

        effective_stress = (
            raw_heat_stress
            * cooling_factor
            * hospital_factor
            * (0.5 + hydration_factor / 2.0)
        )

        # Physical health damage
        age_factor = 0.5 + (self.age / 100.0)
        health_damage = params["health_sensitivity"] * effective_stress * age_factor
        self.health = max(0.0, min(1.0, self.health - health_damage))

        # Small health bump if they go to hospital
        if go_hospital:
            self.health = min(1.0, self.health + 0.01)

        # Mental health damage: heat + isolation
        isolation_factor = 0.0
        if outdoor_hours < self.outdoor_hours * 0.5:
            isolation_factor = (self.outdoor_hours * 0.5 - outdoor_hours) / max(self.outdoor_hours, 0.1)

        mental_damage = (
            params["mental_sensitivity"] * effective_stress
            + params["mental_isolation_weight"] * isolation_factor
        )
        self.mental = max(0.0, min(1.0, self.mental - mental_damage))

        # Hospital mental relief
        if go_hospital:
            self.mental = min(1.0, self.mental + params["hospital_mental_boost"])
