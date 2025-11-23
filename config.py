from typing import Dict


def default_params() -> Dict[str, float]:
    """
    Global parameters controlling:
    - risk perception
    - health and mental dynamics
    - effect of decisions (water / hospital / AC)
    """
    return {
        # Perception / decision thresholds
        "safe_temp": 32.0,
        "safe_humidity": 0.5,
        "w_temp": 0.08,
        "w_hum": 2.0,
        "w_age": 0.8,
        "w_health": 1.0,
        "risk_low": 1.5,
        "risk_high": 3.0,

        # Health dynamics
        "base_safe_temp": 30.0,
        "health_sensitivity": 0.01,
        "mental_sensitivity": 0.005,
        "mental_isolation_weight": 0.01,

        # Decision effect parameters
        "drink_hydration_boost": 0.4,   # hydration increase when drinking water
        "baseline_dehydration": 0.1,    # daily dehydration
        "hospital_relief_factor": 0.3,  # multiply heat stress when in hospital (<1 = relief)
        "hospital_mental_boost": 0.05,  # mental recovery when going to hospital
    }
