from typing import Dict


def default_params() -> Dict[str, float]:
    """
    Global parameters controlling:
    - risk perception
    - health and mental dynamics
    - effect of decisions (water / hospital / AC / mental help)
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

        # --- Comfort / heat thresholds ---
        # Below comfort band: no heat damage, net recovery
        "comfort_temp_max": 30.0,
        "comfort_humidity_max": 0.7,

        # Temperature at which heat damage starts to be noticeable
        "heat_temp": 32.0,
        # Temperature at which damage ramps up strongly
        "extreme_temp": 38.0,

        # Health dynamics
        "health_sensitivity": 0.006,          # slightly reduced
        "mental_heat_sensitivity": 0.004,     # smaller than physical
        "mental_isolation_weight": 0.01,

        # Caps on daily damage so one bad day isn't instantly lethal
        "max_daily_health_damage": 0.06,
        "max_daily_mental_damage": 0.05,

        # Recovery dynamics toward baseline (per day)
        "health_recovery_rate": 0.02,
        "mental_recovery_rate": 0.03,

        # Extra recovery on comfortable days
        "comfort_health_recovery_bonus": 0.03,
        "comfort_mental_recovery_bonus": 0.04,

        # Humidity amplification for heat stress (when humidity > 0.5)
        "humidity_heat_amplifier": 0.5,

        # Decision effect parameters
        "drink_hydration_boost": 0.5,   # water helps more
        "baseline_dehydration": 0.08,   # slightly less baseline dehydration

        # Stronger mitigation for AC & hospital
        "ac_heat_reduction_factor": 0.3,    # 0.3 = strong cooling
        "hospital_relief_factor": 0.3,      # multiply heat stress when in hospital (<1 = relief)
        "hospital_health_boost": 0.06,      # health bump if they go to hospital
        "hospital_mental_boost": 0.04,      # mental bump if they go to hospital

        # Mental help (e.g. professional / gov support) when sanity is low
        "mental_help_threshold": 0.3,
        "mental_help_boost": 0.08,          # strong one-day boost when they seek help
    }
