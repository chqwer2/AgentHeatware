from __future__ import annotations
from typing import List, Dict, Any
import json
import os
import re

from agents import Agent, age_to_group, disease_to_group
from environment import EnvironmentState

# =========================================================
# LLM PROVIDER CONFIG
# =========================================================

PROVIDER = "gemini"  # "gemini", "openai", "qwen", or "none"

LLM_VERBOSE = True   # print prompts and responses

# Does this provider actually correspond to a real LLM backend?
USE_REAL_LLM = PROVIDER in ("gemini", "openai", "qwen")

# Global runtime switch controlled by CLI (--no-llm)
# If False, ALL LLM features should be disabled, regardless of PROVIDER.
GLOBAL_LLM_ENABLED = True


def set_global_llm_enabled(enabled: bool) -> None:
    """
    Called from main.py (or tests) to globally enable/disable LLM usage.

    Example in main.py:
        from llm_interfaces import set_global_llm_enabled
        set_global_llm_enabled(not args.no_llm)
    """
    global GLOBAL_LLM_ENABLED
    GLOBAL_LLM_ENABLED = bool(enabled)
    print(f"[LLM] Global LLM enabled set to {GLOBAL_LLM_ENABLED}")


# ---- Gemini config ----
# pip install google-genai
# export GEMINI_API_KEY="your_key"
GEMINI_MODEL = "gemini-3-pro-preview"  # as you requested


# ---- OpenAI GPT config ----
# pip install openai
# export OPENAI_API_KEY="your_key"
OPENAI_MODEL = "gpt-4.1-mini"


# ---- Qwen (DashScope) config ----
# pip install openai
# export DASHSCOPE_API_KEY="your_key"
QWEN_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
QWEN_MODEL = "qwen-max"


# =========================================================
# CORE LLM CALL
# =========================================================

def call_llm_api(prompt: str, tag: str = "") -> str:
    """
    Core LLM call. Uses the provider selected by PROVIDER.
    Respects GLOBAL_LLM_ENABLED so that a single flag controls ALL LLM use.
    """
    if not USE_REAL_LLM or not GLOBAL_LLM_ENABLED:
        raise RuntimeError(
            "LLM call requested but LLM is disabled. "
            "Either PROVIDER='none' or GLOBAL_LLM_ENABLED=False. "
            "If you are using CLI, check the --no-llm flag."
        )

    if PROVIDER == "gemini":
        return _call_gemini(prompt, tag)
    elif PROVIDER == "openai":
        return _call_openai(prompt, tag)
    elif PROVIDER == "qwen":
        return _call_qwen(prompt, tag)
    else:
        raise RuntimeError(f"Unknown PROVIDER '{PROVIDER}'.")


def _print_request(provider: str, model: str, tag: str, prompt: str) -> None:
    if not LLM_VERBOSE:
        return
    print("\n================ LLM REQUEST ==================")
    print(f"[LLM-{provider}] tag={tag}, model={model}")
    print("---------- PROMPT BEGIN ----------")
    print(prompt)
    print("----------- PROMPT END -----------")


def _print_response(provider: str, model: str, tag: str, text: str) -> None:
    if not LLM_VERBOSE:
        return
    print("=============== LLM RESPONSE =================")
    print(f"[LLM-{provider}] tag={tag}, model={model}")
    print("--------- RAW RESPONSE BEGIN ---------")
    print(text)
    print("---------- RAW RESPONSE END ----------\n")


def _call_openai(prompt: str, tag: str) -> str:
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment.")
    client = OpenAI(api_key=api_key)
    model = OPENAI_MODEL
    provider = "OPENAI"

    _print_request(provider, model, tag, prompt)

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    text = resp.choices[0].message.content

    _print_response(provider, model, tag, text)
    return text


def _call_qwen(prompt: str, tag: str) -> str:
    from openai import OpenAI

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY not set in environment.")
    client = OpenAI(
        api_key=api_key,
        base_url=QWEN_BASE_URL,
    )
    model = QWEN_MODEL
    provider = "QWEN"

    _print_request(provider, model, tag, prompt)

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    text = resp.choices[0].message.content

    _print_response(provider, model, tag, text)
    return text


def _call_gemini(prompt: str, tag: str) -> str:
    from google import genai

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set in environment.")

    client = genai.Client(api_key=api_key)
    model = GEMINI_MODEL
    provider = "GEMINI"

    _print_request(provider, model, tag, prompt)

    resp = client.models.generate_content(
        model=model,
        contents=prompt,
    )
    text = getattr(resp, "text", "") or ""

    _print_response(provider, model, tag, text)
    return text


# =========================================================
# SIMPLE PER-AGENT DECISION PROMPT (optional, only if decision_mode="llm")
# =========================================================

def build_llm_prompt_for_agent(agent: Agent, env: EnvironmentState) -> str:
    return f"""
You are advising ONE person during a dangerous heatwave.

Person:
- id: {agent.id}
- age: {agent.age}
- age_group: {age_to_group(agent.age)}
- sex: {agent.sex}
- job: {agent.job}
- health: {agent.health:.2f} (0..1)
- mental: {agent.mental:.2f} (0..1)
- typical_outdoor_hours: {agent.outdoor_hours:.1f}

Weather today:
- temperature: {env.temperature:.1f} °C
- humidity: {env.humidity:.0%}
- AC available: {env.ac_available}

You must choose today's actions:
1) drink_water: true/false
2) go_hospital: true/false
3) use_ac: true/false
4) target_outdoor_hours: 0-24

Return JSON ONLY:
{{
  "drink_water": <true or false>,
  "go_hospital": <true or false>,
  "use_ac": <true or false>,
  "target_outdoor_hours": <float 0-24>
}}
""".strip()


def clean_llm_json(raw: str) -> str:
    """
    Clean common formatting noise from LLM responses:
    - strips whitespace
    - removes ```json ... ``` fences if present
    """
    if raw is None:
        raise ValueError("Empty response from LLM")

    text = raw.strip()
    if not text:
        raise ValueError("LLM returned an empty string")

    # If the model wrapped it in ```json ... ``` or ``` ... ```
    if text.startswith("```"):
        # remove the first line (```json or ```):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        # remove a trailing ``` if present
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

    return text


def parse_agent_decision(response: str) -> Dict[str, Any]:
    response = clean_llm_json(response)
    data = json.loads(response)
    return {
        "drink_water": bool(data["drink_water"]),
        "go_hospital": bool(data["go_hospital"]),
        "use_ac": bool(data["use_ac"]),
        "target_outdoor_hours": float(data["target_outdoor_hours"]),
    }


# =========================================================
# PERSONAL RISK PROMPT (age_group & disease_group influence)
# =========================================================

def build_personal_risk_prompt(agent: Agent, env: EnvironmentState) -> str:
    """
    This is ALWAYS used (when GLOBAL_LLM_ENABLED=True and USE_REAL_LLM=True)
    for each agent each day.
    """
    age_group_str = age_to_group(agent.age)
    disease_group_str = disease_to_group(agent.disease_name, agent.has_chronic_disease)

    return f"""
You are a medical risk assistant for heatwaves.

You must reason carefully about HOW heat and humidity influence different
age groups and chronic disease groups, then adjust risk for this specific person.

AGE GROUPS (increasing baseline vulnerability):
- infant_0_4: very fragile, cannot regulate heat well, depends on adults.
- child_5_12: more resilient than infants but still sensitive to dehydration.
- teen_13_17: generally resilient, but can overexert outdoors.
- young_adult_18_39: baseline resilience; risk mostly from extreme exposure.
- middle_aged_40_59: some decline; chronic disease becomes more common.
- older_adult_60_74: clearly more vulnerable to heat stress.
- senior_75_plus: highest vulnerability; very prone to heatstroke and death.

DISEASE GROUPS (how they change sensitivity):
- none: no chronic disease.
- cardio: heart disease, hypertension, stroke history.
  Heat and dehydration strongly increase heart strain and risk of death.
- diabetes: impaired blood vessels and kidneys, worse dehydration tolerance.
- respiratory: asthma, COPD, chronic lung diseases; hot and humid air worsens breathing.
- renal: chronic kidney disease; dehydration is especially dangerous.
- other: some increased risk but less specific than cardio/renal/respiratory.

Your job has TWO levels:

1) GROUP-LEVEL reasoning:
   For the person's age_group and disease_group, think how:
   - high temperature
   - high humidity
   - being young vs old
   - each disease group
   change the risk. More fragile groups MUST have higher severities.

   For each factor, estimate severity from 0 to 1:
   - 0 = no extra risk beyond baseline
   - 1 = very severe extra risk

2) INDIVIDUAL-LEVEL adjustment:
   Adjust the overall risk using this specific person's:
   - current_health (low health increases risk)
   - current_mental (low mental may reduce self-care)
   - typical_outdoor_hours (more exposure increases risk)
   - AC availability (reduces risk when used)

Then you must produce:

- overall_risk_level: "low", "medium", "high", or "critical"
- overall_risk_multiplier:
   - low: around 1.0
   - medium: 1.1 to 1.3
   - high: 1.3 to 1.6
   - critical: 1.6 to 2.0
  Older age groups and more dangerous disease groups (cardio, renal,
  respiratory, diabetes) SHOULD push the multiplier toward the higher end.

- sickness_base_probability:
   - float between 0 and 1
   - probability that THIS person develops a new acute heat-related condition TODAY
   - This should be much HIGHER for:
     - infant_0_4, older_adult_60_74, senior_75_plus
     - disease_group in ["cardio", "renal", "respiratory", "diabetes"]
     - low health, low mental, high temp, high humidity.

- Recommended actions for TODAY:
   - drink_water: true/false
   - go_hospital: true/false
   - use_ac: true/false
   - target_outdoor_hours: 0-24

Person:
- id: {agent.id}
- age: {agent.age}
- age_group: {age_group_str}
- sex: {agent.sex}
- job: {agent.job}
- disease_group: {disease_group_str}
- chronic_disease_name: {agent.disease_name}
- current_health: {agent.health:.2f} (0=very bad, 1=perfect)
- current_mental: {agent.mental:.2f} (0=very bad, 1=perfect)
- typical_outdoor_hours: {agent.outdoor_hours:.1f}

Today's weather:
- temperature_celsius: {env.temperature:.1f}
- humidity_fraction: {env.humidity:.2f}
- AC_available: {env.ac_available}

Return JSON ONLY in this exact structure:

{{
  "age_group": "{age_group_str}",
  "disease_group": "{disease_group_str}",
  "risk_explanation": {{
    "temperature": {{
      "severity": <float 0-1>,
      "how": "<short explanation>"
    }},
    "humidity": {{
      "severity": <float 0-1>,
      "how": "<short explanation>"
    }},
    "age_group": {{
      "severity": <float 0-1>,
      "how": "<short explanation>"
    }},
    "disease_group": {{
      "severity": <float 0-1>,
      "how": "<short explanation>"
    }}
  }},
  "overall_risk_level": "low" | "medium" | "high" | "critical",
  "overall_risk_multiplier": <float>,
  "sickness_base_probability": <float 0-1>,
  "actions": {{
    "drink_water": <true or false>,
    "go_hospital": <true or false>,
    "use_ac": <true or false>,
    "target_outdoor_hours": <float>
  }}
}}
""".strip()


def parse_personal_risk_response(text: str) -> Dict[str, Any]:
    text = clean_llm_json(text)
    data = json.loads(text)

    actions = data.get("actions", {})
    expl = data.get("risk_explanation", {})

    return {
        "age_group": data.get("age_group", ""),
        "disease_group": data.get("disease_group", ""),
        "risk_level": data.get("overall_risk_level", "medium"),
        "overall_risk_multiplier": float(data.get("overall_risk_multiplier", 1.0)),
        "sickness_base_probability": float(data.get("sickness_base_probability", 0.0)),
        "actions": {
            "drink_water": bool(actions.get("drink_water", True)),
            "go_hospital": bool(actions.get("go_hospital", False)),
            "use_ac": bool(actions.get("use_ac", False)),
            "target_outdoor_hours": float(actions.get("target_outdoor_hours", 0.0)),
        },
        "factors": {
            "temperature": expl.get("temperature", {}),
            "humidity": expl.get("humidity", {}),
            "age_group": expl.get("age_group", {}),
            "disease_group": expl.get("disease_group", {}),
        },
    }


# =========================================================
# HEAT-RELATED DISEASE PROMPT (probabilistic, varied)
# =========================================================

def build_heat_disease_prompt(
    agent: Agent,
    env: EnvironmentState,
    final_outdoor_hours: float,
    base_prob: float,
) -> str:
    """
    Used in simulation.py when health or mental < threshold.
    """
    age_group_str = age_to_group(agent.age)
    disease_group_str = disease_to_group(agent.disease_name, agent.has_chronic_disease)

    return f"""
You are a clinician specializing in heat-related illness.

The patient has LOW health and/or LOW mental resilience after a heatwave day.
You must reason about HOW age and chronic disease change:
  - which acute condition is most likely
  - how severe it is
  - how likely they are to get sick at all.

AGE/Disease vulnerability hints:
- infant_0_4 and senior_75_plus: most fragile; more likely to develop heatstroke
  or severe dehydration with high severity.
- older_adult_60_74: clearly more vulnerable than middle_aged and young adults.
- cardio: more likely to develop dangerous "heat_exhaustion" or "heatstroke".
- renal: more likely to get "dehydration" or "kidney_strain".
- respiratory: heat and humidity worsen breathing; may tilt toward "heat_exhaustion".
- diabetes: worse dehydration tolerance, higher risk of "heat_exhaustion" or "dehydration".
- none/other: lower but still non-zero risk.

Your tasks:

1) Decide a PROBABILITY that they actually develop a new acute heat-related condition today.
   - sickness_probability: float between 0 and 1
   - Start from base_sickness_probability_hint but adjust:
     - INCREASE it for infants, older_adults, seniors.
     - INCREASE it for cardio/renal/respiratory/diabetes.
     - DECREASE it if temp/humidity are mild or outdoor_exposure is low.

2) If they DO get sick, choose ONE condition from:
   - "heat_exhaustion"
   - "heatstroke"
   - "dehydration"
   - "kidney_strain"
   - "heat_rash"
   Match the condition to the age_group and disease_group as much as possible.

3) Estimate severity between 0 and 1, where 1 is life-threatening.
   Older and fragile groups with cardio/renal diseases should tend to higher severity.

4) Decide if they NEED HOSPITAL urgently TODAY.

You must still output a condition and severity, even if sickness_probability is low.
The simulation will draw a random number and only apply the condition
if random < sickness_probability.

Patient:
- id: {agent.id}
- age: {agent.age}
- age_group: {age_group_str}
- disease_group: {disease_group_str}
- chronic_disease_name: {agent.disease_name}
- sex: {agent.sex}
- job: {agent.job}
- current_health_after_update: {agent.health:.2f}
- current_mental_after_update: {agent.mental:.2f}
- current_hydration: {agent.hydration:.2f}
- existing_heat_condition: {agent.heat_condition}
- today's_outdoor_hours: {final_outdoor_hours:.2f}
- base_sickness_probability_hint: {base_prob:.2f}

Weather today:
- temperature_celsius: {env.temperature:.1f}
- humidity_fraction: {env.humidity:.2f}
- AC_available: {env.ac_available}

Return JSON ONLY:

{{
  "sickness_probability": <float 0-1>,
  "condition": "heat_exhaustion" | "heatstroke" | "dehydration" |
               "kidney_strain" | "heat_rash",
  "severity": <float 0-1>,
  "needs_hospital": <true or false>,
  "description": "<short explanation of why>"
}}
""".strip()


def parse_heat_disease_response(text: str) -> Dict[str, Any]:
    text = clean_llm_json(text)
    data = json.loads(text)
    return {
        "sickness_probability": float(data.get("sickness_probability", 0.0)),
        "condition": data.get("condition", "dehydration"),
        "severity": float(data.get("severity", 0.0)),
        "needs_hospital": bool(data.get("needs_hospital", False)),
        "description": data.get("description", ""),
    }


# =========================================================
# FAMILY & COMPANY PROMPTS
# =========================================================

def build_family_conversation_prompt(members: List[Agent], env: EnvironmentState) -> str:
    people_desc = "\n".join(
        f"- id {a.id}: {a.age}y ({age_to_group(a.age)}), {a.sex}, job={a.job}, "
        f"health={a.health:.2f}, mental={a.mental:.2f}"
        for a in members
    )
    return f"""
You are simulating a FAMILY group chat during a heatwave.
The family discusses today's weather and agrees what each person should do.

Family members:
{people_desc}

Weather today:
- temperature: {env.temperature:.1f} °C
- humidity: {env.humidity:.0%}
- AC available at home: {env.ac_available}

For EACH member, choose:
- drink_water: true/false
- go_hospital: true/false
- use_ac: true/false
- target_outdoor_hours: 0-24

Return JSON ONLY like:
{{
  "members": [
    {{
      "agent_id": <int>,
      "drink_water": <true|false>,
      "go_hospital": <true|false>,
      "use_ac": <true|false>,
      "target_outdoor_hours": <float>
    }},
    ...
  ]
}}
""".strip()


def parse_family_plan(response: str) -> Dict[int, Dict[str, Any]]:
    response = clean_llm_json(response)
    data = json.loads(response)   # Here to handle.
    out: Dict[int, Dict[str, Any]] = {}
    for m in data["members"]:
        out[int(m["agent_id"])] = {
            "drink_water": bool(m["drink_water"]),
            "go_hospital": bool(m["go_hospital"]),
            "use_ac": bool(m["use_ac"]),
            "target_outdoor_hours": float(m["target_outdoor_hours"]),
        }
    return out


def build_company_conversation_prompt(members: List[Agent], env: EnvironmentState) -> str:
    people_desc = "\n".join(
        f"- id {a.id}: {age_to_group(a.age)}, job={a.job}, "
        f"health={a.health:.2f}, mental={a.mental:.2f}"
        for a in members
    )
    return f"""
You are simulating a COMPANY safety meeting during a heatwave.
Managers and workers decide safety rules for today.

Workers:
{people_desc}

Weather:
- temperature: {env.temperature:.1f} °C
- humidity: {env.humidity:.0%}
- AC available in the workplace: {env.ac_available}

For EACH worker, choose:
- drink_water: true/false
- go_hospital: true/false
- use_ac: true/false
- target_outdoor_hours: 0-24

Return JSON ONLY like:
{{
  "members": [
    {{
      "agent_id": <int>,
      "drink_water": <true|false>,
      "go_hospital": <true|false>,
      "use_ac": <true|false>,
      "target_outdoor_hours": <float>
    }},
    ...
  ]
}}
""".strip()


def parse_company_plan(response: str) -> Dict[int, Dict[str, Any]]:
    response = clean_llm_json(response)
    data = json.loads(response)
    out: Dict[int, Dict[str, Any]] = {}
    for m in data["members"]:
        out[int(m["agent_id"])] = {
            "drink_water": bool(m["drink_water"]),
            "go_hospital": bool(m["go_hospital"]),
            "use_ac": bool(m["use_ac"]),
            "target_outdoor_hours": float(m["target_outdoor_hours"]),
        }
    return out


# =========================================================
# WEATHER BROADCAST (optional)
# =========================================================

def build_weather_broadcast_prompt(env: EnvironmentState) -> str:
    return f"""
You are a weather broadcaster issuing a heatwave warning.

Raw data:
- temperature: {env.temperature:.1f} °C
- humidity: {env.humidity:.0%}

Classify risk as "low", "medium", "high", or "critical" and provide a short warning.

Return JSON ONLY like:
{{
  "risk_level": "low" | "medium" | "high" | "critical",
  "message": "<short human-readable warning>"
}}
""".strip()


def parse_weather_announcement(response: str) -> Dict[str, Any]:
    response = clean_llm_json(response)
    data = json.loads(response)
    risk = data.get("risk_level", "medium")
    if risk == "critical":
        mult = 1.4
    elif risk == "high":
        mult = 1.3
    elif risk == "medium":
        mult = 1.15
    else:
        mult = 1.0
    return {
        "risk_multiplier": mult,
        "message": data.get("message", ""),
    }


def llm_weather_announcement(env: EnvironmentState) -> Dict[str, Any]:
    """
    Optional global weather announcement (for logging / UI).
    Respects GLOBAL_LLM_ENABLED, so if you pass --no-llm this won't be used.
    """
    if USE_REAL_LLM and GLOBAL_LLM_ENABLED:
        prompt = build_weather_broadcast_prompt(env)
        text = call_llm_api(prompt, tag=f"weather_day_{env.day}")
        return parse_weather_announcement(text)

    # Heuristic fallback when LLM is disabled or not configured
    risk_multiplier = 1.0
    if env.temperature > 40 or env.humidity > 0.7:
        risk_multiplier = 1.3
        risk_level = "high"
    elif env.temperature > 35:
        risk_multiplier = 1.15
        risk_level = "medium"
    else:
        risk_level = "low"

    message = (
        f"Day {env.day}: {env.temperature:.1f}°C, {env.humidity:.0%} humidity. "
        f"Heat stress risk is {risk_level}. Stay hydrated and avoid the hottest hours."
    )
    print(f"[WEATHER] Heuristic announcement: {message}")
    return {
        "risk_multiplier": risk_multiplier,
        "message": message,
    }
