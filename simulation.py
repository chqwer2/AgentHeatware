from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Any

from agents import Agent, age_to_group, disease_to_group
from environment import EnvironmentState
from groups import Family, Company
from llm_interfaces import (
    llm_weather_announcement,
    USE_REAL_LLM,
    GLOBAL_LLM_ENABLED,
    build_personal_risk_prompt,
    call_llm_api,
    parse_personal_risk_response,
    build_heat_disease_prompt,
    parse_heat_disease_response,
)


@dataclass
class SimulationResult:
    history: List[Dict[str, float]]
    agents: List[Agent]
    agent_daily_log: List[Dict[str, Any]]


@dataclass
class Simulation:
    agents: List[Agent]
    weather_series: List[Tuple[float, float]]
    params: Dict[str, float]
    families: List[Family] = field(default_factory=list)
    companies: List[Company] = field(default_factory=list)

    # "rule" or "llm" for DIRECT per-agent decisions
    # (but if use_llm=False, we will force "rule" internally)
    decision_mode: str = "rule"

    # SINGLE flag that controls ALL LLM features:
    # - weather announcement
    # - personal risk LLM
    # - heat disease LLM
    # - family/company LLM plans
    # - (optional) per-agent LLM decisions if decision_mode="llm"
    use_llm: bool = True

    ac_available_fraction: float = 0.5
    group_influence_prob: float = 0.6
    verbose: bool = True

    history: List[Dict[str, float]] = field(default_factory=list)
    agent_daily_log: List[Dict[str, Any]] = field(default_factory=list)

    def run(self) -> SimulationResult:
        agents_by_id = {a.id: a for a in self.agents}
        total_days = len(self.weather_series)

        # Effective: we can only really use LLM if BOTH
        # - Simulation.use_llm is True
        # - llm_interfaces knows we have a real provider & global is enabled
        effective_use_llm = self.use_llm and USE_REAL_LLM and GLOBAL_LLM_ENABLED

        if self.verbose:
            print(
                f"[SIM] Starting simulation for {total_days} days, "
                f"{len(self.agents)} agents. use_llm={self.use_llm}, "
                f"effective_use_llm={effective_use_llm}"
            )

        import random
        for day, (temp, hum) in enumerate(self.weather_series):
            if self.verbose:
                print(f"\n[SIM] Day {day+1}/{total_days}: temp={temp:.1f}°C, hum={hum:.0%}")

            env = EnvironmentState(
                day=day,
                temperature=temp,
                humidity=hum,
                ac_available=(random.random() < self.ac_available_fraction),
            )

            # 1) Optional global weather broadcast (ONLY if effective_use_llm)
            if effective_use_llm:
                _announcement = llm_weather_announcement(env)

            # 2) Family + company coordination -> group_plans
            group_plans: Dict[int, Dict[str, Any]] = {}

            for fam in self.families:
                if effective_use_llm:
                    fam_plan = fam.plan_with_llm(agents_by_id, env)
                else:
                    fam_plan = fam.plan_rule_based(agents_by_id, env)

                for aid, action in fam_plan.items():
                    if aid not in group_plans:
                        group_plans[aid] = {"action": action.copy(), "sources": ["family"]}
                    else:
                        group_plans[aid]["action"].update(action)
                        group_plans[aid]["sources"].append("family")

            for comp in self.companies:
                if effective_use_llm:
                    comp_plan = comp.plan_with_llm(agents_by_id, env)
                else:
                    comp_plan = comp.plan_rule_based(agents_by_id, env)

                for aid, action in comp_plan.items():
                    if aid not in group_plans:
                        group_plans[aid] = {"action": action.copy(), "sources": ["company"]}
                    else:
                        group_plans[aid]["action"].update(action)
                        group_plans[aid]["sources"].append("company")

            # 3) Each agent acts with personal LLM risk + sometimes group plan
            for ag in self.agents:
                health_before = ag.health
                mental_before = ag.mental
                hydration_before = ag.hydration

                personal_risk_hint = 1.0
                personal_risk_info = None

                # ---- personal risk via LLM (ONLY if effective_use_llm) ----
                if effective_use_llm:
                    prompt = build_personal_risk_prompt(ag, env)
                    try:
                        text = call_llm_api(prompt, tag=f"personal_risk_agent_{ag.id}_day_{day}")
                        personal_risk_info = parse_personal_risk_response(text)
                        personal_risk_hint = personal_risk_info["overall_risk_multiplier"]
                        if self.verbose:
                            print(
                                f"[PERSONAL-LLM] Day {day} Agent {ag.id} "
                                f"age_group={personal_risk_info['age_group']} "
                                f"disease_group={personal_risk_info['disease_group']} "
                                f"level={personal_risk_info['risk_level']} "
                                f"mult={personal_risk_hint:.2f} "
                                f"sick_prob={personal_risk_info['sickness_base_probability']:.2f}"
                            )
                    except Exception as e:
                        print(f"[PERSONAL-LLM] Error for agent {ag.id} day {day}: {e}")

                # ---- Base action: from personal risk LLM or rule-based ----
                # If no personal_risk_info, fall back to agent.decide()
                if personal_risk_info is not None:
                    base_action = personal_risk_info["actions"]
                    sickness_base_prob = personal_risk_info["sickness_base_probability"]
                else:
                    # If LLM is disabled, ALWAYS force rule-based decisions here
                    decision_mode = self.decision_mode
                    if not effective_use_llm:
                        decision_mode = "rule"

                    base_action = ag.decide(
                        env,
                        self.params,
                        mode=decision_mode,
                        risk_hint=personal_risk_hint,
                    )
                    sickness_base_prob = 0.0

                group_info = group_plans.get(ag.id, None)
                used_group_plan = False
                group_sources: List[str] = []

                if group_info is not None:
                    group_action = group_info["action"]
                    group_sources = group_info["sources"]
                    if random.random() < self.group_influence_prob:
                        final_action = {**base_action, **group_action}
                        used_group_plan = True
                        if self.verbose:
                            print(
                                f"[ACT] Day {day} Agent {ag.id} "
                                f"FOLLOWED group {group_sources}. "
                                f"base={base_action} group={group_action} -> final={final_action}"
                            )
                    else:
                        final_action = base_action
                        if self.verbose:
                            print(
                                f"[ACT] Day {day} Agent {ag.id} "
                                f"IGNORED group {group_sources}. "
                                f"final={final_action}"
                            )
                else:
                    final_action = base_action
                    if self.verbose:
                        print(
                            f"[ACT] Day {day} Agent {ag.id} "
                            f"NO group plan. final={final_action}"
                        )

                # Update state
                ag.update_state(env, self.params, final_action)

                # 4) Heat-related disease LLM prompt with probability
                heat_disease_info = None
                final_outdoor_hours = float(final_action.get("target_outdoor_hours", ag.outdoor_hours))

                if (
                    effective_use_llm
                    and ag.heat_condition == "none"
                    and (ag.health < 0.5 or ag.mental < 0.5)
                ):
                    try:
                        hd_prompt = build_heat_disease_prompt(
                            agent=ag,
                            env=env,
                            final_outdoor_hours=final_outdoor_hours,
                            base_prob=sickness_base_prob,
                        )
                        hd_text = call_llm_api(hd_prompt, tag=f"heat_disease_agent_{ag.id}_day_{day}")
                        heat_disease_info = parse_heat_disease_response(hd_text)

                        prob = heat_disease_info["sickness_probability"]
                        cond = heat_disease_info["condition"]
                        sev = heat_disease_info["severity"]

                        r = random.random()
                        if self.verbose:
                            print(
                                f"[HEAT-DISEASE-LLM] Day {day} Agent {ag.id} "
                                f"candidate={cond} sev={sev:.2f} prob={prob:.2f} rand={r:.2f}"
                            )

                        if r < prob:
                            ag.heat_condition = cond
                            ag.heat_condition_severity = sev

                            extra_h = 0.1 * sev
                            extra_m = 0.05 * sev
                            ag.health = max(0.0, ag.health - extra_h)
                            ag.mental = max(0.0, ag.mental - extra_m)

                            if self.verbose:
                                print(
                                    f"[HEAT-DISEASE] Day {day} Agent {ag.id} "
                                    f"=> SICK with {cond} (sev={sev:.2f})"
                                )
                    except Exception as e:
                        print(f"[HEAT-DISEASE] Error for agent {ag.id} day {day}: {e}")

                # Log
                self._log_agent_day(
                    day=day,
                    agent=ag,
                    env=env,
                    base_action=base_action,
                    final_action=final_action,
                    used_group_plan=used_group_plan,
                    group_sources=group_sources,
                    personal_risk_hint=personal_risk_hint,
                    personal_risk_info=personal_risk_info,
                    heat_disease_info=heat_disease_info,
                    health_before=health_before,
                    mental_before=mental_before,
                    hydration_before=hydration_before,
                )

            self._record_day(day)

        if self.verbose:
            print("[SIM] Simulation complete.")

        return SimulationResult(
            history=self.history,
            agents=self.agents,
            agent_daily_log=self.agent_daily_log,
        )

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

    def _log_agent_day(
        self,
        day: int,
        agent: Agent,
        env: EnvironmentState,
        base_action: Dict[str, Any],
        final_action: Dict[str, Any],
        used_group_plan: bool,
        group_sources: List[str],
        personal_risk_hint: float,
        personal_risk_info: Any,
        heat_disease_info: Any,
        health_before: float,
        mental_before: float,
        hydration_before: float,
    ) -> None:
        risk_factors = personal_risk_info["factors"] if personal_risk_info else {}
        temp_f = risk_factors.get("temperature", {})
        hum_f = risk_factors.get("humidity", {})
        age_f = risk_factors.get("age_group", {})
        dis_f = risk_factors.get("disease_group", {})

        log_entry = {
            "day": day,
            "agent_id": agent.id,
            "age": agent.age,
            "age_group": age_to_group(agent.age),
            "sex": agent.sex,
            "job": agent.job,
            "has_chronic_disease": agent.has_chronic_disease,
            "disease_name": agent.disease_name,
            "disease_group": disease_to_group(agent.disease_name, agent.has_chronic_disease),
            "heat_condition": agent.heat_condition,
            "heat_condition_severity": agent.heat_condition_severity,

            "temperature": env.temperature,
            "humidity": env.humidity,
            "ac_available_env": env.ac_available,

            "personal_risk_multiplier": personal_risk_hint,
            "personal_risk_level": personal_risk_info["risk_level"] if personal_risk_info else "",
            "personal_sickness_base_prob": personal_risk_info["sickness_base_probability"] if personal_risk_info else 0.0,
            "factor_temp_severity": float(temp_f.get("severity", 0.0)),
            "factor_temp_how": temp_f.get("how", ""),
            "factor_hum_severity": float(hum_f.get("severity", 0.0)),
            "factor_hum_how": hum_f.get("how", ""),
            "factor_age_severity": float(age_f.get("severity", 0.0)),
            "factor_age_how": age_f.get("how", ""),
            "factor_dis_severity": float(dis_f.get("severity", 0.0)),
            "factor_dis_how": dis_f.get("how", ""),

            "base_drink_water": bool(base_action.get("drink_water", False)),
            "base_go_hospital": bool(base_action.get("go_hospital", False)),
            "base_use_ac": bool(base_action.get("use_ac", False)),
            "base_outdoor_hours": float(base_action.get("target_outdoor_hours", agent.outdoor_hours)),

            "final_drink_water": bool(final_action.get("drink_water", False)),
            "final_go_hospital": bool(final_action.get("go_hospital", False)),
            "final_use_ac": bool(final_action.get("use_ac", False)),
            "final_outdoor_hours": float(final_action.get("target_outdoor_hours", agent.outdoor_hours)),

            "used_group_plan": used_group_plan,
            "group_sources": ",".join(group_sources) if group_sources else "",

            "health_before": health_before,
            "mental_before": mental_before,
            "hydration_before": hydration_before,

            "health_after": agent.health,
            "mental_after": agent.mental,
            "hydration_after": agent.hydration,
        }

        if heat_disease_info:
            log_entry.update(
                {
                    "heat_disease_candidate": heat_disease_info["condition"],
                    "heat_disease_candidate_severity": heat_disease_info["severity"],
                    "heat_disease_candidate_prob": heat_disease_info["sickness_probability"],
                    "heat_disease_needs_hospital": heat_disease_info["needs_hospital"],
                    "heat_disease_description": heat_disease_info["description"],
                }
            )
        else:
            log_entry.update(
                {
                    "heat_disease_candidate": "",
                    "heat_disease_candidate_severity": 0.0,
                    "heat_disease_candidate_prob": 0.0,
                    "heat_disease_needs_hospital": False,
                    "heat_disease_description": "",
                }
            )

        self.agent_daily_log.append(log_entry)
