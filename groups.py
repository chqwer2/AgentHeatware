from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any

from agents import Agent, age_to_group
from environment import EnvironmentState
from llm_interfaces import (
    USE_REAL_LLM,
    GLOBAL_LLM_ENABLED,
    build_family_conversation_prompt,
    build_company_conversation_prompt,
    call_llm_api,
    parse_family_plan,
    parse_company_plan,
)


@dataclass
class Family:
    id: int
    member_ids: List[int]

    # ------------ Pure rule-based planning (NO LLM) ------------

    def plan_rule_based(
        self,
        agents_by_id: Dict[int, Agent],
        env: EnvironmentState,
    ) -> Dict[int, Dict[str, Any]]:
        """
        Always rule-based, no LLM calls.
        """
        members = [agents_by_id[i] for i in self.member_ids]
        plan: Dict[int, Dict[str, Any]] = {}
        print(f"[FAMILY] Family {self.id} day {env.day} using RULE-BASED plan.")
        for ag in members:
            high_risk = (ag.health < 0.5) or (ag.age > 70) or ag.has_chronic_disease
            act = {
                "drink_water": True,
                "go_hospital": high_risk and (env.temperature > 40),
                "use_ac": env.ac_available,
                "target_outdoor_hours": ag.outdoor_hours * (0.4 if high_risk else 0.7),
            }
            plan[ag.id] = act
            print(f"   member {ag.id} ({age_to_group(ag.age)}): {act}")
        return plan

    # ------------ LLM-augmented planning ------------

    def plan_with_llm(
        self,
        agents_by_id: Dict[int, Agent],
        env: EnvironmentState,
    ) -> Dict[int, Dict[str, Any]]:
        """
        Family-level coordination via LLM.
        Falls back to rule-based if LLM is not effectively available/enabled.
        """
        members = [agents_by_id[i] for i in self.member_ids]

        # If no real LLM or disabled globally → rule-based
        if not (USE_REAL_LLM and GLOBAL_LLM_ENABLED):
            return self.plan_rule_based(agents_by_id, env)

        try:
            prompt = build_family_conversation_prompt(members, env)
            print(f"[LLM-FAMILY] Family {self.id} day {env.day} calling LLM ...")
            response = call_llm_api(prompt, tag=f"family_{self.id}_day_{env.day}")
            plan = parse_family_plan(response)
            print(f"[LLM-FAMILY] Family {self.id} got plan:")
            for aid, act in plan.items():
                print(f"   member {aid}: {act}")
            return plan
        except Exception as e:
            print(f"[LLM-FAMILY] Error for family {self.id} day {env.day}: {e}")
            print("[LLM-FAMILY] Falling back to RULE-BASED plan.")
            return self.plan_rule_based(agents_by_id, env)


@dataclass
class Company:
    id: int
    member_ids: List[int]

    # ------------ Pure rule-based planning (NO LLM) ------------

    def plan_rule_based(
        self,
        agents_by_id: Dict[int, Agent],
        env: EnvironmentState,
    ) -> Dict[int, Dict[str, Any]]:
        """
        Always rule-based, no LLM calls.
        """
        members = [agents_by_id[i] for i in self.member_ids]
        plan: Dict[int, Dict[str, Any]] = {}
        print(f"[COMPANY] Company {self.id} day {env.day} using RULE-BASED plan.")
        for ag in members:
            outdoor_base = ag.outdoor_hours
            if ag.job in ("construction", "delivery", "farmer"):
                target = outdoor_base * 0.4
            else:
                target = outdoor_base * 0.8

            act = {
                "drink_water": True,
                "go_hospital": False,
                "use_ac": env.ac_available,
                "target_outdoor_hours": target,
            }
            plan[ag.id] = act
            print(f"   worker {ag.id} ({age_to_group(ag.age)}): {act}")
        return plan

    # ------------ LLM-augmented planning ------------

    def plan_with_llm(
        self,
        agents_by_id: Dict[int, Agent],
        env: EnvironmentState,
    ) -> Dict[int, Dict[str, Any]]:
        """
        Company-level coordination via LLM.
        Falls back to rule-based if LLM is not effectively available/enabled.
        """
        members = [agents_by_id[i] for i in self.member_ids]

        if not (USE_REAL_LLM and GLOBAL_LLM_ENABLED):
            return self.plan_rule_based(agents_by_id, env)

        try:
            prompt = build_company_conversation_prompt(members, env)
            print(f"[LLM-COMPANY] Company {self.id} day {env.day} calling LLM ...")
            response = call_llm_api(prompt, tag=f"company_{self.id}_day_{env.day}")
            plan = parse_company_plan(response)
            print(f"[LLM-COMPANY] Company {self.id} got plan:")
            for aid, act in plan.items():
                print(f"   worker {aid}: {act}")
            return plan
        except Exception as e:
            print(f"[LLM-COMPANY] Error for company {self.id} day {env.day}: {e}")
            print("[LLM-COMPANY] Falling back to RULE-BASED plan.")
            return self.plan_rule_based(agents_by_id, env)
