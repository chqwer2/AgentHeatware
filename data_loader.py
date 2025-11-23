from __future__ import annotations
from typing import List, Tuple, Dict
from collections import defaultdict
import csv

from agents import Agent
from groups import Family, Company


def load_agents_from_csv(path: str) -> Tuple[List[Agent], List[Family], List[Company]]:
    """
    Load agents and their group memberships from a CSV file.

    Required columns:
      agent_id, age, sex, job,
      baseline_health, baseline_mental,
      outdoor_hours, risk_sensitivity,
      family_id, company_id

    Optional columns:
      has_chronic_disease (0/1, true/false)
      disease_name

    family_id / company_id can be empty.
    """
    print(f"[INIT] Loading agents from {path} ...")

    agents: List[Agent] = []
    families_map: Dict[str, List[int]] = defaultdict(list)
    companies_map: Dict[str, List[int]] = defaultdict(list)

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            agent_id = int(row["agent_id"])
            age = int(row["age"])
            sex = row["sex"]
            job = row["job"]

            baseline_health = float(row["baseline_health"])
            baseline_mental = float(row["baseline_mental"])
            outdoor_hours = float(row["outdoor_hours"])
            risk_sensitivity = float(row["risk_sensitivity"])

            has_chronic = (row.get("has_chronic_disease") or "0").strip().lower() in ("1", "true", "yes")
            disease_name = (row.get("disease_name") or "").strip()

            agent = Agent(
                id=agent_id,
                age=age,
                sex=sex,
                job=job,
                baseline_health=baseline_health,
                baseline_mental=baseline_mental,
                outdoor_hours=outdoor_hours,
                risk_sensitivity=risk_sensitivity,
                has_chronic_disease=has_chronic,
                disease_name=disease_name,
            )
            agents.append(agent)

            family_id = (row.get("family_id") or "").strip()
            if family_id != "":
                families_map[family_id].append(agent_id)

            company_id = (row.get("company_id") or "").strip()
            if company_id != "":
                companies_map[company_id].append(agent_id)

    families: List[Family] = [
        Family(id=int(fid), member_ids=member_ids)
        for fid, member_ids in families_map.items()
    ]
    companies: List[Company] = [
        Company(id=int(cid), member_ids=member_ids)
        for cid, member_ids in companies_map.items()
    ]

    print(f"[INIT] Loaded {len(agents)} agents.")
    print(f"[INIT] Created {len(families)} families and {len(companies)} companies.")
    return agents, families, companies
