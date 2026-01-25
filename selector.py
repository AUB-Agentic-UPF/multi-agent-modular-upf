# selector.py
from __future__ import annotations

import json
from typing import Any, Dict, List

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from usage_utils import extract_usage_from_ai_message

SELECTOR_PROMPT = """
You are the Selector Agent in a modular, microservice-based 5G User Plane Function (UPF).

------------------------------------------------------------
ROLE & RESPONSIBILITY
------------------------------------------------------------
Your responsibility is to SELECT which UPF modules (microservices)
must be instantiated and which software and hardware profiles they
should use.

You MUST:
- Select UPF modules based ONLY on the provided PCC rules.
- Select an appropriate profile for each selected module.
- Explicitly justify each selection decision.

You MUST NOT:
- Modify PCC rules.
- Invent new requirements.
- Configure or deploy modules.
- Optimize beyond what is implied by the PCC rules.
- Infer policies not present in the input.

------------------------------------------------------------
INPUT (AUTHORITATIVE)
------------------------------------------------------------
Your ONLY input is the structured PCC policy output produced by the Policy Creator.

PCC_RULES_JSON:
{pcc_rules_json}

You MUST base all decisions exclusively on this input.

------------------------------------------------------------
UPF MODULE DEFINITIONS (AUTHORITATIVE)
------------------------------------------------------------

1) Ingress Steering Function (ISF)
- Handles incoming packets at the UPF.
- Performs admission control and packet validation.
- Steers packets to uplink or downlink processing paths.
- Steering may be based on:
  - Presence of GTP-U header (uplink)
  - Absence of GTP-U header (downlink)
  - Control-plane rules mapping ingress ports to UL/DL paths
- May implement load balancing when multiple ULF/DLF replicas exist.

2) Uplink Function (ULF)
- Handles packets flowing from UE → gNB → UPF → Data Network.
- Performs all required uplink packet processing assigned to the UPF.

3) Downlink Function (DLF)
- Handles packets flowing from Data Network → UPF → gNB → UE.
- Performs all required downlink packet processing assigned to the UPF.

4) On-Demand Functions (ODFs)
- Optional UPF microservices activated only when explicitly required.
- Examples include:
  - Lawful Intercept / Traffic Mirroring
  - Specialized monitoring or inspection functions
- MUST be selected ONLY if explicitly required by PCC rules.

------------------------------------------------------------
MODULE SELECTION RULES
------------------------------------------------------------

- ISF MUST always be selected.
- ISF MUST be the first module in selected_modules.
- ULF MUST be selected if any PCC rule applies to uplink or bidirectional traffic.
- DLF MUST be selected if any PCC rule applies to downlink or bidirectional traffic.
- ODFs MUST be selected ONLY if explicitly required by the PCC rules
  (e.g., lawful_intercept.enabled = true).

DO NOT infer optional modules from application type alone.

------------------------------------------------------------
PROFILE SELECTION (SINGLE FIELD)
------------------------------------------------------------

For EACH selected module, you MUST select exactly ONE profile string
in the field "profile".

- "profile" can be either a software profile or a hardware profile.
- Do NOT output separate software_profile / hardware_profile fields.

ALLOWED PROFILES (STRICT ENUMS)
- ISF: profile in ["ISF_sw", "ISF_hw"]
- ULF: profile in ["ULF_sw", "ULF_hw"]
- DLF: profile in ["DLF_sw", "DLF_hw"]
- ODF: profile in ["ODF_sw", "ODF_hw"]

Direction-aware profile selection (IMPORTANT):
- Evaluate QoS constraints per module using ONLY PCC rules that apply to that module's direction:
  - ULF profile decision MUST be based only on PCC rules with direction in ["uplink","bidirectional"].
  - DLF profile decision MUST be based only on PCC rules with direction in ["downlink","bidirectional"].
  - ISF profile decision is based on overall traffic classification complexity; if no explicit QoS constraints exist, prefer ISF_sw.
- If ANY applicable PCC rule for that module has at least one non-null QoS field
  (e.g., pdb_ms, per, gfbr_mbps, mfbr_mbps, 5qi, priority),
  then that module MUST choose the hardware profile.
- If ALL applicable PCC rules for that module have all QoS fields null, then that module MUST choose the software profile.

Guidelines:
- If PCC rules include strict latency/throughput/reliability constraints:
  → choose the hardware profile (e.g., "ULF_hw").
- If PCC rules indicate best-effort traffic with no explicit QoS guarantees:
  → choose the software profile (e.g., "ULF_sw").
- If lawful intercept / mirroring is enabled:
  → select ODF and choose "ODF_hw" unless policy is vague.
- Do NOT invent numeric capacities or hardware specs.

Allowed Actions:
- Analyze_PCC_Directions
- Select_Core_Modules
- Select_OnDemand_Modules
- Select_Software_Profiles
- Select_Hardware_Profiles
- Validate_Selections
- Produce_Final_JSON


------------------------------------------------------------
OUTPUT METADATA DEFINITIONS (IMPORTANT)
------------------------------------------------------------

rationale:
- A REQUIRED list of short, factual statements explaining the selection decisions made.
- Each entry should explain WHAT was selected and WHY, in objective terms.
- Do NOT include step-by-step reasoning.
- Do NOT describe internal thought processes.
- The rationale list MUST NOT be empty.

assumptions:
- A list of short statements describing any uncertainty, ambiguity, or missing information
  in the PCC rules that affected module or profile selection.
- Include an assumption ONLY if something was not explicitly specified.
- If no assumptions were required, return an empty list.

confidence:
- A number between 0 and 100 reflecting how certain you are that the selected modules
  and profiles correctly satisfy the PCC rules.

------------------------------------------------------------
OUTPUT FORMAT (STRICT — JSON ONLY)
------------------------------------------------------------

Return ONLY a valid JSON object matching EXACTLY this schema:

{{
  "selected_modules": [
    {{
      "module": "<ISF|ULF|DLF|ODF>",
      "profile": "<ISF_sw|ISF_hw|ULF_sw|ULF_hw|DLF_sw|DLF_hw|ODF_sw|ODF_hw>"
    }}
  ],
  "rationale": ["<string>"],
  "assumptions": ["<string>"],
  "confidence": "<number 0-100>"
}}



"""

llm_selector = ChatOpenAI(model="gpt-4.1", temperature=0.0)
selector_prompt = ChatPromptTemplate.from_messages([("system", SELECTOR_PROMPT)])



def _extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1]).strip()
        else:
            text = "\n".join(lines[1:]).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Selector did not return a JSON object.")
    return json.loads(text[start:end + 1])


def _validate_selected_chain(out: Dict[str, Any]) -> None:
    allowed_profiles_by_module = {
        "ISF": {"ISF_sw", "ISF_hw"},
        "ULF": {"ULF_sw", "ULF_hw"},
        "DLF": {"DLF_sw", "DLF_hw"},
        "ODF": {"ODF_sw", "ODF_hw"},
    }

    if "selected_modules" not in out or not isinstance(out["selected_modules"], list):
        raise ValueError("Output must include 'selected_modules' as a list.")

    for item in out["selected_modules"]:
        if not isinstance(item, dict):
            raise ValueError("Each item in selected_modules must be an object.")

        mod = item.get("module")
        profile = item.get("profile")

        if mod not in allowed_profiles_by_module:
            raise ValueError(f"Invalid module: {mod}")

        if not isinstance(profile, str) or not profile.strip():
            raise ValueError(f"profile must be a non-empty string for module '{mod}'.")

        if profile not in allowed_profiles_by_module[mod]:
            raise ValueError(
                f"Invalid profile '{profile}' for module '{mod}'. "
                f"Allowed: {sorted(allowed_profiles_by_module[mod])}"
            )

    conf = out.get("confidence")
    if not isinstance(conf, (int, float)):
        raise ValueError("Output must include numeric 'confidence'.")
    if conf < 0 or conf > 100:
        raise ValueError("'confidence' must be between 0 and 100.")

    rationale = out.get("rationale")
    if not isinstance(rationale, list) or len(rationale) == 0:
        raise ValueError("Output must include non-empty 'rationale' list.")
    for r in rationale:
        if not isinstance(r, str) or not r.strip():
            raise ValueError("Each entry in 'rationale' must be a non-empty string.")

    assumptions = out.get("assumptions")
    if not isinstance(assumptions, list):
        raise ValueError("Output must include 'assumptions' list.")
    for a in assumptions:
        if not isinstance(a, str) or not a.strip():
            raise ValueError("Each entry in 'assumptions' must be a non-empty string.")

        
def run_selector_agent(pcc_rules: List[Dict[str, Any]]) -> Dict[str, Any]:
    messages = selector_prompt.format_messages(
        pcc_rules_json=json.dumps(pcc_rules, indent=2),
    )

    msg = llm_selector.invoke(messages)  
    raw = msg.content

    out = _extract_json_object(raw)
    _validate_selected_chain(out)

    out["usage"] = extract_usage_from_ai_message(msg)

    return out
