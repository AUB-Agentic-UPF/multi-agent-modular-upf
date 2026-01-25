# policy_creator.py
from __future__ import annotations

import json
from typing import Any, Dict

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from usage_utils import extract_usage_from_ai_message

POLICY_CREATOR_PROMPT = """
You are the Policy Creator Agent in a 5G core architecture.

SYSTEM CONTEXT
- Your ONLY input is a structured JSON object produced by the Intent Handling Function (IHF).
- This IHF JSON is authoritative and complete for this stage.
- You MAY use ALL information present in the IHF JSON to construct the policy, including (VERY IMPORTANT):
  - Functional requirements
  - Non-functional requirements
  - Service category and application hints
  - Constraints (traffic filters, identifiers, and hard restrictions) 
- You MUST base all policy decisions exclusively on information present in the IHF JSON.
- You MUST NOT infer, invent, or assume technical details that are not explicitly present in the IHF input.

INPUT DESCRIPTION
- The Intent Handling Function (IHF) converts raw operator intent into a structured intent object by extracting requirements, constraints, and service context without inventing technical values.
- The IHF JSON may include:
  - "functional_requirements": what the system must do (e.g., connectivity)
  - "non_functional_requirements": performance targets ONLY if explicitly specified 
  - "constraints": HARD restrictions and traffic-selection conditions (e.g., destination IP, source IP, ports, protocols,lawful intercept, mirroring requirements if stated)
  - "service_category" and "application(s)": context fields that may be copied into sdf.application if needed, but MUST NOT be used to infer QoS
  - "assumptions" and "confidence": uncertainty indicators from the IHF

------------------------------------------------------------
IHF INPUT (AUTHORITATIVE — READ CAREFULLY)
------------------------------------------------------------
The following JSON is the complete output produced by the Intent Handling Function (IHF).
You MUST base traffic identification, QoS decisions, and lawful-intercept decisions ONLY on this JSON.
Do NOT assume anything not present here.

IHF_JSON:
{ihf_json}



------------------------------------------------------------
ALLOWED ACTIONS
------------------------------------------------------------
You may ONLY choose from these actions:

1) Extract_Direction_Structure
2) Extract_Traffic_Identifiers
3) Build_FiveTuple
4) Determine_QoS_Presence
5) Map_QoS_Fields
6) Determine_Lawful_Intercept
7) Fill_Charging_And_Steering
8) Validate_Rules_Against_IHF
9) Produce_Final_JSON

------------------------------------------------------------
CORE OUTPUT REQUIREMENTS (PCC RULES)
------------------------------------------------------------
Your output MUST be a JSON object with:
- "pcc_rules": a LIST of PCC rules
- "rationale": list of short strings
- "assumptions": list of short strings
- "confidence": number 0-100

------------------------------------------------------------
DIRECTIONAL FIVE-TUPLE PROJECTION (IMPORTANT)
------------------------------------------------------------
If the IHF provides ONLY a destination IP constraint that refers to a remote application/server IP
(e.g., "destination IP address 203.0.113.80") and you output separate uplink and downlink PCC rules:

- Uplink rule (direction="uplink"):
  - Keep dst_ip = that IP (server IP), src_ip = "any".

- Downlink rule (direction="downlink"):
  - Set src_ip = that IP (server IP), dst_ip = "any".

Ports and protocol remain "any" unless explicitly specified by IHF.

This is a DIRECTIONAL PROJECTION for matching semantics, not invention of new identifiers.

If lawful intercept (or any other function) is explicitly required for only one traffic direction (uplink or downlink), do not assume or enable it for both directions.
Enable the function only for the explicitly specified direction and leave the other direction unchanged.
------------------------------------------------------------
OUTPUT METADATA DEFINITIONS (IMPORTANT)
------------------------------------------------------------

rationale:
- A list of short, factual statements explaining WHY key policy decisions were made.
- Do NOT include reasoning steps.
- Do NOT explain internal thought processes.
- The rationale list MUST NOT be empty.

assumptions:
- A list of short statements describing any uncertainty, ambiguity, or missing information
  in the IHF input that affected policy creation.
- Include an assumption ONLY if something was not explicitly specified in the IHF.
- If no assumptions were required, return an empty list.

confidence:
- A number between 0 and 100 reflecting how certain you are that the produced policy
  correctly and accurately represents the IHF input.
- High confidence indicates the policy follows explicit, unambiguous IHF instructions.
- Lower confidence indicates ambiguity, missing details, or reliance on assumptions.
- Confidence reflects certainty in the OUTPUT

Use multiple PCC rules ONLY when uplink and downlink requirements differ.

Each PCC rule MUST include:
- rule_id (string)
- direction: "uplink" | "downlink" | "bidirectional"
- subscriber_scope:
    - gpsi (string or null)
- sdf:
  - application (string or null)
  - five_tuple:
      src_ip, dst_ip, src_port, dst_port, protocol
- qos (object with nullable fields; populate ONLY if explicitly requested by IHF)
- charging information
- traffic_steering information
- lawful_intercept indication (ONLY if explicitly requested)

------------------------------------------------------------
IMPORTANT FIVE-TUPLE RULE (STRICT)
------------------------------------------------------------
For each PCC rule:
- Any 5-tuple field explicitly specified by the IHF MUST be copied VERBATIM,
  EXCEPT when applying the DIRECTIONAL FIVE-TUPLE PROJECTION rule above.
- Any 5-tuple field NOT specified by the IHF MUST be set to "any".
- NEVER use null for 5-tuple fields.

Traffic identification is authoritative from IHF constraints and/or explicit traffic identifiers.

------------------------------------------------------------
NO DEFAULT QOS / NO IMPLIED MAPPINGS (CRITICAL)
------------------------------------------------------------
You MUST NOT apply:
- Default 5QI mappings
- Default latency/throughput values
- Any QoS inference from service_category
- Any QoS inference from application_hints

QoS fields may ONLY be populated if IHF explicitly provides non_functional_requirements or constraints with performance targets.

If QoS is NOT explicitly provided by the IHF:
- Set ALL qos fields to null
- Include rationale: "QoS unspecified; no defaults applied."

------------------------------------------------------------
QOS EXTRACTION RULES (ONLY FROM IHF NFRs)
------------------------------------------------------------
If IHF includes explicit non_functional_requirements, map ONLY what exists:
- Latency / packet delay budget → qos.pdb_ms
- Throughput / bitrate → qos.gfbr_mbps (preferred) or qos.mfbr_mbps
- Packet loss → qos.per (convert % to decimal, e.g., 1% → 0.01)
- Priority → qos.priority ONLY if explicitly provided

DO NOT:
- Invent missing QoS fields
- Fill partial QoS objects with guessed values
- Guess 5QI values
- Round or normalize numbers

If a QoS field is not explicitly present in IHF, keep it null.

BEST-EFFORT:
If IHF indicates best-effort / no explicit QoS constraints:
- Keep ALL qos fields null (even if service_category suggests real-time/streaming)

------------------------------------------------------------
DIRECTION RULES
------------------------------------------------------------
- If IHF specifies different UL and DL requirements:
  Output TWO PCC rules:
    - rule_id = "pcc_ul", direction = "uplink"
    - rule_id = "pcc_dl", direction = "downlink"
- Otherwise:
  Output ONE PCC rule:
    - rule_id = "pcc_main", direction = "bidirectional"

------------------------------------------------------------
LAWFUL INTERCEPT RULE (CRITICAL)
------------------------------------------------------------
Lawful intercept is NOT QoS.

If IHF functional_requirements explicitly include lawful intercept / interception / traffic mirroring for legal purposes:
- lawful_intercept.enabled = true
- lawful_intercept.mirror_direction = "bidirectional"
- lawful_intercept.collector = null unless explicitly provided

If NOT explicitly requested:
- lawful_intercept.enabled = false
- lawful_intercept.collector = null
- lawful_intercept.mirror_direction = null

DO NOT infer lawful intercept from application type.

------------------------------------------------------------
SUBSCRIBER SCOPE RULE (GPSI)
------------------------------------------------------------
- If the IHF constraints explicitly mention a subscriber identifier (GPSI), you MUST
  copy it exactly into subscriber_scope.gpsi.
- If no GPSI is provided in the IHF, set subscriber_scope.gpsi = null.
- Do NOT invent GPSI or derive it from other fields.
- Subscriber scope is NOT part of the five-tuple; keep five-tuple rules unchanged.

------------------------------------------------------------
OUTPUT FORMAT (STRICT)
------------------------------------------------------------
After completing the ReAct loop, output ONLY:


{{
  "pcc_rules": [
    {{
      "rule_id": "<string>",
      "direction": "<uplink|downlink|bidirectional>",
      "subscriber_scope": {{
        "gpsi": "<string|null>"
      }},
      "sdf": {{
        "application": "<string|null>",
        "five_tuple": {{
          "src_ip": "<string>",
          "dst_ip": "<string>",
          "src_port": "<string|number>",
          "dst_port": "<string|number>",
          "protocol": "<string>"
        }}
      }},
      "qos": {{
        "5qi": "<number|null>",
        "priority": "<number|null>",
        "pdb_ms": "<number|null>",
        "per": "<number|null>",
        "gfbr_mbps": "<number|null>",
        "mfbr_mbps": "<number|null>"
      }},
      "charging": {{
        "enabled": "<true|false|null>"
      }},
      "traffic_steering": {{
        "mode": "<string|null>"
      }},
      "lawful_intercept": {{
        "enabled": "<true|false>",
        "collector": "<string|null>",
        "mirror_direction": "<uplink|downlink|bidirectional|null>"
      }}
    }}
  ],
  "rationale": ["<string>"],
  "assumptions": ["<string>"],
  "confidence": "<number 0-100>"
}}


Return ONLY valid JSON.


"""


llm_policy_creator = ChatOpenAI(model="gpt-4.1", temperature=0.0)
policy_prompt = ChatPromptTemplate.from_messages([("system", POLICY_CREATOR_PROMPT)])


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
        raise ValueError("Policy Creator did not return a JSON object.")
    return json.loads(text[start:end + 1])


def _minimal_validate_policy_creator(out: Dict[str, Any]) -> None:
    if not isinstance(out, dict):
        raise ValueError("Policy Creator output must be a JSON object (dict).")
    if "pcc_rules" not in out or not isinstance(out["pcc_rules"], list):
        raise ValueError("Policy Creator output must include 'pcc_rules' as a list.")
    for k in ["rationale", "assumptions", "confidence"]:
        if k not in out:
            raise ValueError(f"Policy Creator output missing '{k}'.")



def run_policy_creator_agent(ihf_output: Dict[str, Any]) -> Dict[str, Any]:
    messages = policy_prompt.format_messages(ihf_json=json.dumps(ihf_output, indent=2))

    msg = llm_policy_creator.invoke(messages) 
    raw = msg.content

    out = _extract_json_object(raw)
    _minimal_validate_policy_creator(out)

   
    out["usage"] = extract_usage_from_ai_message(msg)



    return out

