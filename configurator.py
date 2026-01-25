# configurator.py
from __future__ import annotations

import json
from typing import Any, Dict, List

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from usage_utils import extract_usage_from_ai_message

CONFIGURATOR_PROMPT = """
You are the Configurator Agent in a modular 5G UPF architecture.

------------------------------------------------------------
ROLE & RESPONSIBILITY
------------------------------------------------------------
Your responsibility is to translate:
1) Selected UPF modules and their profiles, and
2) PCC rules (policy),

into a consistent PFCP-like configuration consisting of:
- PDR (Packet Detection Rule)
- FAR (Forwarding Action Rule)
- QER (QoS Enforcement Rule)
- URR (Usage Reporting Rule)

You MUST:
- Produce logically consistent PFCP-like rules that can be applied by the UPF pipeline.
- Use the PCC rules as the authoritative source for traffic matching and QoS.
- Ensure UL and DL directions are handled correctly.
- Ensure the selected modules are sufficient to realize the produced rules.

You MUST NOT:
- Modify or reinterpret the PCC rules.
- Invent missing identifiers or numeric QoS values.
- Add default QoS values or default 5QI mappings.
- Add lawful intercept or mirroring unless explicitly required by PCC rules.

------------------------------------------------------------
INPUTS (AUTHORITATIVE)
------------------------------------------------------------

SELECTOR_OUTPUT_JSON:
{selected_modules_json}

POLICY_CREATOR_OUTPUT_JSON:
{pcc_rules_json}
------------------------------------------------------------
MODULE AUTHORITY (CRITICAL)
------------------------------------------------------------
- The Selector output is the SINGLE source of truth for module availability.
- You MUST use EXACTLY the modules listed in SELECTOR_OUTPUT_JSON.
- You MUST NOT:
  - Assume additional modules exist
  - Add missing modules
  - Remove selected modules
  - Downgrade or upgrade module capabilities
- If a PCC rule requires a capability that is NOT supported by the selected modules:
  - You MUST still produce a configuration using ONLY the provided modules
  - You MUST record this as an assumption
  - You MUST lower confidence
- You are NOT allowed to invent module availability under any circumstances.


Notes:
- Selector output provides selected modules and their profiles.
- Policy Creator output provides PCC rules, including:
  - direction
  - subscriber_scope:
    - gpsi: {{string or null}}
  - sdf.five_tuple
  - qos (nullable fields)
  - traffic_steering
  - lawful_intercept

------------------------------------------------------------
MODULE AVAILABILITY CONSTRAINT
------------------------------------------------------------
You MUST ensure the generated PFCP-like rules are feasible with the selected modules:

- ISF: must exist for ingress steering (assumed always selected).
- ULF: required for any uplink or bidirectional PCC rule.
- DLF: required for any downlink or bidirectional PCC rule.
- ODF: required if lawful intercept / mirroring is enabled by any PCC rule.


------------------------------------------------------------
PFCP-LIKE RULE MODEL (SIMPLIFIED)
------------------------------------------------------------

1) PDR (Packet Detection Rule)
- Purpose: classify packets (which flow/session they belong to) using PDI fields.
Key fields:
- pdr_id: unique string
- source_interface: {{Access|Core}}
- pdi:
  - local_teid: {{string or null}}  (optional; can be null if not specified)
  - ue_ip: {{string or "any"}}      (use "any" if not specified)
  - sdf_filter:
      src_ip, dst_ip, src_port, dst_port, protocol  (use PCC five-tuple; unspecified -> "any")
- precedence: integer (lower = higher priority)
- outer_header_removal: {{GTP-U/UDP/IP|null}} (use for uplink from Access when applicable)

2) FAR (Forwarding Action Rule)
- Purpose: define what to do with packets matched by a PDR.
Key fields:
- far_id: unique string
- apply_action: list of one or more actions chosen from {{FORW, DROP, BUFF, NOCP, DUPL}}
  - Multiple actions are allowed (e.g., ["FORW","DUPL"]).
- destination_interface: {{Access|Core|LI|null}}
- outer_header_creation: {{GTP-U/UDP/IP|null}} (use when forwarding toward Access typically)
- forwarding_policy: {{string or null}} (can be populated if traffic_steering.mode exists)
- dupl_far_id: {{string or null}}
  - If "DUPL" is in apply_action, dupl_far_id MUST point to the FAR that handles the duplicated copy.
  - If ODF is selected, dupl_far_id MUST point to an ODF FAR with destination_interface="LI".
  - If "DUPL" is NOT in apply_action, dupl_far_id MUST be null.

3) QER (QoS Enforcement Rule)
- Purpose: enforce QoS constraints if and only if explicitly present in PCC rules.
Key fields:
- qer_id: unique string
- gate_status: {{OPEN|CLOSED}}
- pdb_ms: {{number or null}}
- mbr_ul_mbps: {{number or null}}
- mbr_dl_mbps: {{number or null}}
- gbr_ul_mbps: {{number or null}}
- gbr_dl_mbps: {{number or null}}
- qfi: {{number or null}}

Rules:
- Create QER ONLY if at least one PCC qos field is non-null
  (e.g., pdb_ms, per, gfbr_mbps, mfbr_mbps, priority, 5qi).
- If PCC qos fields are all null → do NOT create QER objects.
- If PCC provides pdb_ms → copy it into QER.pdb_ms exactly (do NOT invent).
- If PCC provides throughput/GBR/MFBR → map to QER gbr_* / mbr_* as applicable.
- If PCC provides only pdb_ms (delay) without bitrate → keep bitrate fields null (do NOT invent).

4) URR (Usage Reporting Rule)
- Purpose: accounting / charging triggers.
Key fields:
- urr_id: unique string
- measurement_method: list from {{VOLUM|DURAT|EVENT}}
- reporting_triggers: list from {{QUOTA|PERIOD|START|STOP}}
- volume_threshold_bytes: {{number or null}}
- measurement_period_seconds: {{number or null}}
Rules:
- If PCC charging.enabled is true → include URR with logical defaults BUT do not invent numeric thresholds.
  Use START/STOP triggers at minimum.
- If charging.enabled is false or null → URR may be omitted or included with START/STOP only if required.

------------------------------------------------------------
MAPPING RULES FROM PCC → PFCP-LIKE CONFIG
------------------------------------------------------------

A) Direction mapping
- If PCC rule direction is "uplink":
  - Create PDR with source_interface = "Access"
  - FAR destination_interface typically = "Core" if forwarding
- If PCC rule direction is "downlink":
  - Create PDR with source_interface = "Core"
  - FAR destination_interface typically = "Access" if forwarding
- If PCC rule direction is "bidirectional":
  - Create TWO PDRs (one UL, one DL) with consistent matching fields:
    - UL: source_interface="Access", DL: source_interface="Core"
  - FARs forward accordingly.

B) Five-tuple mapping (strict)
- PCC sdf.five_tuple fields MUST be copied into PDR pdi.sdf_filter exactly.
- Any PCC five-tuple field that is "any" stays "any".
- Never output null inside sdf_filter fields.

C) Best-effort mapping
- If PCC qos fields are all null (best-effort / unspecified):
  - FAR applies FORW
  - You MUST NOT create QER objects
  - Set PDR.qer_id = null

D) Lawful intercept / mirroring (STRICT + FUNCTIONAL)
If PCC lawful_intercept.enabled is true:

- The traffic MUST be forwarded normally AND duplicated to LI.

Implementation rule:
1) Create a normal forwarding FAR for each direction (UL and/or DL) with:
   - apply_action includes "FORW"
   - destination_interface = "Core" for uplink, "Access" for downlink
   - bound_module = "ULF" (uplink) or "DLF" (downlink)

2) Ensure duplication is ACTIVE for the same matched traffic:
   - The SAME FAR referenced by the PDR MUST include "DUPL" in apply_action
     (i.e., apply_action = ["FORW","DUPL"]).

3) LI destination representation (ODF linkage REQUIRED):
   - If ODF is selected, create an LI FAR bound_module="ODF" with:
     - apply_action = ["DUPL"]
     - destination_interface = "LI"
     - outer_header_creation = null
     - dupl_far_id = null

   - The normal forwarding FAR (ULF/DLF) MUST include:
     - apply_action = ["FORW","DUPL"]
     - dupl_far_id = "<the ODF LI FAR id>"
       Example: far_ul_1.dupl_far_id = "far_li_ul_1"

VALIDATION (MANDATORY):
- If lawful_intercept.enabled is true AND ODF is selected:
  - The FAR referenced by each affected PDR MUST include "DUPL" in apply_action
  - That FAR MUST set dupl_far_id to an ODF FAR with destination_interface="LI"


E) Drop / block
- If PCC implies blocking (explicitly stated in policy output):
  - FAR apply_action includes "DROP"
  - QER gate_status can be "CLOSED" if needed
- Do NOT introduce DROP unless policy explicitly indicates it.

F) Traffic steering
- If PCC traffic_steering.mode is not null:
  - Map it into FAR.forwarding_policy (string)
- Do NOT invent routing IDs.

IMPORTANT:
- Session-specific tunnel identifiers (e.g., TEIDs, remote tunnel IPs)
  are NOT derived from intent.
- If not explicitly provided, these fields MUST be set to null and
  resolved at deployment/runtime.

Directional five-tuple handling:
- For ONLY BIDIRECTIONAL PCC rules:
  - Uplink PDR MUST use the PCC five-tuple as-is.
  - Downlink PDR MUST mirror the five-tuple by swapping src_ip and dst_ip.
- Ports and protocol MUST remain unchanged.
- This transformation is considered a directional projection, not policy reinterpretation.

G) Subscriber scope mapping:
- If PCC rule includes subscriber_scope.gpsi (non-null), copy it into PDR.pdi.subscriber_scope.gpsi
  for every PDR created from that PCC rule (UL and/or DL).
- If null, set PDR.pdi.subscriber_scope.gpsi = null.

------------------------------------------------------------
MODULE RULES (INTENDED ARCHITECTURE)
------------------------------------------------------------

These rules describe which PFCP-like rule types are relevant to each module in the modular UPF.

- ISF (Ingress Steering Function)
  Relevant rules:
  - PDR: classification / identification of flows (PDR.bound_module="ISF")

- ULF (Uplink Function)
  Relevant rules (uplink enforcement only):
  - FAR/QER/URR used for uplink behavior (bound_module="ULF")

- DLF (Downlink Function)
  Relevant rules (downlink enforcement only):
  - FAR/QER/URR used for downlink behavior (bound_module="DLF")

- ODF (Observation/Duplication Function)
  Relevant rules (lawful intercept / mirroring only):
  - FAR for LI leg representation (apply_action includes "DUPL", destination_interface="LI") when ODF is available

Binding rules:
- PDRs perform identification/classification, so:
  - Every PDR MUST have bound_module="ISF" (unless ISF not selected -> fallback + assumption).

- FAR/QER/URR perform enforcement, so:
  - For uplink PCC rules: FAR/QER/URR MUST have bound_module="ULF"
  - For downlink PCC rules: FAR/QER/URR MUST have bound_module="DLF"
  - For bidirectional PCC rules: create separate UL enforcement objects bound to ULF and DL enforcement objects bound to DLF.

- Lawful intercept duplication:
  - If lawful_intercept is enabled AND ODF exists:
    - An ODF FAR MUST exist for the LI leg (destination_interface="LI")
    - The forwarding FAR (ULF/DLF) MUST reference it via dupl_far_id

------------------------------------------------------------
OUTPUT REQUIREMENTS (STRICT)
------------------------------------------------------------
Return ONLY valid JSON.

- rationale: REQUIRED non-empty list of short factual statements explaining decisions.
- assumptions: list of any missing info or module mismatches (can be empty).
- confidence: 0-100 reflecting certainty that config correctly implements PCC with selected modules.

OUTPUT SCHEMA:
{{
  "pfcp_config": {{
    "pdr_list": [
      {{
        "pdr_id": "<string>",
        "bound_module": "<ISF|ULF|DLF|ODF>",
        "source_interface": "<Access|Core>",
        "pdi": {{
          "local_teid": "<string|null>",
          "ue_ip": "<string>",
          "subscriber_scope": {{
            "gpsi": "<string|null>"
          }},
          "sdf_filter": {{
            "src_ip": "<string>",
            "dst_ip": "<string>",
            "src_port": "<string|number>",
            "dst_port": "<string|number>",
            "protocol": "<string>"
          }}
        }},
        "precedence": "<number>",
        "outer_header_removal": "<GTP-U/UDP/IP|null>",
        "far_id": "<string>",
        "qer_id": "<string|null>",
        "urr_id": "<string|null>"
      }}
    ],
    "far_list": [
      {{
        "far_id": "<string>",
        "bound_module": "<ISF|ULF|DLF|ODF>",
        "apply_action": ["<FORW|DROP|BUFF|NOCP|DUPL>", "<FORW|DROP|BUFF|NOCP|DUPL>"],
        "destination_interface": "<Access|Core|LI|null>",
        "outer_header_creation": "<GTP-U/UDP/IP|null>",
        "forwarding_policy": "<string|null>",
        "dupl_far_id": "<string|null>",
        "li_collector_ip": "<string|null>"
      }}
    ],
    "qer_list": [
      {{
        "qer_id": "<string>",
        "bound_module": "<ISF|ULF|DLF|ODF>",
        "gate_status": "<OPEN|CLOSED>",
        "pdb_ms": "<number|null>",
        "mbr_ul_mbps": "<number|null>",
        "mbr_dl_mbps": "<number|null>",
        "gbr_ul_mbps": "<number|null>",
        "gbr_dl_mbps": "<number|null>",
        "qfi": "<number|null>"
      }}
    ],
    "urr_list": [
      {{
        "urr_id": "<string>",
        "bound_module": "<ISF|ULF|DLF|ODF>",
        "measurement_method": ["<VOLUM|DURAT|EVENT>"],
        "reporting_triggers": ["<QUOTA|PERIOD|START|STOP>"],
        "volume_threshold_bytes": "<number|null>",
        "measurement_period_seconds": "<number|null>"
      }}
    ]
  }},
  "rationale": ["<string>"],
  "assumptions": ["<string>"],
  "confidence": "<number 0-100>"
}}



Return ONLY valid JSON.

"""

llm_configurator = ChatOpenAI(model="gpt-4.1", temperature=0.0)
config_prompt = ChatPromptTemplate.from_messages([("system", CONFIGURATOR_PROMPT)])



def _extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()

    
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1]).strip()
        else:
            text = "\n".join(lines[1:]).strip()


    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass


    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch == "{":
            try:
                obj, _end = decoder.raw_decode(text[i:])
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                continue

    raise ValueError("Configurator did not return a valid JSON object.")



allowed_modules = {"ISF", "ULF", "DLF", "ODF"}

def _require_bound_module(obj: Dict[str, Any], obj_name: str, id_key: str) -> None:
    bm = obj.get("bound_module")
    if not isinstance(bm, str) or not bm.strip():
        raise ValueError(f"{obj_name} '{obj.get(id_key)}' must include non-empty 'bound_module'.")
    if bm not in allowed_modules:
        raise ValueError(f"{obj_name} '{obj.get(id_key)}' has invalid bound_module='{bm}'.")


def _validate_configurator_output_min(out: Dict[str, Any]) -> None:
    if not isinstance(out, dict):
        raise ValueError("Configurator output must be a JSON object.")

    cfg = out.get("pfcp_config")
    if not isinstance(cfg, dict):
        raise ValueError("Configurator output must include 'pfcp_config' object.")

    for k in ("pdr_list", "far_list", "qer_list", "urr_list"):
        if k not in cfg or not isinstance(cfg[k], list):
            raise ValueError(f"Configurator output must include '{k}' as a list.")

  
    rationale = out.get("rationale")
    if not isinstance(rationale, list) or len(rationale) == 0:
        raise ValueError("Configurator output must include non-empty 'rationale' list.")
    if not all(isinstance(x, str) and x.strip() for x in rationale):
        raise ValueError("Each rationale entry must be a non-empty string.")

    assumptions = out.get("assumptions")
    if not isinstance(assumptions, list):
        raise ValueError("Configurator output must include 'assumptions' list.")
    if len(assumptions) > 0 and not all(isinstance(x, str) and x.strip() for x in assumptions):
        raise ValueError("Each assumptions entry must be a non-empty string.")

    conf = out.get("confidence")
    if not isinstance(conf, (int, float)) or conf < 0 or conf > 100:
        raise ValueError("Configurator output must include 'confidence' between 0 and 100.")


    for pdr in cfg["pdr_list"]:
        if isinstance(pdr, dict):
            _require_bound_module(pdr, "PDR", "pdr_id")
    for far in cfg["far_list"]:
        if isinstance(far, dict):
            _require_bound_module(far, "FAR", "far_id")
    for qer in cfg["qer_list"]:
        if isinstance(qer, dict):
            _require_bound_module(qer, "QER", "qer_id")
    for urr in cfg["urr_list"]:
        if isinstance(urr, dict):
            _require_bound_module(urr, "URR", "urr_id")

    far_ids = {f.get("far_id") for f in cfg["far_list"] if isinstance(f, dict)}
    for p in cfg["pdr_list"]:
        if isinstance(p, dict):
            fid = p.get("far_id")
            if fid is None or fid not in far_ids:
                raise ValueError("Each PDR must reference an existing FAR via 'far_id'.")


def run_configurator_agent(
    selected_modules: List[Dict[str, Any]],
    pcc_rules: List[Dict[str, Any]],
) -> Dict[str, Any]:
    messages = config_prompt.format_messages(
        selected_modules_json=json.dumps(selected_modules, indent=2),
        pcc_rules_json=json.dumps(pcc_rules, indent=2),
    )

    msg = llm_configurator.invoke(messages)
    raw = msg.content
    out = _extract_json_object(raw)

    
    cfg = out.get("pfcp_config", {})
    if isinstance(cfg, dict):
        for far in cfg.get("far_list", []):
            if not isinstance(far, dict):
                continue
            ohc = far.get("outer_header_creation")
            if isinstance(ohc, dict):
                far["outer_header_creation"] = None

    _validate_configurator_output_min(out)

    out["usage"] = extract_usage_from_ai_message(msg)
    return out
