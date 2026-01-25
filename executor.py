# executor.py
from __future__ import annotations

import json
from typing import Any, Dict, List

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from usage_utils import extract_usage_from_ai_message
import uuid

EXECUTOR_PROMPT = """
You are the Executor Agent under SMF scope in an agentic 5G core system.

UPDATED FLOW (IMPORTANT)
- You generate a deployment plan artifact (orchestrator-agnostic IR) to be sent to a cloud orchestrator.
- The deployment plan is NOT a real Kubernetes/Helm manifest and MUST NOT include vendor-specific fields.
- You do NOT produce monitor instructions.
- You do NOT talk to the Monitor and you do NOT control retries/escalations.

INPUTS (AUTHORITATIVE)
PCC rules (LIST) from Policy Creator:
{pcc_rules_json}

Selected modules (LIST) from Selector:
{selected_modules_json}

PFCP-like rules from Configurator:
{pfcp_rules_json}



RULES (IMPORTANT)
- ONLY deploy modules that exist in selected_modules, in the same order as provided.
- Do NOT add modules, do NOT remove modules, do NOT reorder modules.
- Use the single field "profile" exactly as provided by Selector.
- Do NOT invent or modify profile values.


WIRING RULE (BRANCHING)
- If ISF, ULF, and DLF are present, wire:
    ISF -> ULF
    ISF -> DLF
  (ISF acts as classifier/steering into separate uplink/downlink pipelines.)
- Otherwise, wire sequentially between consecutive modules in the provided order.

RULE MAPPING
- pfcp_rules_json contains an object "pfcp_config" with:
  pdr_list, far_list, qer_list, urr_list.
- For each module present in the chain, populate apply_rules[module] with the IDs of objects whose bound_module equals that module:
  - pdr_ids: from pdr_list[*].pdr_id
  - far_ids: from far_list[*].far_id
  - qer_ids: from qer_list[*].qer_id
  - urr_ids: from urr_list[*].urr_id
- The deployment artifact MUST reference ONLY IDs that exist in the configurator output.
- If any PFCP object lacks bound_module, infer conservatively and add an assumption.

ORCHESTRATOR TARGET
- This deployment plan is orchestrator-agnostic.
- cluster and namespace are out of scope and MUST be set to null.


OUTPUT METADATA DEFINITIONS (IMPORTANT)
rationale:
- REQUIRED non-empty list of short, factual statements explaining key deployment decisions.
- Do NOT restate inputs verbatim.
- Do NOT include step-by-step reasoning.

assumptions:
- List any missing information or out-of-scope orchestrator details.
- If none, return an empty list.

confidence:
- 0-100 reflecting how certain you are that the deployment_plan correctly represents the given inputs.

OUTPUT REQUIREMENTS
- Output ONLY valid JSON (no markdown, no extra text).
- target_system must be "cloud_orchestrator".

OUTPUT SCHEMA (STRICT JSON)
{{
  "status": "ready_to_deploy",
  "deployment_plan": {{
    "target_system": "cloud_orchestrator",
    "chain": [
      {{
        "module": "<ISF|ULF|DLF|ODF>",
        "profile": "<ISF_sw|ISF_hw|ULF_sw|ULF_hw|DLF_sw|DLF_hw|ODF_sw|ODF_hw>"
      }}
    ],
    "wiring": [
      {{
        "from": "<string>",
        "to": "<string>"
      }}
    ],
    "apply_rules": {{
      "<module_name>": {{
        "pdr_ids": ["<string>"],
        "far_ids": ["<string>"],
        "qer_ids": ["<string>"],
        "urr_ids": ["<string>"]
      }}
    }},
    "orchestrator_target": {{
      "cluster": "<string|null>",
      "namespace": "<string|null>"
    }}
  }},
  "rationale": ["<string>"],
  "assumptions": ["<string>"],
  "confidence": "<number 0-100>"
}}

"""



llm_executor = ChatOpenAI(model="gpt-4.1", temperature=0.0)
exec_prompt = ChatPromptTemplate.from_messages([("system", EXECUTOR_PROMPT)])


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
        raise ValueError("Executor did not return a JSON object.")
    return json.loads(text[start:end + 1])



def _validate_executor_output(
    out: Dict[str, Any],
    selected_modules: List[Dict[str, Any]],
    pfcp_rules: Dict[str, Any],
) -> None:
    if not isinstance(out, dict):
        raise ValueError("Executor output must be a JSON object.")

    if out.get("status") != "ready_to_deploy":
        raise ValueError("Executor output status must be 'ready_to_deploy'.")

    dp = out.get("deployment_plan")
    if not isinstance(dp, dict):
        raise ValueError("Executor output must include 'deployment_plan' object.")

    if dp.get("target_system") != "cloud_orchestrator":
        raise ValueError("deployment_plan.target_system must be 'cloud_orchestrator'.")

    
    chain = dp.get("chain")
    if not isinstance(chain, list) or len(chain) == 0:
        raise ValueError("deployment_plan.chain must be a non-empty list.")

    if len(chain) != len(selected_modules):
        raise ValueError("deployment_plan.chain length must match selected_modules length exactly.")

    for i, (c, s) in enumerate(zip(chain, selected_modules)):
        if not isinstance(c, dict) or not isinstance(s, dict):
            raise ValueError("chain and selected_modules entries must be objects.")

        cm = c.get("module")
        cp = c.get("profile")
        sm = s.get("module")
        sp = s.get("profile")

        if cm != sm:
            raise ValueError(f"chain[{i}].module='{cm}' does not match selected_modules[{i}].module='{sm}'.")
        if cp != sp:
            raise ValueError(f"chain[{i}].profile='{cp}' does not match selected_modules[{i}].profile='{sp}'.")

    
    ot = dp.get("orchestrator_target")
    if not isinstance(ot, dict):
        raise ValueError("deployment_plan.orchestrator_target must be an object.")
    if ot.get("cluster") is not None or ot.get("namespace") is not None:
        raise ValueError("orchestrator_target.cluster and orchestrator_target.namespace must be null.")

    wiring = dp.get("wiring")
    if not isinstance(wiring, list):
        raise ValueError("deployment_plan.wiring must be a list.")

    modules = [m.get("module") for m in chain if isinstance(m, dict)]
    has_branch = ("ISF" in modules and "ULF" in modules and "DLF" in modules)

    if has_branch:
        required = {("ISF", "ULF"), ("ISF", "DLF")}
        edges = {(e.get("from"), e.get("to")) for e in wiring if isinstance(e, dict)}
        if not required.issubset(edges):
            raise ValueError("Branching wiring must include edges ISF->ULF and ISF->DLF when ISF+ULF+DLF are present.")
    else:
        expected = []
        for a, b in zip(modules, modules[1:]):
            expected.append({"from": a, "to": b})

        if len(wiring) != len(expected):
            raise ValueError("Sequential wiring length is incorrect.")

        for i, (we, exp) in enumerate(zip(wiring, expected)):
            if we.get("from") != exp["from"] or we.get("to") != exp["to"]:
                raise ValueError(f"Sequential wiring mismatch at index {i}: got {we}, expected {exp}.")

    ar = dp.get("apply_rules")
    if not isinstance(ar, dict):
        raise ValueError("deployment_plan.apply_rules must be an object.")

    cfg = pfcp_rules.get("pfcp_config")
    if not isinstance(cfg, dict):
        raise ValueError("pfcp_rules must contain pfcp_config object.")

    pdr_ids = {x.get("pdr_id") for x in cfg.get("pdr_list", []) if isinstance(x, dict)}
    far_ids = {x.get("far_id") for x in cfg.get("far_list", []) if isinstance(x, dict)}
    qer_ids = {x.get("qer_id") for x in cfg.get("qer_list", []) if isinstance(x, dict)}
    urr_ids = {x.get("urr_id") for x in cfg.get("urr_list", []) if isinstance(x, dict)}

    for mod in modules:
        entry = ar.get(mod)
        if not isinstance(entry, dict):
            raise ValueError(f"apply_rules must include an object for module '{mod}'.")

        for key, allowed in [
            ("pdr_ids", pdr_ids),
            ("far_ids", far_ids),
            ("qer_ids", qer_ids),
            ("urr_ids", urr_ids),
        ]:
            ids = entry.get(key)
            if not isinstance(ids, list):
                raise ValueError(f"apply_rules['{mod}'].{key} must be a list.")

            for _id in ids:
                if not isinstance(_id, str):
                    raise ValueError(f"apply_rules['{mod}'].{key} entries must be strings.")
                if _id not in allowed:
                    raise ValueError(f"apply_rules['{mod}'].{key} references unknown id '{_id}'.")

def run_executor_agent(
    pcc_rules: List[Dict[str, Any]],
    selected_modules: List[Dict[str, Any]],
    pfcp_rules: Dict[str, Any],
) -> Dict[str, Any]:
    messages = exec_prompt.format_messages(
        pcc_rules_json=json.dumps(pcc_rules, indent=2),
        selected_modules_json=json.dumps(selected_modules, indent=2),
        pfcp_rules_json=json.dumps(pfcp_rules, indent=2),
    )

    msg = llm_executor.invoke(messages)
    raw = msg.content
    out = _extract_json_object(raw)

    _validate_executor_output(out, selected_modules=selected_modules, pfcp_rules=pfcp_rules)

    out["usage"] = extract_usage_from_ai_message(msg)
    return out
