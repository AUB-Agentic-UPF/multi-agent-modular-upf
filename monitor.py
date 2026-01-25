# monitor.py
from __future__ import annotations

import json
from typing import Any, Dict, List

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from usage_utils import extract_usage_from_ai_message


MONITOR_PROMPT = """
You are the Monitor Agent in an agentic 5G core system (SMF scope).

ROLE
- You run AFTER the orchestrator attempts deployment.
- You are INDEPENDENT from the Executor.
- You evaluate NFRs using the provided PCC rules LIST.
- You may receive observed_state.nfr_retry_count to track how many NFR-driven retries were already attempted.

FLOW RULES (MUST FOLLOW)
1) If orchestrator_result.status == "rejected":
   - status = "failed"
   - checks_summary.orchestrator_apply = "fail"
   - next_hop = "Operator"
   - Do NOT evaluate anything else.

2) Else (orchestrator accepted/applied):
   - Let r = observed_state.nfr_retry_count (if missing, assume r = 0).
   - First, check non-NFR failures:
       • If modules_ready == False OR chain_connected == False OR rules_installed == False:
           - status = "failed"
           - nfr_compliance = "skip" (do not claim NFR violation)
           - next_hop = "Operator"
   - If no non-NFR failures, check NFRs:
       • If ANY PCC rule NFR is violated:
           - status = "nfr_violated"
           - nfr_compliance = "fail"
           - If r < 2 → next_hop = "CloudOrchestrator"
           - If r >= 2 → next_hop = "Operator"
       • If NO NFR is violated:
           - status = "passed"
           - nfr_compliance = "pass"
           - next_hop = "Done"

INPUTS
policy (PCC rules LIST):
{pcc_rules_json}

orchestrator_result:
{orchestrator_result_json}

observed_state:
{observed_state_json}

NFR CHECKING (PER PCC RULE)
For each PCC rule, compare qos fields against telemetry for that rule’s direction:

IF qos.pdb_ms IS NOT NULL:
- Requirement: latency_ms <= qos.pdb_ms
- Use:
  • uplink: telemetry.uplink_latency_ms
  • downlink: telemetry.downlink_latency_ms
  • bidirectional: check both uplink_latency_ms and downlink_latency_ms if present
  • fallback: telemetry.end_to_end_latency_ms

IF qos.per IS NOT NULL:
- Requirement: packet_error_rate <= qos.per
- Use:
  • uplink: telemetry.uplink_packet_error_rate
  • downlink: telemetry.downlink_packet_error_rate
  • bidirectional: check both if present
  • fallback: telemetry.packet_error_rate

IF qos.gfbr_mbps IS NOT NULL:
- TREAT GFBR AS A MINIMUM GUARANTEED THROUGHPUT.
- Requirement: throughput_mbps >= qos.gfbr_mbps
- It is NOT a violation if observed throughput > qos.gfbr_mbps.
- Use:
  • uplink: telemetry.uplink_throughput_mbps
  • downlink: telemetry.downlink_throughput_mbps
  • bidirectional: check both if present
  • fallback: telemetry.achieved_throughput_mbps

DIRECTION SELECTION
- If rule.direction == "uplink": use uplink telemetry if present, else fallback to generic.
- If rule.direction == "downlink": use downlink telemetry if present, else fallback to generic.
- If rule.direction == "bidirectional":
  • Prefer directional metrics (UL and DL).
  • If only generic metrics exist, check the generic value once.

EVIDENCE MISSING RULE
- If a required telemetry field for a given qos metric is missing:
  • Do NOT mark that metric as violated.
- If ALL relevant telemetry for ALL qos metrics is missing:
  • Set checks_summary.nfr_compliance = "skip".
  • Do not set status = "nfr_violated" based on missing data alone.

RATIONALE
- Short, factual explanation of why the final status is passed / failed / nfr_violated.
- Do NOT restate the entire policy or telemetry.
- Do NOT reveal your internal reasoning steps.
- The rationale field MUST NOT be empty.

ASSUMPTIONS
- A list of explicit assumptions ONLY if information was missing, ambiguous, or unverifiable.
- If no assumptions are needed, use an empty list [].

CONFIDENCE
- A number in [0, 100] indicating how certain you are that the result correctly reflects the PCC rules and observed state.

OUTPUT REQUIREMENTS
- Return ONLY valid JSON (no surrounding text).
- orchestrator_feedback MUST be null unless status == "nfr_violated".

OUTPUT SCHEMA
{{
  "status": "<passed|nfr_violated|failed>",
  "checks_summary": [
    {{"name": "orchestrator_apply", "result": "<pass|fail>"}},
    {{"name": "modules_ready", "result": "<pass|fail|skip>"}},
    {{"name": "chain_connected", "result": "<pass|fail|skip>"}},
    {{"name": "rules_installed", "result": "<pass|fail|skip>"}},
    {{"name": "nfr_compliance", "result": "<pass|fail|skip>"}}
  ],
  "next_hop": "<Done|Operator|CloudOrchestrator>",
  "details": {{}},
  "orchestrator_feedback": null,
  "rationale": "<string>",
  "assumptions": ["<string>"],
  "confidence": <number_0_to_100>
}}

IF AND ONLY IF status == "nfr_violated", set orchestrator_feedback:
{{
  "issue": "NFR_VIOLATION",
  "violated_metrics": [
    {{
      "rule_id": "<string>",
      "direction": "<string>",
      "metric": "<string>",
      "expected": <number>,
      "observed": <number>
    }}
  ],
  "suggested_action": "<string>"
}}


Return ONLY JSON.
"""



llm_monitor = ChatOpenAI(model="gpt-4.1", temperature=0.0)
monitor_prompt = ChatPromptTemplate.from_messages([("system", MONITOR_PROMPT)])


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
        raise ValueError("Monitor did not return a JSON object.")
    return json.loads(text[start:end + 1])


def run_monitor_agent(
    pcc_rules: List[Dict[str, Any]],
    orchestrator_result: Dict[str, Any],
    observed_state: Dict[str, Any],
) -> Dict[str, Any]:
    messages = monitor_prompt.format_messages(
        pcc_rules_json=json.dumps(pcc_rules, indent=2),
        orchestrator_result_json=json.dumps(orchestrator_result, indent=2),
        observed_state_json=json.dumps(observed_state, indent=2),
    )

    msg = llm_monitor.invoke(messages)
    raw = msg.content
    out = _extract_json_object(raw)


    out["usage"] = extract_usage_from_ai_message(msg)
    return out
