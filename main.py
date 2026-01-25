# main.py
from __future__ import annotations

from typing import Any, Dict, List
import json
import time

from IHF import run_ihf
from policy_creator import run_policy_creator_agent
from selector import run_selector_agent
from configurator import run_configurator_agent
from executor import run_executor_agent

CONF_THRESHOLD = 65


def _usage_dict() -> Dict[str, int]:
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def _get_usage(out: Dict[str, Any]) -> Dict[str, int]:
    u = out.get("usage", {})
    if not isinstance(u, dict):
        return _usage_dict()

    def to_int(x: Any) -> int:
        try:
            return int(x)
        except (TypeError, ValueError):
            return 0

    pt = to_int(u.get("prompt_tokens"))
    ct = to_int(u.get("completion_tokens"))
    tt = to_int(u.get("total_tokens"))
    if tt == 0:
        tt = pt + ct
    return {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": tt}


def _add_usage(total: Dict[str, int], add: Dict[str, int]) -> None:
    total["prompt_tokens"] += add.get("prompt_tokens", 0)
    total["completion_tokens"] += add.get("completion_tokens", 0)
    total["total_tokens"] += add.get("total_tokens", 0)



def pretty_print(title: str, data: Dict[str, Any]) -> None:
    print(f"\n[{title}]")
    print(json.dumps(data, indent=2))


def validate_confidence_or_stop(agent_name: str, out: Dict[str, Any], threshold: int = CONF_THRESHOLD) -> None:
    conf_raw = out.get("confidence", 0)
    try:
        conf = float(conf_raw)
    except (TypeError, ValueError):
        conf = 0.0

    if 0 <= conf <= 1:
        conf *= 100

    conf = max(0.0, min(100.0, conf))
    if conf >= threshold:
        return

    print(f"\nWARNING: Low confidence from {agent_name}: {conf:.1f}")
    if isinstance(out.get("assumptions"), list) and out["assumptions"]:
        print("Assumptions:")
        for a in out["assumptions"]:
            print(f" - {a}")

    choice = input("Continue anyway? (y/n): ").strip().lower()
    if choice != "y":
        raise RuntimeError(f"Stopped by operator due to low confidence in {agent_name}.")


def banner(text: str) -> None:
    print("\n" + "=" * 90)
    print(text)
    print("=" * 90 + "\n")



# Orchestrator (fixed accept for this evaluation)
def orchestrator_apply(_deployment_plan: Dict[str, Any]) -> Dict[str, Any]:
    return {"status": "applied", "reason": None}



INTENTS: List[Dict[str, Any]] = [
     {
         "id": "intent_1",
         "text": "Provision a basic PDU session that provides bidirectional Internet connectivity (uplink and downlink) with no explicit QoS guarantees, operating under a best-effort service model.",
     },
     {
         "id": "intent_2",
         "text": "Create a PDU session for Netflix video streaming traffic, identified by destination IP address 203.0.113.80. Enforce a minimum downlink throughput of ≥ 5 Mbps and a packet delay budget of ≤ 100 ms for downlink traffic, while treating uplink traffic as best-effort.",
     },
     {
         "id": "intent_3",
         "text": "Create a PDU session for Netflix video streaming traffic, identified by destination IP address 203.0.113.80, applying appropriate QoS requirements for non-conversational video services.",
     },
     {
         "id": "intent_4",
         "text": "Create a PDU session for Netflix video streaming traffic, identified by destination IP address 203.0.113.80, with QoS guarantees.",
     },
     {
         "id": "intent_5",
         "text": "Establish a PDU session for the user identified by GPSI +012111111 to support a real-time gaming application, enforcing appropriate QoS requirements, including an uplink packet delay budget of 30 ms.",
     },

    {
        "id": "intent_6",
        "text": "Establish a PDU session for the user identified by GPSI +012111111 to support a real-time gaming application. Enforce appropriate QoS requirements, including an uplink packet delay budget of 30 ms. Enable lawful intercept by duplicating only the downlink traffic to collector IP 198.51.100.10.",
    },
]


# Evaluation of intents (without monitor agent)
def run_intent(intent_obj: Dict[str, Any]) -> Dict[str, Any]:
    operator_intent = intent_obj["text"]
    intent_id = intent_obj["id"]

    banner(f"INTENT: {intent_id}")

    print("\n[OPERATOR INTENT]")
    print(operator_intent)


    t_start = time.perf_counter()
    usage_total = _usage_dict()

    ihf_out = run_ihf(operator_intent)
    _add_usage(usage_total, _get_usage(ihf_out))
    pretty_print("IHF OUTPUT", ihf_out)
    validate_confidence_or_stop("IHF", ihf_out)

    pc_out = run_policy_creator_agent(ihf_output=ihf_out)
    _add_usage(usage_total, _get_usage(pc_out))
    pretty_print("POLICY CREATOR OUTPUT", pc_out)
    validate_confidence_or_stop("Policy Creator", pc_out)

    pcc_rules = pc_out.get("pcc_rules", [])
    if not isinstance(pcc_rules, list):
        pcc_rules = []

    sel_out = run_selector_agent(pcc_rules=pcc_rules)
    _add_usage(usage_total, _get_usage(sel_out))
    pretty_print("SELECTOR OUTPUT", sel_out)
    validate_confidence_or_stop("Selector", sel_out)

    selected_modules = sel_out.get("selected_modules", [])
    if not isinstance(selected_modules, list):
        selected_modules = []

    cfg_out = run_configurator_agent(
        selected_modules=selected_modules,
        pcc_rules=pcc_rules,
    )
    _add_usage(usage_total, _get_usage(cfg_out))
    pretty_print("CONFIGURATOR OUTPUT", cfg_out)
    validate_confidence_or_stop("Configurator", cfg_out)

    exe_out = run_executor_agent(
        pcc_rules=pcc_rules,
        selected_modules=selected_modules,
        pfcp_rules=cfg_out,
    )
    _add_usage(usage_total, _get_usage(exe_out))
    pretty_print("EXECUTOR OUTPUT", exe_out)
    validate_confidence_or_stop("Executor", exe_out)

    deployment_plan = exe_out.get("deployment_plan", {})
    orch_out = orchestrator_apply(deployment_plan)
    pretty_print("ORCHESTRATOR RESULT (default)", orch_out)

    total_time_s = time.perf_counter() - t_start

    print(
        f"\n[UPSTREAM METRICS] intent={intent_id} | "
        f"time_s={total_time_s:.3f} | "
        f"tokens_total={usage_total['total_tokens']} "
        f"(prompt={usage_total['prompt_tokens']}, completion={usage_total['completion_tokens']}) | "
        f"includes=IHF+PolicyCreator+Selector+Configurator+Executor"
    )


    return {
        "intent_id": intent_id,
        "intent_text": operator_intent,
        "upstream_time_seconds": total_time_s,
        "upstream_token_usage": usage_total,
        "ihf": ihf_out,
        "policy_creator": pc_out,
        "selector": sel_out,
        "configurator": cfg_out,
        "executor": exe_out,
        "orchestrator_result": orch_out,
    }



if __name__ == "__main__":
    all_results: List[Dict[str, Any]] = []
    for intent_obj in INTENTS:
        out = run_intent(intent_obj)
        all_results.append(out)

    banner("DONE")
