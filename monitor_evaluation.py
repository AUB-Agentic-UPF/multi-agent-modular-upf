# monitor_only_test.py
from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from monitor import run_monitor_agent  



INTENT_TEXTS: Dict[str, str] = {
    "intent_1": "Provision a basic PDU session that provides bidirectional Internet connectivity (uplink and downlink) with no explicit QoS guarantees, operating under a best-effort service model.",
    "intent_2": "Create a PDU session for Netflix video streaming traffic, identified by destination IP address 203.0.113.80. Enforce a minimum downlink throughput of ≥ 5 Mbps and a packet delay budget of ≤ 100 ms for downlink traffic, while treating uplink traffic as best-effort.",
    "intent_3": "Create a PDU session for Netflix video streaming traffic, identified by destination IP address 203.0.113.80, applying appropriate QoS requirements for non-conversational video services.",
    "intent_4": "Create a PDU session for Netflix video streaming traffic, identified by destination IP address 203.0.113.80, with QoS guarantees.",
    "intent_5": "Establish a PDU session for the user identified by GPSI +012111111 to support a real-time gaming application, enforcing appropriate QoS requirements, including an uplink packet delay budget of 30 ms.",
    "intent_6": "Establish a PDU session for the user identified by GPSI +012111111 to support a real-time gaming application. Enforce appropriate QoS requirements, including an uplink packet delay budget of 30 ms. Enable lawful intercept by duplicating only the downlink traffic to collector IP 198.51.100.10.",
}



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


def banner(text: str) -> None:
    print("\n" + "=" * 90)
    print(text)
    print("=" * 90 + "\n")


def pretty_print(title: str, data: Dict[str, Any]) -> None:
    print(f"\n[{title}]")
    print(json.dumps(data, indent=2))



def _call_monitor(
    *,
    pcc_rules: List[Dict[str, Any]],
    orchestrator_result: Dict[str, Any],
    observed_state: Dict[str, Any],
) -> Dict[str, Any]:
    try:
        return run_monitor_agent(
            pcc_rules=pcc_rules,
            orchestrator_result=orchestrator_result,
            observed_state=observed_state,
        )
    except TypeError:
        return run_monitor_agent(
            pcc_rules=pcc_rules,
            session_context={},  
            orchestrator_result=orchestrator_result,
            observed_state=observed_state,
        )



# Ground-truth PCC policies 
def ground_truth_policies() -> List[Dict[str, Any]]:
    policies: List[Dict[str, Any]] = []

    def _rule(
        *,
        rule_id: str,
        direction: str,
        application: Optional[str],
        src_ip: str,
        dst_ip: str,
        gpsi: Optional[str],
        qos_5qi: Optional[int],
        qos_priority: Optional[int],
        qos_pdb_ms: Optional[int],
        qos_per: Optional[float],
        qos_gfbr_mbps: Optional[float],
        li_enabled: bool,
        li_dir: Optional[str],
    ) -> Dict[str, Any]:
        return {
            "rule_id": rule_id,
            "direction": direction,
            "subscriber_scope": {"gpsi": gpsi},
            "sdf": {
                "application": application,
                "five_tuple": {
                    "src_ip": src_ip,
                    "dst_ip": dst_ip,
                    "src_port": "any",
                    "dst_port": "any",
                    "protocol": "any",
                }
            },
            "qos": {
                "5qi": qos_5qi,
                "priority": qos_priority,
                "pdb_ms": qos_pdb_ms,
                "per": qos_per,
                "gfbr_mbps": qos_gfbr_mbps,
                "mfbr_mbps": None,
            },
            "charging": {"enabled": None},
            "traffic_steering": {"mode": None},
            "lawful_intercept": {
                "enabled": li_enabled,
                "collector": None,
                "mirror_direction": li_dir,
            },
        }

 
    policies.append({
        "policy_id": "intent_1",
        "pcc_policy": {
            "pcc_rules": [
                _rule(
                    rule_id="pcc_main",
                    direction="bidirectional",
                    application=None,
                    src_ip="any",
                    dst_ip="any",
                    gpsi=None,
                    qos_5qi=None,
                    qos_priority=None,
                    qos_pdb_ms=None,
                    qos_per=None,
                    qos_gfbr_mbps=None,
                    li_enabled=False,
                    li_dir=None,
                )
            ],
            "rationale": ["Best-effort bidirectional Internet access; QoS fields null."],
            "assumptions": [],
            "confidence": 100,
        },
    })

    
    policies.append({
        "policy_id": "intent_2",
        "pcc_policy": {
            "pcc_rules": [
                _rule(
                    rule_id="pcc_ul",
                    direction="uplink",
                    application="Netflix",
                    src_ip="any",
                    dst_ip="203.0.113.80",
                    gpsi=None,
                    qos_5qi=None,
                    qos_priority=None,
                    qos_pdb_ms=None,
                    qos_per=None,
                    qos_gfbr_mbps=None,
                    li_enabled=False,
                    li_dir=None,
                ),
                _rule(
                    rule_id="pcc_dl",
                    direction="downlink",
                    application="Netflix",
                    src_ip="203.0.113.80",
                    dst_ip="any",
                    gpsi=None,
                    qos_5qi=None,
                    qos_priority=None,
                    qos_pdb_ms=100,
                    qos_per=None,
                    qos_gfbr_mbps=5.0,
                    li_enabled=False,
                    li_dir=None,
                ),
            ],
            "rationale": ["Downlink has explicit throughput+delay; uplink is best-effort."],
            "assumptions": [],
            "confidence": 100,
        },
    })

    
    policies.append({
        "policy_id": "intent_3",
        "pcc_policy": {
            "pcc_rules": [
                _rule(
                    rule_id="pcc_ul",
                    direction="uplink",
                    application="Netflix",
                    src_ip="any",
                    dst_ip="203.0.113.80",
                    gpsi=None,
                    qos_5qi=4,
                    qos_priority=50,
                    qos_pdb_ms=300,
                    qos_per=1e-6,
                    qos_gfbr_mbps=None,
                    li_enabled=False,
                    li_dir=None,
                ),
                _rule(
                    rule_id="pcc_dl",
                    direction="downlink",
                    application="Netflix",
                    src_ip="203.0.113.80",
                    dst_ip="any",
                    gpsi=None,
                    qos_5qi=4,
                    qos_priority=50,
                    qos_pdb_ms=300,
                    qos_per=1e-6,
                    qos_gfbr_mbps=None,
                    li_enabled=False,
                    li_dir=None,
                ),
            ],
            "rationale": ["QoS requested; applied mapping for non-conversational video (5QI=4)."],
            "assumptions": [],
            "confidence": 100,
        },
    })

    
    policies.append({
        "policy_id": "intent_4",
        "pcc_policy": {
            "pcc_rules": [
                _rule(
                    rule_id="pcc_ul",
                    direction="uplink",
                    application="Netflix",
                    src_ip="any",
                    dst_ip="203.0.113.80",
                    gpsi=None,
                    qos_5qi=4,
                    qos_priority=50,
                    qos_pdb_ms=300,
                    qos_per=1e-6,
                    qos_gfbr_mbps=None,
                    li_enabled=False,
                    li_dir=None,
                ),
                _rule(
                    rule_id="pcc_dl",
                    direction="downlink",
                    application="Netflix",
                    src_ip="203.0.113.80",
                    dst_ip="any",
                    gpsi=None,
                    qos_5qi=4,
                    qos_priority=50,
                    qos_pdb_ms=300,
                    qos_per=1e-6,
                    qos_gfbr_mbps=None,
                    li_enabled=False,
                    li_dir=None,
                ),
            ],
            "rationale": ["QoS guarantees requested; applied mapping for non-conversational video (5QI=4)."],
            "assumptions": [],
            "confidence": 100,
        },
    })

    
    policies.append({
        "policy_id": "intent_5",
        "pcc_policy": {
            "pcc_rules": [
                _rule(
                    rule_id="pcc_ul",
                    direction="uplink",
                    application="Real-time gaming",
                    src_ip="any",
                    dst_ip="any",
                    gpsi="+012111111",
                    qos_5qi=3,
                    qos_priority=30,
                    qos_pdb_ms=30,
                    qos_per=1e-3,
                    qos_gfbr_mbps=None,
                    li_enabled=False,
                    li_dir=None,
                ),
                _rule(
                    rule_id="pcc_dl",
                    direction="downlink",
                    application="Real-time gaming",
                    src_ip="any",
                    dst_ip="any",
                    gpsi="+012111111",
                    qos_5qi=3,
                    qos_priority=30,
                    qos_pdb_ms=50,
                    qos_per=1e-3,
                    qos_gfbr_mbps=None,
                    li_enabled=False,
                    li_dir=None,
                ),
            ],
            "rationale": ["QoS requested; UL PDB explicitly 30ms; remaining QoS from gaming mapping (5QI=3)."],
            "assumptions": [],
            "confidence": 100,
        },
    })

    policies.append({
        "policy_id": "intent_6",
        "pcc_policy": {
            "pcc_rules": [
                _rule(
                    rule_id="pcc_ul",
                    direction="uplink",
                    application="Real-time gaming",
                    src_ip="any",
                    dst_ip="any",
                    gpsi="+012111111",
                    qos_5qi=3,
                    qos_priority=30,
                    qos_pdb_ms=30,        
                    qos_per=1e-3,        
                    qos_gfbr_mbps=None,
                    li_enabled=False,
                    li_dir=None,
                ),
                _rule(
                    rule_id="pcc_dl",
                    direction="downlink",
                    application="Real-time gaming",
                    src_ip="any",
                    dst_ip="any",
                    gpsi="+012111111",
                    qos_5qi=3,
                    qos_priority=30,
                    qos_pdb_ms=50,        
                    qos_per=1e-3,
                    qos_gfbr_mbps=None,
                    li_enabled=True,     
                    li_dir="downlink",
                ),
            ],
            "rationale": [
                "Real-time gaming service mapped to 5QI=3 for both uplink and downlink.",
                "Uplink packet delay budget explicitly set to 30 ms as requested in the intent; other QoS fields from the 5QI=3 mapping.",
                "Downlink uses default 5QI=3 packet delay budget of 50 ms and packet error rate 1e-3.",
                "Lawful intercept enabled for downlink traffic only (mirror_direction='downlink'); uplink remains without lawful intercept."
            ],
            "assumptions": [],
            "confidence": 100,
        },
    })



    return policies



# Telemetry generators
ORCH_ALWAYS_APPLIED = {"status": "applied", "reason": None}


def base_observed_state(policy_id: str) -> Dict[str, Any]:
    return {
        "policy_id": policy_id,
        "nfr_retry_count": 0, 
        "modules_ready": True,
        "chain_connected": True,
        "rules_installed": True,
        "modules": {
            "ISF": {"state": "Running", "ready": True},
            "ULF": {"state": "Running", "ready": True},
            "DLF": {"state": "Running", "ready": True},
        },
        "connectivity": {
            "ISF->ULF": {"reachable": True},
            "ISF->DLF": {"reachable": True},
        },
        "rules": {},
        "telemetry": {},
    }


def telemetry_met(policy_id: str) -> Dict[str, Any]:
    obs = base_observed_state(policy_id)

    if policy_id == "intent_2":
        obs["telemetry"] = {"downlink_throughput_mbps": 12.0, "downlink_latency_ms": 60}

    elif policy_id in ("intent_3", "intent_4"):
        obs["telemetry"] = {
            "uplink_latency_ms": 120,
            "downlink_latency_ms": 200,
            "uplink_packet_error_rate": 5e-7,
            "downlink_packet_error_rate": 5e-7,
        }

    elif policy_id == "intent_5":
        obs["telemetry"] = {
            "uplink_latency_ms": 20,
            "downlink_latency_ms": 40,
            "uplink_packet_error_rate": 5e-4,
            "downlink_packet_error_rate": 5e-4,
        }

    elif policy_id == "intent_6":
        obs["telemetry"] = {
            "uplink_latency_ms": 20,    
            "downlink_latency_ms": 25,  
            "uplink_packet_error_rate": 5e-4,   
            "downlink_packet_error_rate": 5e-4, 
        }
        obs["modules"]["ODF"] = {"state": "Running", "ready": True}

    else:
        
        obs["telemetry"] = {"end_to_end_latency_ms": 80, "achieved_throughput_mbps": 10.0}

    return obs


def telemetry_not_met(policy_id: str) -> Dict[str, Any]:
    obs = base_observed_state(policy_id)

    if policy_id == "intent_2":
        obs["telemetry"] = {"downlink_throughput_mbps": 2.0, "downlink_latency_ms": 180}

    elif policy_id in ("intent_3", "intent_4"):
        obs["telemetry"] = {
            "uplink_latency_ms": 450,
            "downlink_latency_ms": 600,
            "uplink_packet_error_rate": 1e-4,
            "downlink_packet_error_rate": 1e-4,
        }

    elif policy_id == "intent_5":
        obs["telemetry"] = {
            "uplink_latency_ms": 70,   
            "downlink_latency_ms": 90,  
            "uplink_packet_error_rate": 5e-4,
            "downlink_packet_error_rate": 5e-4,
        }

    elif policy_id == "intent_6":
        obs["telemetry"] = {
            "uplink_latency_ms": 40,    
            "downlink_latency_ms": 80,  
            "uplink_packet_error_rate": 2e-3,  
            "downlink_packet_error_rate": 2e-3, 
        }
        obs["modules"]["ODF"] = {"state": "CrashLoopBackOff", "ready": False}

    else:
        obs["telemetry"] = {"end_to_end_latency_ms": 300, "achieved_throughput_mbps": 0.5}

    return obs



def run_monitor_only_tests() -> None:
    policies = ground_truth_policies()
    totals = {"runs": 0, "time_s": 0.0, "usage": _usage_dict()}

    for item in policies:
        policy_id = item["policy_id"]
        policy = item["pcc_policy"]
        pcc_rules = policy.get("pcc_rules", [])
        if not isinstance(pcc_rules, list):
            pcc_rules = []

        banner(f"POLICY: {policy_id}")

        print("\n[OPERATOR INTENT]")
        print(INTENT_TEXTS.get(policy_id, "(missing intent text)"))

        pretty_print("GROUND TRUTH PCC POLICY", policy)

        if policy_id == "intent_1":
            obs = telemetry_met(policy_id)

            t0 = time.perf_counter()
            mon = _call_monitor(
                pcc_rules=pcc_rules,
                orchestrator_result=ORCH_ALWAYS_APPLIED,
                observed_state=obs,
            )
            dt = time.perf_counter() - t0
            usage = _get_usage(mon)

            pretty_print("OBSERVED STATE", obs)
            pretty_print("MONITOR OUTPUT", mon)
            print(f"\n[MONITOR METRICS] policy={policy_id} runs=1 time_s={dt:.3f} tokens={usage}")

            totals["runs"] += 1
            totals["time_s"] += dt
            _add_usage(totals["usage"], usage)
            continue

        for case_name, obs in [("MET", telemetry_met(policy_id)), ("NOT_MET", telemetry_not_met(policy_id))]:
            t0 = time.perf_counter()
            mon = _call_monitor(
                pcc_rules=pcc_rules,
                orchestrator_result=ORCH_ALWAYS_APPLIED,
                observed_state=obs,
            )
            dt = time.perf_counter() - t0
            usage = _get_usage(mon)

            pretty_print(f"OBSERVED STATE ({case_name})", obs)
            pretty_print(f"MONITOR OUTPUT ({case_name})", mon)
            print(f"\n[MONITOR METRICS] policy={policy_id} case={case_name} time_s={dt:.3f} tokens={usage}")

            totals["runs"] += 1
            totals["time_s"] += dt
            _add_usage(totals["usage"], usage)

    banner("SUMMARY")
    print(
        f"Total runs={totals['runs']} | total_time_s={totals['time_s']:.3f} | "
        f"tokens_total={totals['usage']['total_tokens']} "
        f"(prompt={totals['usage']['prompt_tokens']}, completion={totals['usage']['completion_tokens']})"
    )


if __name__ == "__main__":
    run_monitor_only_tests()
