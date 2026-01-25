# usage_utils.py
from __future__ import annotations
from typing import Any, Dict

def extract_usage_from_ai_message(msg: Any) -> Dict[str, int]:
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    um = getattr(msg, "usage_metadata", None)
    if isinstance(um, dict):
        usage["prompt_tokens"] = int(um.get("input_tokens", um.get("prompt_tokens", 0)) or 0)
        usage["completion_tokens"] = int(um.get("output_tokens", um.get("completion_tokens", 0)) or 0)
        usage["total_tokens"] = int(um.get("total_tokens", 0) or 0)
        if usage["total_tokens"] == 0:
            usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
        return usage

    rm = getattr(msg, "response_metadata", None)
    if isinstance(rm, dict):
        tu = rm.get("token_usage") or rm.get("usage")
        if isinstance(tu, dict):
            usage["prompt_tokens"] = int(tu.get("prompt_tokens", 0) or 0)
            usage["completion_tokens"] = int(tu.get("completion_tokens", 0) or 0)
            usage["total_tokens"] = int(tu.get("total_tokens", 0) or 0)
            if usage["total_tokens"] == 0:
                usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
            return usage

    return usage
