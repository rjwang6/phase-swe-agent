#!/usr/bin/env python3
"""
Extract actions and their corresponding phases from a mini-swe-agent trajectory JSON.

Usage:
    python extract_actions.py trajectory.json

Optional:
    python extract_actions.py trajectory.json --json-out actions.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_tool_arguments(raw_args: Any) -> Dict[str, Any]:
    """
    Tool arguments sometimes come in as a JSON string, sometimes already parsed.
    """
    if isinstance(raw_args, dict):
        return raw_args
    if isinstance(raw_args, str):
        raw_args = raw_args.strip()
        if not raw_args:
            return {}
        try:
            return json.loads(raw_args)
        except json.JSONDecodeError:
            return {"raw_arguments": raw_args}
    return {}


def extract_actions(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    messages = data.get("messages", [])
    extracted: List[Dict[str, Any]] = []

    for msg_idx, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue

        extra = msg.get("extra", {}) or {}
        phase = extra.get("phase")
        step_index = extra.get("step_index")
        timestamp = extra.get("timestamp")

        # Best case: actions are already normalized in extra.actions
        extra_actions = extra.get("actions")
        if isinstance(extra_actions, list) and extra_actions:
            for action_idx, action in enumerate(extra_actions, start=1):
                tool = action.get("tool")
                record: Dict[str, Any] = {
                    "message_index": msg_idx,
                    "step_index": step_index,
                    "phase": phase,
                    "timestamp": timestamp,
                    "action_index": action_idx,
                    "tool": tool,
                    "tool_call_id": action.get("tool_call_id"),
                }

                # Preserve common fields nicely
                if "command" in action:
                    record["command"] = action["command"]
                if "phase" in action:
                    record["target_phase"] = action["phase"]
                if "reason" in action:
                    record["reason"] = action["reason"]

                # Include any extra fields not already captured
                for k, v in action.items():
                    if k not in record and k not in {"tool"}:
                        record.setdefault(k, v)

                extracted.append(record)
            continue

        # Fallback: reconstruct from raw tool_calls
        tool_calls = msg.get("tool_calls") or []
        for action_idx, tc in enumerate(tool_calls, start=1):
            fn = tc.get("function", {}) or {}
            tool_name = fn.get("name")
            parsed_args = parse_tool_arguments(fn.get("arguments"))

            record = {
                "message_index": msg_idx,
                "step_index": step_index,
                "phase": phase,
                "timestamp": timestamp,
                "action_index": action_idx,
                "tool": tool_name,
                "tool_call_id": tc.get("id"),
            }

            if tool_name == "bash":
                record["command"] = parsed_args.get("command")
            elif tool_name == "switch_phase":
                record["target_phase"] = parsed_args.get("phase")
                record["reason"] = parsed_args.get("reason")
            else:
                record["arguments"] = parsed_args

            extracted.append(record)

    return extracted


def format_action(action: Dict[str, Any]) -> str:
    header_bits = []

    if action.get("step_index") is not None:
        header_bits.append(f"step {action['step_index']}")
    if action.get("phase"):
        header_bits.append(f"phase={action['phase']}")
    if action.get("action_index") is not None:
        header_bits.append(f"action {action['action_index']}")

    header = " | ".join(header_bits) if header_bits else "action"

    lines = [f"[{header}]"]

    tool = action.get("tool", "<unknown>")
    lines.append(f"  tool: {tool}")

    if "command" in action and action["command"] is not None:
        lines.append("  command:")
        for line in str(action["command"]).splitlines():
            lines.append(f"    {line}")

    if "target_phase" in action and action["target_phase"] is not None:
        lines.append(f"  target_phase: {action['target_phase']}")

    if "reason" in action and action["reason"] is not None:
        lines.append(f"  reason: {action['reason']}")

    return "\n".join(lines)


def print_nice(actions: List[Dict[str, Any]]) -> None:
    if not actions:
        print("No actions found.")
        return

    current_phase: Optional[str] = None
    for action in actions:
        phase = action.get("phase", "unknown")
        if phase != current_phase:
            current_phase = phase
            print()
            print("=" * 80)
            print(f"PHASE: {current_phase}")
            print("=" * 80)

        print(format_action(action))
        print()


def print_phase_commands(actions):
    """
    Print a compact list of phases followed by their commands.
    Multi-line bash commands are collapsed to one line.
    """
    phase_map = {}

    for a in actions:
        phase = a.get("phase", "unknown")
        tool = a.get("tool")

        if tool == "bash":
            cmd = a.get("command", "").strip()

            # Collapse multiline commands
            if "\n" in cmd:
                first_line = cmd.splitlines()[0].strip()

                if first_line.startswith("cat <<"):
                    cmd = first_line + " ..."
                else:
                    cmd = first_line + " ..."

        elif tool == "switch_phase":
            cmd = f"SWITCH_PHASE -> {a.get('target_phase')}"
        else:
            continue

        phase_map.setdefault(phase, []).append(cmd)

    for phase, cmds in phase_map.items():
        print(f"\n{phase}")
        for c in cmds:
            print(f"  {c}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectory_json", type=Path, help="Path to trajectory JSON file")
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to save extracted actions as JSON",
    )
    args = parser.parse_args()

    data = load_json(args.trajectory_json)
    actions = extract_actions(data)

    print_phase_commands(actions)

    if args.json_out is not None:
        with args.json_out.open("w", encoding="utf-8") as f:
            json.dump(actions, f, indent=2, ensure_ascii=False)
        print(f"Saved extracted actions to: {args.json_out}")


if __name__ == "__main__":
    main()