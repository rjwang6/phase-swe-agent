from types import SimpleNamespace

from minisweagent.exceptions import FormatError
from minisweagent.models.utils.actions_toolcall import parse_toolcall_actions
from minisweagent.models.litellm_model import LitellmModel
from minisweagent.environments.local import LocalEnvironment
from minisweagent.phases.phase import Phase
from minisweagent.agents.default import DefaultAgent


# ----------------------------
# Fake model to avoid real API calls
# ----------------------------
class FakeModel:
    def format_observation_messages(
        self, message: dict, outputs: list[dict], template_vars: dict | None = None
    ) -> list[dict]:
        results = []
        actions = message.get("extra", {}).get("actions", [])
        for action, output in zip(actions, outputs):
            results.append(
                {
                    "role": "tool",
                    "tool_call_id": action.get("tool_call_id", "no-id"),
                    "content": (
                        f"TOOL={action.get('tool', 'bash')}\n"
                        f"RETURNCODE={output.get('returncode')}\n"
                        f"EXCEPTION={output.get('exception_info')}\n"
                        f"OUTPUT={output.get('output')}"
                    ),
                    "extra": output.get("extra", {}),
                }
            )
        return results

    def get_template_vars(self, **kwargs):
        return {}

    def format_message(self, role: str, content: str, extra: dict | None = None) -> dict:
        msg = {"role": role, "content": content}
        if extra is not None:
            msg["extra"] = extra
        return msg


# ----------------------------
# Minimal agent config
# ----------------------------
# AgentConfig requires every phase/memory prompt template. These tests only
# exercise phase-switch control flow, never prompt rendering, so stub them.
_REQUIRED_TEMPLATES = [
    "distilled_memory_update_after_exploration_template",
    "distilled_memory_update_after_validation_template",
    "exploration_handoff_write_template",
    "execution_report_write_template",
    "exploration_scratchpad_write_template",
    "execution_scratchpad_write_template",
    "phase_memory_update_template",
    "memory_write_template",
    "memory_summarizer_template",
    "exploration_system_template",
    "exploration_instance_template",
    "execution_system_template",
    "execution_instance_template",
    "validation_system_template",
    "validation_instance_template",
]


class OfflineAgent(DefaultAgent):
    """DefaultAgent with the LLM-backed memory writer stubbed out.

    Phase switches normally spend an API call writing the handoff and
    distilled memory. These tests only assert on control flow, so the
    generation seam is replaced with a deterministic string.
    """

    def _generate_memory_block(self, template, required_headers, **template_kwargs) -> str:
        return "stub memory block"


def make_agent(**overrides):
    """Build an offline DefaultAgent with all required templates stubbed out."""
    kwargs = {name: f"stub {name}" for name in _REQUIRED_TEMPLATES}
    kwargs.update(
        system_template="test system template",
        instance_template="test instance template",
        phase_recent_message_window=4,
    )
    kwargs.update(overrides)
    return OfflineAgent(model=FakeModel(), env=LocalEnvironment(), **kwargs)

# ----------------------------
# Helper printing
# ----------------------------
def print_header(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_pass(name: str):
    print(f"[PASS] {name}")


def print_fail(name: str, msg: str):
    print(f"[FAIL] {name}: {msg}")


# ----------------------------
# Tests
# ----------------------------
def test_1_parser_valid_switch_phase():
    print_header("TEST 1: parser accepts valid switch_phase tool call")

    tool_call = SimpleNamespace(
        id="test-call-1",
        function=SimpleNamespace(
            name="switch_phase",
            arguments='{"phase": "execution", "reason": "done exploring"}',
        ),
    )

    actions = parse_toolcall_actions(
        [tool_call],
        format_error_template="{{ error }}",
    )

    print("Parsed actions:", actions)

    assert len(actions) == 1
    assert actions[0]["tool"] == "switch_phase"
    assert actions[0]["phase"] == "execution"
    assert actions[0]["reason"] == "done exploring"
    assert actions[0]["tool_call_id"] == "test-call-1"

    print_pass("parser valid switch_phase")


def test_2_parser_invalid_phase():
    print_header("TEST 2: parser rejects invalid phase value")

    bad_tool_call = SimpleNamespace(
        id="test-call-2",
        function=SimpleNamespace(
            name="switch_phase",
            arguments='{"phase": "bad_phase", "reason": "oops"}',
        ),
    )

    caught = False
    try:
        parse_toolcall_actions(
            [bad_tool_call],
            format_error_template="{{ error }}",
        )
    except FormatError as e:
        caught = True
        print("Caught expected FormatError")
        print("Exception payload:", e.args)

    assert caught, "Expected FormatError for invalid phase"

    print_pass("parser invalid phase rejected")


def test_3_model_exposes_tools():
    print_header("TEST 3: LitellmModel._query source contains switch_phase tool")

    import inspect
    import minisweagent.models.litellm_model as litellm_model_module

    source = inspect.getsource(litellm_model_module.LitellmModel._query)
    print(source)

    compact = source.replace(" ", "").replace("\n", "")

    assert "SWITCH_PHASE_TOOL" in source, "LitellmModel._query does not reference SWITCH_PHASE_TOOL"
    assert "BASH_TOOL" in source, "LitellmModel._query does not reference BASH_TOOL"
    assert "tools=[BASH_TOOL,SWITCH_PHASE_TOOL]" in compact, (
        "LitellmModel._query does not appear to pass both tools"
    )

    print_pass("model wrapper exposes switch_phase")


def test_4_execute_action_switch_phase():
    print_header("TEST 4: direct execute_action() changes phase")

    agent = make_agent()

    print("Initial phase:", agent.phase)
    assert agent.phase == Phase.EXPLORATION

    result = agent.execute_action(
        {
            "tool": "switch_phase",
            "phase": "execution",
            "reason": "manual test",
            "tool_call_id": "switch-1",
        }
    )

    print("Result:", result)
    print("New phase:", agent.phase)

    assert agent.phase == Phase.EXECUTION
    assert result["returncode"] == 0
    assert "Switched phase from exploration to execution" in result["output"]

    print_pass("execute_action switch_phase")


def test_5_execute_actions_switch_phase_message():
    print_header("TEST 5: execute_actions() processes switch_phase action list")

    agent = make_agent()

    fake_message = {
        "extra": {
            "actions": [
                {
                    "tool": "switch_phase",
                    "phase": "execution",
                    "reason": "manual integration test",
                    "tool_call_id": "fake-call-1",
                }
            ]
        }
    }

    print("Initial phase:", agent.phase)
    obs_messages = agent.execute_actions(fake_message)
    print("New phase:", agent.phase)
    print("Observation messages:")
    for m in obs_messages:
        print(m)

    assert agent.phase == Phase.EXECUTION
    assert len(obs_messages) >= 1
    assert "switch_phase" in obs_messages[0]["content"]
    assert "RETURNCODE=0" in obs_messages[0]["content"]

    print_pass("execute_actions switch_phase")


def test_6_mixed_switch_phase_and_bash():
    print_header("TEST 6: mixed switch_phase + bash actions")

    agent = make_agent()

    fake_message = {
        "extra": {
            "actions": [
                {
                    "tool": "switch_phase",
                    "phase": "execution",
                    "reason": "ready to run commands",
                    "tool_call_id": "call-1",
                },
                {
                    "tool": "bash",
                    "command": 'python -c "print(\'hello from bash\')"',
                    "tool_call_id": "call-2",
                },
            ]
        }
    }

    print("Initial phase:", agent.phase)
    obs_messages = agent.execute_actions(fake_message)
    print("Final phase:", agent.phase)
    print("Observation messages:")
    for m in obs_messages:
        print("-" * 40)
        print(m)

    assert agent.phase == Phase.EXECUTION
    assert len(obs_messages) == 2
    assert "switch_phase" in obs_messages[0]["content"]
    assert "hello from bash" in obs_messages[1]["content"]

    print_pass("mixed switch_phase + bash")


def run_all_tests():
    tests = [
        test_1_parser_valid_switch_phase,
        test_2_parser_invalid_phase,
        test_3_model_exposes_tools,
        test_4_execute_action_switch_phase,
        test_5_execute_actions_switch_phase_message,
        test_6_mixed_switch_phase_and_bash,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            failed += 1
            print_fail(test.__name__, str(e))
        except Exception as e:
            failed += 1
            print_fail(test.__name__, f"{type(e).__name__}: {e}")

    print_header("FINAL SUMMARY")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if failed == 0:
        print("\nALL TESTS PASSED")
    else:
        print("\nSOME TESTS FAILED")


if __name__ == "__main__":
    run_all_tests()