# phase-swe-agent

**A phase-structured LLM agent for software engineering, with hierarchical memory that keeps prompt size bounded as tasks get longer.**

Stanford CS 224N course project. Built on top of [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) (MIT).

---

## The problem

A standard ReAct-style coding agent appends every action and every command output to one growing message list. On SWE-bench tasks that run 40–100 steps, that transcript becomes the dominant cost: the prompt grows linearly with step count, the model re-reads stale `grep` output from step 3 while trying to fix a test at step 60, and long runs eventually hit the context ceiling.

The usual fix is summarization or truncation, which is lossy in an unstructured way — you cannot control *what* survives.

## The approach

Give the agent an explicit notion of **what it is currently doing**, and let that structure decide what it carries forward.

The agent moves between three phases and switches between them deliberately, via a `switch_phase` tool call alongside the usual `bash` tool:

| Phase | Job | Bias |
|---|---|---|
| **Exploration** | Locate relevant code, form a root-cause hypothesis, produce a concrete plan | Read and search, don't edit |
| **Execution** | Apply the plan as targeted edits | Act on the plan, don't re-explore |
| **Validation** | Run tests, confirm the fix, decide if it actually worked | Verify, then report |

Validation loops back to exploration when the fix didn't hold, so the agent cycles rather than running a fixed pipeline.

### Memory is rebuilt every step, not appended to

The key departure from the base agent: the prompt is **reconstructed from scratch on every step** rather than accumulated. `_refresh_phase_prompts()` assembles exactly four things:

```
1. Phase-specific system + instance prompt   (what am I doing right now)
2. Handoff document                          (what the previous phase concluded)
3. Distilled task memory                     (what is true about this task, across all phases)
4. Current phase transcript only             (verbatim — but reset at every phase switch)
```

Everything older than the current phase is not dropped — it is **compressed at the phase boundary** by a separate LLM call:

- **Handoff** (`memory/components/agent_handoff.py`) — written when leaving a phase, addressed to the next one. Exploration hands execution a plan; validation hands exploration a failure report.
- **Distilled task memory** (`memory/components/agent_distilled_memory.py`) — a persistent, rewritten-in-place summary of confirmed file locations, the bug mechanism, and reproduction commands. It survives every phase switch.

So the raw transcript is bounded by the length of the *current phase* (median 7 steps), not the length of the run (median 33 steps). Compression happens at a semantically meaningful boundary, where the agent has just finished a coherent unit of work and knows what mattered.

Both compression steps are separate LLM calls driven by dedicated prompt templates in
`config/benchmarks/swebench.yaml` — the handoff writer and the distilled-memory updater are
prompted differently depending on which phase is being left.

A real distilled memory from the run is in [`results/examples/astropy__astropy-12907/`](results/examples/astropy__astropy-12907/) — it isolates the buggy branch in `_cstack` and carries a reproduction script forward.

---

## Results

Evaluated on **SWE-bench Verified** with `gemini-3-flash-preview`.

**Best run** — 47 of the 300 instances attempted (budget-limited, not a full sweep):

| | |
|---|---|
| Attempted | 47 |
| Completed (produced a patch) | 42 |
| **Resolved** | **25** |
| Unresolved | 17 |
| Empty patch | 5 |

That is **53% of attempted** / 60% of completed instances. Of the 5 non-completions, all hit the step/cost limit rather than erroring.

> Scope note: this is a partial run on a 47-instance subset, not a full 300-instance SWE-bench Verified score, and it is not directly comparable to published leaderboard numbers.

### Prompt size stays flat

Measured across all 47 tasks (`results/token_stats/`, 236 phase segments):

| Metric | Value |
|---|---|
| Mean prompt tokens per step | **5,613** |
| Median prompt tokens per step | 5,104 |
| Largest prompt observed | 22,748 |
| Mean steps per task | 43.3 (max 100) |
| Mean phase segment length | 8.6 steps |

The point of the middle row against the bottom two: a 43-step run holds an average prompt of ~5.6k tokens. Under naive accumulation the final prompt would be several times that, because every step's output is still in the context. Prompt size tracks the current phase, not the run.

### Iteration history

Earlier design revisions on a 23-instance development subset, showing the effect of the prompt and memory work:

| Run | Attempted | Resolved |
|---|---|---|
| `phase_memory_1` | 23 | 5 |
| `phase_memory_v2_1` | 23 | 4 |
| **`ryan_1`** (final, 47-instance) | 47 | **25** |

Raw reports are in [`results/`](results/).

---

## Repository layout

The parts that are this project's contribution:

```
src/minisweagent/
├── phases/
│   ├── phase.py                          Phase enum (exploration/execution/validation)
│   ├── exploration/prompt.txt            Phase-specific system prompts
│   └── execution/prompt.txt
├── memory/
│   ├── components/
│   │   ├── agent_distilled_memory.py     Persistent cross-phase task memory
│   │   ├── agent_handoff.py              Phase-to-phase handoff documents
│   │   └── agent_scratchpad.py           Within-phase scratchpad (experimental)
│   └── generate/memory_builder.py        LLM-driven memory construction
├── agents/default.py                     Phase state machine, prompt reconstruction,
│                                         phase-switch logic, token instrumentation
└── config/benchmarks/swebench.yaml       15 prompt templates (~1.8k lines): per-phase
                                          system/instance prompts, handoff writers,
                                          distilled-memory update prompts

analysis/                                 Result analysis
├── phase_length_stats.py                 Phase-length distributions
├── phase_lengths_plot.py                 Plot generation
├── prompt_token_composition.py           Where prompt tokens actually go
└── action_by_phase.py                    Action-type breakdown per phase

results/                                  Run data backing the numbers above
```

Everything else — environments, model backends, the SWE-bench runner, tests — is upstream mini-swe-agent, lightly modified.

The core logic to read is `DefaultAgent._refresh_phase_prompts()` and `DefaultAgent._switch_phase()` in `src/minisweagent/agents/default.py`.

---

## Running it

```bash
git clone https://github.com/rjwang6/phase-swe-agent
cd phase-swe-agent
pip install -e .
```

Set an API key for whichever provider you use (see [litellm docs](https://docs.litellm.ai/docs/providers)):

```bash
export GEMINI_API_KEY=...   # or OPENAI_API_KEY, ANTHROPIC_API_KEY, ...
```

Run interactively on a local repo:

```bash
python -m minisweagent.run.mini -m gemini/gemini-2.5-flash
```

Run the SWE-bench benchmark:

```bash
python -m minisweagent.run.benchmarks.swebench \
    --subset verified --split test \
    -m gemini/gemini-2.5-flash \
    --workers 4
```

Each run writes per-step prompt logs to `logs/<task_id>/`, memory artifacts to `current_agent_memory/<task_id>/`, and token statistics to `token_stats/<task_id>`. Point the scripts in `analysis/` at those to reproduce the tables above.

Requires Python ≥ 3.10.

---

## Limitations

- The 47-instance run is budget-limited and single-seed; there is no matched baseline run of the unmodified agent under identical conditions, so the resolve rate is not a controlled comparison.
- Phase switching is left to the model's judgment. It sometimes lingers in exploration — exploration segments average 12.3 steps against 6.3 for execution.
- Every phase switch costs an extra LLM call to write the handoff and distilled memory, trading tokens-per-step against calls-per-task.
- `agent_scratchpad.py` is an earlier within-phase memory design that the final system does not use; it is kept for reference.

---

## Credits

Built on [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) by Kilian Lieret and Carlos E. Jimenez (MIT — see `LICENSE`).

Course project for CS 224N at Stanford, joint with [Sameer Agrawal](https://github.com/agrawalsameer1). The phase-structured control flow and memory system in this repository are my own work; see `NOTICE`.
