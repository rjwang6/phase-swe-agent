# phase-swe-agent

## Can coding agents manage their own memory?

**A coding agent that splits a development cycle into explicit phases and decides for itself when to compress what it has learned — landing within 6 points of an unlimited-context agent's solve rate on SWE-bench Lite at 60% of the prompt tokens.**

Stanford CS 224N course project. Built on top of [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) (MIT).

📄 **[Read the full report (8 pages)](docs/cs224n_report.pdf)**

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

Evaluated on **SWE-bench Lite** with Gemini 3.0 Flash, capped at 70 steps per task.

The dataset is 51 instances sampled to span difficulty: mini-SWE-agent was run on the
first 100 SWE-bench Lite problems, bucketed by how many steps it took
(easy 1–23, medium 24–46, hard 47–70), and 17 tasks drawn at random from each bucket.

### Against context-management baselines

| Method | Correct | Incorrect | Incomplete | Accuracy of submitted |
|---|---|---|---|---|
| Default (unlimited context) | 28/51 (55%) | 21/51 (41%) | 2/51 (4%) | 28/49 (57%) |
| 4K truncation | 10/51 (20%) | 6/51 (12%) | 35/51 (69%) | 10/16 (63%) |
| 8K truncation | 22/51 (43%) | 14/51 (27%) | 15/51 (29%) | 22/36 (61%) |
| 4K summary | 14/51 (27%) | 10/51 (20%) | 27/51 (53%) | 14/24 (58%) |
| 8K summary | 23/51 (45%) | 19/51 (37%) | 9/51 (18%) | 23/42 (55%) |
| **Phase Memory (ours)** | **25/51 (49%)** | 15/51 (29%) | 11/51 (22%) | **25/40 (63%)** |

Two things to take from this:

- **It beats every limited-context baseline** at both 4K and 8K, and lands close to the
  unlimited-context default (49% vs 55%) while using an average of **5,812 prompt tokens
  against the default's 9,708**.
- **It has the highest accuracy on the patches it does submit — 63%.** We attribute this to
  the dedicated validation phase, which gives the agent a structured opportunity to reject
  a patch before submitting it.

### Prompt size stays flat as runs get longer

The default agent's average prompt climbs sharply with step count. Phase Memory's stays
roughly constant, because the transcript resets at every phase boundary — and most tasks
finish under 8K tokens, which is what makes it viable in a limited-context setting.

Average phase length, from the same run (`results/token_stats/`, 236 segments):

| Phase | Average length |
|---|---|
| Exploration | 12.3 steps |
| Execution | 6.3 steps |
| Validation | 7.1 steps |
| **Overall** | **8.6 steps** |

Exploration runs longest; execution and validation are shorter because they largely follow
the instructions already written into the exploration handoff. Average exploration length
also *falls* across later cycles, which is direct evidence the distilled memory is carrying
useful knowledge forward rather than just accumulating text.

### Where the prompt tokens actually go

| Component | Avg. tokens | Share of prompt |
|---|---|---|
| System / instance prompt | 2,123 | 38.5% |
| Phase handoff | 405 | 7.3% |
| Distilled memory | 472 | 8.6% |
| Phase transcript | 3,950 | 45.6% |

The surprise here is the first row: the phase-specific instructions and memory-format
explanations cost nearly as much as the transcript itself, and dominate early in a phase
when the transcript is still short. That is the obvious place to optimise next.

Raw harness reports, per-task token statistics, and model patches are in [`results/`](results/).

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
docs/cs224n_report.pdf                    Full write-up: method, baselines, analysis
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

### Tests

```bash
pip install -e ".[dev]"
pytest
```

441 tests pass, including the upstream suite. The phase and memory tests are
`src/minisweagent/phases/test_phase_switch.py` and
`src/minisweagent/memory/test_agent_memory.py`; they run offline, stubbing the
LLM call that a phase switch would otherwise make.

---

## Limitations

- **Phase boundaries are soft.** Switching is left to the model's judgment, and it does not
  always respect the phase it is in — writing files during exploration, for instance. Strict
  prompting mitigated but did not eliminate this. Hard tool-call restrictions per phase would
  enforce it properly.
- **The agent sometimes hallucinates file paths or validation commands** when writing memory,
  which then propagates into the next phase. Verifying paths before they are written into a
  handoff would help.
- **The system/instance prompt is 38% of the average prompt** — larger than it should be for a
  method whose point is token efficiency.
- **Single seed, one base model.** All runs used Gemini 3.0 Flash on 51 instances; no variance
  estimates across seeds or models.
- **Each phase switch costs an extra LLM call** to write the handoff and distilled memory,
  trading tokens-per-step against calls-per-task.
- `agent_scratchpad.py` is an earlier within-phase memory design the final system does not use;
  it is kept for reference.

---

## Credits

Built on [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) by Kilian Lieret and Carlos E. Jimenez (MIT — see `LICENSE`).

Course project for CS 224N at Stanford by **Sameer Agrawal, Ryan Wang, and Jerry Wang**.

Per the report's contribution statement: I implemented the base Phase Memory framework and
wrote the Introduction, Our Method, and Analysis sections. Sameer implemented the baselines
and built the dataset; Jerry ran the experiments. The code in this repository is the framework
implementation; the results above are the team's joint work. See `NOTICE`.
