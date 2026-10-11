# Evaluation guide

`evaluation_cases.json` contains nine shared synthetic cases. Both architectures use the same requests, data, tools and deterministic controls. EVAL-07 adds the clearly labeled synthetic catalog injection fixture from `fixtures/catalog_prompt_injection.json`; the fixture is not represented as a company record, and its request text is sanitized so the case isolates retrieved-data injection. Assertions cover expected next action, required approvals, required safety flags and mandatory human review. Metrics include recommendation accuracy, reference coverage, deterministic policy compliance, human escalation, latency, model calls, tool calls and failures.

## Offline deterministic mode

No API key is needed. The `demo` mode is explicitly reported and does not measure AI quality:

```powershell
python evals/run_comparison.py --mode demo
```

Artifacts: `results_demo.json`, `results_demo.csv`, `results_demo_summary.md`. The legacy `results.json`, `results.csv`, `results_summary.md` paths mirror this demo run and carry `mode: demo` in the data. The six supplied public cases are run with:

```powershell
python evals/run_public_evals.py --architecture single --mode demo
python evals/run_public_evals.py --architecture staged --mode demo
```

## Real model mode

Add a valid `GEMINI_API_KEY` to project-root `.env`, then run:

```powershell
python evals/run_comparison.py --mode real
```

The real mode is the runner default, fails before execution if the key is missing, and never switches to demo. Begin with one case per architecture:

```powershell
python evals/run_comparison.py --mode real --limit 1 --timeout-seconds 25 --rate-limit-retries 0 --request-pacing-seconds 0
```

That runs EVAL-01 once for each architecture, serially. If it completes successfully, run the full suite with the command above. `--limit N` selects the first N cases identically for both architectures. The diagnostic options reduce the per-request timeout and disable pacing and rate-limit retries; omit them for the normal run. Default 429 handling allows at most two retries, respects numeric/date `Retry-After` or `Retry-After-Ms` when present, and caps a single wait at 30 seconds. Request pacing defaults to one second between starts. Override either setting in `.env` or with CLI options.

Results are checkpointed after each case so partial API failures remain visible. Live artifacts use `results_real.*`, identify the model and execution timestamps, and include per-case status (`pass`, assertion `fail`, or categorized `error`), API error category, interruption or tool-loop termination reason, latency and call counts. Aggregate pass/fail/error and API-error counts are reported per architecture. Infrastructure-error cases are excluded from model-quality accuracy calculations and counted separately. The two architectures run sequentially under equivalent settings. Unit tests mock SDK responses and never spend API credits.

Each case also records sanitized `diagnostics`: model-response finish reason and shape, returned tool names/count, tool-result status/cache use, evidence coverage, and schema validation error locations/types. Prompt text, tool arguments, response bodies, and credentials are never copied into diagnostics. `model_tool_calls` counts function calls requested by the model; `tool_calls` counts actual tool executions, so repeated/cached requests are visible separately. The result metadata includes selected, processed, and unprocessed case IDs by architecture, plus any interrupted case.

The `auto` mode selects real when a key is configured and otherwise explicitly records `demo`; use `real` when a live evaluation is intended. Do not compare demo timing with model latency or describe demo scores as model performance.
