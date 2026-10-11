# Architecture decision memo

## Decision: live-model ship choice is inconclusive

Do not select between the single-agent and staged two-agent architecture yet. Ship neither as a production decision system until the full real-model comparison is run and reviewed. The single-agent path remains the simpler candidate for a controlled pilot, while the staged path remains useful for the required comparison.

## Evidence available

The paired offline run used the same nine synthetic cases, shared tools and deterministic controls:

| Measure | Single | Staged |
|---|---:|---:|
| Expected outcomes | 9/9 | 9/9 |
| Evidence grounding and policy assertions | 9/9 | 9/9 |
| Human review retained | 9/9 | 9/9 |
| Mean latency | 11.32 ms | 10.04 ms |
| Mean tool calls | 4 | 4 |
| Mean LLM calls | 0 | 0 |

Those figures measure deterministic code and local service latency only. They say nothing about model quality. Six supplied public cases also passed in each workflow.

A one-case live diagnostic then exercised EVAL-01 serially through both Gemini workflows. Both completed without an API error and retrieved all four tool results, but both recommended clarification where the rubric expected a hold for human review. Both therefore failed the recommendation assertion. Single took 5.36 seconds and made 2 LLM calls; staged took 8.63 seconds and made 3. Grounding was 0/1 for single and 1/1 for staged; deterministic policy safety and human escalation were retained in both. One case is insufficient to compare model quality or select an architecture. The full nine-case live run has not been completed.

## Trade-offs and next decision

The single architecture has one model workflow and a smaller operational surface. The staged variant adds an independent reviewer call, which may improve detection of unsupported claims or risk gaps but adds latency, cost and another failure point. The full real evaluation command is `python evals/run_comparison.py --mode real`; it records matched per-case outcomes, grounding, policy assertions, escalation, latency, tool/LLM call counts, and classified failures. The one-case diagnostic shows the real path runs but currently misses its expected outcome; it does not justify choosing either architecture. Human authorization stays outside both designs.
