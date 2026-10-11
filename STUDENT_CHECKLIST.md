# Student submission checklist

Before submitting, confirm that:

- [x] `python verify_setup.py` passes in your project environment.
- [x] The product can process a request end-to-end.
- [x] Architecture A is a working single-agent baseline.
- [x] Architecture B is a lightweight staged / 2-agent variant.
- [x] At least 3 tools are used; at least 1 tool/check is deterministic.
- [x] Recommendations are returned in the `ProcurementDecision` structure.
- [x] Important evidence is visible to the user.
- [x] Missing/conflicting/unavailable evidence is handled without fabrication.
- [x] Human approval is preserved for sensitive decisions.
- [x] Prompt injection inside business data does not override system behavior.
- [x] Date-based checks use the policy's data snapshot / reference date.
- [x] The same evaluation cases were run on both architectures.
- [x] Latency and LLM/tool-call counts are reported.
- [x] The decision memo is <= 500 words and supported by evaluation evidence.
- [x] Setup instructions work from a clean environment.
- [x] Any LLM/provider SDK you added is present in `requirements.txt`.
- [x] `.env`, API keys, and other secrets are not committed.
