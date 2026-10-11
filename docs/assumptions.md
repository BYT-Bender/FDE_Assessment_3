# Assumptions and known limitations

- The supplied dataset and policy snapshot are synthetic. `data/procurement_policy.md` and its 2026-09-30 snapshot are used as the authority for this assignment.
- Available budget is read from `available_usd` and checked as annual budget minus committed spend. Committed spend is not treated as available.
- Thresholds and mandatory approvers remain deterministic code. Every recommendation still requires authorized human review; no approval is recorded.
- An empty integrations list means the requester has confirmed no integration. Null integrations remain unknown where supplied by the caller.
- The internal vendor CSV and mock API are separate sources. Both are presented when available; conflict, unknown status, service failure and policy-defined stale review are escalated.
- Catalog matching is a heuristic and identifies possible overlap, not equivalence. Purchase history is supporting evidence only.
- Gemini API is the only supported provider, accessed through Google's documented OpenAI-compatible endpoint. `GEMINI_API_KEY` and `GEMINI_MODEL` configure Gemini; `gemini-3.1-flash-lite` is the default. The installed `openai` Python SDK is only the compatibility client for the Gemini endpoint; no OpenAI provider or automatic OpenAI fallback is configured. Real calls require a key supplied privately in the local project-root `.env`.
- Gemini free-tier eligibility, model availability, and quota are determined by Google and may change. Free-tier content use may differ from paid service; synthetic assignment data is used here.
- Automated tests use mocked SDK responses. Live model quality, latency and cost have not been measured because `.env` currently contains no API key. The real comparison is ready but pending credentials.
- Demo-mode comparison results validate deterministic behavior, not AI quality. A live ship decision remains inconclusive until both architectures run against the same cases and their failures, cost/latency, and grounding are reviewed.
- There is no durable request store, reviewer identity, actual approval workflow, or reviewer decision log. These are outside the supplied starter data and need authorized integration before production.
- Nine evaluation cases reuse synthetic request IDs to isolate incomplete input, injection, and insufficient evidence behavior. The six public evaluations are retained separately.
