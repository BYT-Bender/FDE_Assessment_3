# Architecture and workflow

```mermaid
sequenceDiagram
  actor Employee
  participant UI as Streamlit UI
  participant Flow as Validated workflow
  participant LLM as Gemini API (OpenAI-compatible endpoint)
  participant Tools as Allow-listed local tools
  participant Sources as CSV/JSON and policy snapshot
  participant API as FastAPI vendor-risk mock
  participant Rules as Deterministic policy controls
  participant Human as Authorized reviewers
  Employee->>UI: Submit request and select architecture/mode
  UI->>Flow: Validate request fields
  alt Single-agent real mode
    Flow->>LLM: Request analysis with strict tool definitions
    LLM->>Tools: Function calls (budget, catalog, vendor, policy)
    Tools->>Sources: Read evidence
    Tools->>API: GET vendor-risk/{vendor_name}
    Tools-->>LLM: Structured results or explicit unavailable status
    LLM-->>Flow: Structured ProcurementAssessment
  else Staged two-agent real mode
    Flow->>LLM: Stage 1 analyst call with shared tools
    LLM->>Tools: Function calls
    Tools->>Sources: Read evidence
    Tools->>API: GET vendor-risk/{vendor_name}
    Tools-->>LLM: Evidence and status
    LLM-->>Flow: Structured analyst evidence package
    Flow->>LLM: Stage 2 independent policy/risk review
    LLM-->>Flow: Structured review and recommendation
  else Deterministic demo mode
    Flow->>Tools: Shared tools
    Tools->>Sources: Read evidence
    Tools->>API: GET vendor-risk/{vendor_name}
  end
  Flow->>Rules: Apply thresholds, required approvals and escalation rules
  Rules-->>Flow: Authoritative deterministic results
  Flow-->>UI: Validated recommendation, evidence, risks and next steps
  UI-->>Human: Explicit handoff; no authorization recorded
```

## Component boundaries

- `app.py`: request form, architecture/mode selection, status and result presentation; does not receive the API key.
- `src/solution.py`: Pydantic input validation, real/demo selection, deterministic checks, conservative merge and shared result contract.
- `src/agents.py`: Gemini chat-completion function calls through Google's documented OpenAI-compatible endpoint, validated arguments, structured output schemas, bounded call budgets, refusal/provider/malformed-output handling. `GEMINI_API_KEY` and `GEMINI_MODEL` select/configure Gemini; the `openai` Python SDK is only the compatibility client and no OpenAI model/provider fallback exists.
- `src/tools.py`: budget, catalog/purchase history, vendor/security and policy tools. Each has an input/output contract and useful service-failure state.
- `src/data_access.py`: supplied local CSV/JSON sources.
- `mock_api/app.py`: supplied local FastAPI mock service; `src/vendor_client.py` calls it with a timeout.
- `evals/`: shared nine-case suite, deterministic assertions and mode-separated result artifacts.

## Safety and stopping conditions

Only named local tools are exposed. Tool arguments are checked against verified requester department, submitted annual cost and vendor name. The tool loop stops after the required evidence tools have been gathered, no more than 8 tool calls or 10 model calls. The analyst and reviewer each produce schema-validated JSON; the staged reviewer runs once and does not loop. A model response cannot reduce deterministic approvals/risks or clear human review. Model-only statements are labeled as interpretations, and unsupported citations become a risk. Missing, stale, conflicting or unavailable evidence cannot become a safe-vendor finding.

`.env` is loaded from the project root; the OS environment takes precedence. `GEMINI_API_KEY` stays in the Python server process. Deterministic demo is available without credentials, and live evaluation requires an explicit `--mode real` plus a key; it stops rather than falling back. Free-tier availability and quota are controlled by Google and can change; Google documents that free-tier content may be used to improve its products.
