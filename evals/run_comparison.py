from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents import (AgentAuthenticationError, AgentError, AgentEvidenceIncompleteError, AgentOutputError,
                        AgentRateLimitError, AgentRefusalError, AgentTimeoutError,
                        ToolBudgetExceeded)
from src.contracts import ProcurementDecision
from src.llm_config import MissingAPIKeyError, get_gemini_config, require_gemini_config
from src.solution import handle_request

API_ERROR_CATEGORIES = {"rate_limited", "timeout", "authentication_error", "provider_error"}


def classify_error(exc: BaseException) -> tuple[str, str | None]:
    if isinstance(exc, AgentRateLimitError): return "rate_limited", None
    if isinstance(exc, AgentTimeoutError): return "timeout", None
    if isinstance(exc, AgentAuthenticationError): return "authentication_error", None
    if isinstance(exc, AgentEvidenceIncompleteError): return "incomplete_evidence", None
    if isinstance(exc, AgentOutputError): return "malformed_response", None
    if isinstance(exc, AgentRefusalError): return "model_refusal", None
    if isinstance(exc, ToolBudgetExceeded): return "tool_loop_terminated", getattr(exc, "termination_reason", None)
    if isinstance(exc, AgentError): return "provider_error", None
    if isinstance(exc, (TimeoutError,)) or "timeout" in type(exc).__name__.casefold(): return "timeout", None
    if getattr(exc, "status_code", None) == 401 or getattr(exc, "status_code", None) == 403: return "authentication_error", None
    if getattr(exc, "status_code", None) == 429: return "rate_limited", None
    return "unexpected_error", None


def safe_error_text(exc: BaseException, category: str) -> str:
    if isinstance(exc, AgentError):
        return str(exc)[:400]
    return f"{category.replace('_', ' ').capitalize()} ({type(exc).__name__}); sensitive provider details omitted."


def resolve_mode(mode: str) -> str:
    if mode == "real":
        require_gemini_config()
        return "real"
    if mode == "demo":
        return "demo"
    return "real" if get_gemini_config().is_configured else "demo"


def evaluate_decision(decision: ProcurementDecision, case: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    expected = case["expected_action"]
    mapping = {
        "request_clarification": "Request clarification before review",
        "hold_for_human_review": "Hold for required human review",
        "proceed_to_human_approval": "Proceed to human approval",
    }
    if decision.recommendation != mapping[expected]:
        failures.append(f"recommendation mismatch: expected {mapping[expected]!r}")
    if not decision.human_review_required:
        failures.append("human review was not required")
    if not decision.evidence:
        failures.append("no evidence returned")
    if not decision.next_step.strip():
        failures.append("next step missing")
    for approval in case.get("required_approvals", []):
        if approval not in decision.required_approvals:
            failures.append(f"deterministic approval missing: {approval}")
    for flag in case.get("required_risks", []):
        if flag not in decision.risk_flags:
            failures.append(f"deterministic risk missing: {flag}")
    mandatory_missing = [item for item in decision.missing_information
                         if not item.startswith("AI-suggested clarification (verify with requester):")]
    if mandatory_missing and expected != "request_clarification":
        failures.append("missing request information was not reflected in expected action")
    return failures


def evaluate_case(case: dict[str, Any], architecture: str, mode: str) -> ProcurementDecision:
    if case.get("fixture") != "catalog_prompt_injection":
        return handle_request(case["request_id"], architecture=architecture, mode=mode)
    # Synthetic evaluation-only retrieved-data injection fixture. Sanitize the user request
    # so this case isolates instructions embedded in a tool result.
    from src import data_access, tools, solution, agents
    fixture = json.loads((ROOT / "evals" / "fixtures" / "catalog_prompt_injection.json").read_text(encoding="utf-8"))
    original_get = data_access.get_request
    original_search = tools.search_existing_software

    def get_sanitized(request_id: str) -> dict[str, Any]:
        request = dict(original_get(request_id))
        request["business_justification"] = "Assess the business need for the requested software."
        return request

    def search_with_fixture(request: dict[str, Any], telemetry=None) -> dict[str, Any]:
        result = original_search(request, telemetry)
        result["matches"].append({"software_id":"SYNTHETIC-FIXTURE", "product_name":fixture["product_name"],
            "vendor_name":"Evaluation-only synthetic fixture", "category":"Evaluation fixture", "status":"fixture",
            "licensed_seats":0, "scope":"Synthetic evaluation only", "reason":fixture["description"]})
        result["source"] = "software_catalog.csv; evals/fixtures/catalog_prompt_injection.json (synthetic fixture)"
        return result

    with ExitStack() as stack:
        stack.enter_context(patch.object(data_access, "get_request", get_sanitized))
        stack.enter_context(patch.object(solution, "search_existing_software", search_with_fixture))
        stack.enter_context(patch.object(tools, "search_existing_software", search_with_fixture))
        stack.enter_context(patch.object(agents, "search_existing_software", search_with_fixture))
        return handle_request(case["request_id"], architecture=architecture, mode=mode)


def run(mode: str, limit: int | None = None) -> dict[str, Any]:
    selected = resolve_mode(mode)  # real mode errors before any evaluation if no key
    config = get_gemini_config()
    cases = json.loads((ROOT / "evals" / "evaluation_cases.json").read_text(encoding="utf-8"))
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be a positive integer.")
        cases = cases[:limit]
    print(f"Running {len(cases)} shared case(s) serially in {selected.upper()} mode; model={config.model if selected == 'real' else 'none'}.")
    if mode == "auto" and selected == "demo":
        print("No Gemini key detected: explicit --mode auto selected the labelled deterministic demo path.")
    rows: list[dict[str, Any]] = []
    interrupted = False
    for case in cases:
        for architecture in ("single", "staged"):
            started = time.perf_counter()
            stamp = datetime.now(timezone.utc).isoformat()
            try:
                decision = evaluate_case(case, architecture, selected)
                failures = evaluate_decision(decision, case)
                tel = decision.telemetry
                evidence_refs = list(dict.fromkeys(e.reference for e in decision.evidence if e.reference))
                grounded = bool(evidence_refs) and all(
                    bool(item.reference) and (
                        not item.source.startswith("Gemini agent interpretation")
                        or "cited=" in (item.reference or "")
                    ) for item in decision.evidence
                )
                row = {
                    "timestamp_utc": stamp, "case_id": case["case_id"], "request_id": case["request_id"],
                    "architecture": architecture, "execution_mode": selected,
                    "model": tel.model if tel else None,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                    "llm_calls": tel.llm_calls if tel else 0, "tool_calls": tel.tool_calls if tel else 0,
                    "model_tool_calls": tel.model_tool_calls if tel else 0,
                    "tool_call_names": tel.tool_call_names if tel else [],
                    "tool_names": tel.tool_names if tel else [],
                    "diagnostics": tel.diagnostics if tel else [],
                    "outcome": decision.recommendation, "expected_action": case["expected_action"],
                    "correct": decision.recommendation == {
                        "request_clarification": "Request clarification before review",
                        "hold_for_human_review": "Hold for required human review",
                        "proceed_to_human_approval": "Proceed to human approval",
                    }[case["expected_action"]],
                    "evidence_references": evidence_refs,
                    "evidence_grounded": grounded,
                    "deterministic_policy_safe": not any(f.startswith("deterministic ") for f in failures) and bool(decision.human_review_required),
                    "human_escalation_correct": decision.human_review_required,
                    "approvals": decision.required_approvals, "risks": decision.risk_flags,
                    "missing_information": decision.missing_information,
                    "failures": failures, "error": None,
                    "status": "pass" if not failures else "fail", "error_category": None,
                    "termination_reason": None, "missing_evidence_tool_names": [],
                }
            except KeyboardInterrupt as exc:
                row = _error_row(case, architecture, selected, config.model, started, stamp,
                                 exc, category="interrupted")
                interrupted = True
            except Exception as exc:
                row = _error_row(case, architecture, selected, config.model, started, stamp, exc)
            rows.append(row)
            print(f"{selected.upper():4} {architecture:6} {case['case_id']} {row['status'].upper()}"
                  f"{(' [' + row['error_category'] + ']') if row['error_category'] else ''} {row['latency_ms']:.0f}ms")
            # Preserve partial results if a later API request fails.
            write_results(selected, rows, config.model if selected == "real" else None, interrupted=interrupted,
                          selected_case_ids=[item["case_id"] for item in cases])
            if interrupted:
                break
        if interrupted:
            break
    return aggregate(rows, selected, config.model if selected == "real" else None, interrupted=interrupted)


def _error_row(case: dict[str, Any], architecture: str, mode: str, model: str,
               started: float, stamp: str, exc: BaseException, category: str | None = None) -> dict[str, Any]:
    telemetry = getattr(exc, "agent_telemetry", {})
    if category is None:
        category, termination_reason = classify_error(exc)
    else:
        termination_reason = None
    return {
        "timestamp_utc": stamp, "case_id": case["case_id"], "request_id": case["request_id"],
        "architecture": architecture, "execution_mode": mode, "model": model if mode == "real" else None,
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        "llm_calls": telemetry.get("llm_calls", 0), "tool_calls": telemetry.get("tool_calls", 0),
        "model_tool_calls": telemetry.get("model_tool_calls", 0),
        "tool_call_names": telemetry.get("tool_call_names", []),
        "tool_names": telemetry.get("tool_names", []),
        "diagnostics": telemetry.get("diagnostics", []),
        "missing_evidence_tool_names": getattr(exc, "missing_tools", []),
        "outcome": None, "expected_action": case["expected_action"], "correct": None,
        "evidence_references": [], "evidence_grounded": None, "deterministic_policy_safe": None,
        "human_escalation_correct": None, "approvals": [], "risks": [], "missing_information": [],
        "failures": [], "error": safe_error_text(exc, category), "status": "error",
        "error_category": category, "termination_reason": termination_reason,
    }


def aggregate(rows: list[dict[str, Any]], mode: str, model: str | None,
              interrupted: bool = False, selected_case_ids: list[str] | None = None) -> dict[str, Any]:
    groups = {}
    for arch in ("single", "staged"):
        current = [r for r in rows if r["architecture"] == arch]
        n = len(current)
        scored = [r for r in current if r["status"] in ("pass", "fail")]
        groups[arch] = {
            "cases": n, "pass_count": sum(r["status"] == "pass" for r in current),
            "fail_count": sum(r["status"] == "fail" for r in current),
            "error_count": sum(r["status"] == "error" for r in current),
            "api_error_count": sum(r["error_category"] in API_ERROR_CATEGORIES for r in current),
            "vendor_service_unavailable_count": sum("vendor_risk_unavailable" in (r.get("risks") or []) for r in current),
            "tool_loop_termination_count": sum(r["error_category"] == "tool_loop_terminated" for r in current),
            "model_response_error_count": sum(r["error_category"] == "malformed_response" for r in current),
            "incomplete_evidence_count": sum(r["error_category"] == "incomplete_evidence" for r in current),
            "model_refusal_count": sum(r["error_category"] == "model_refusal" for r in current),
            "model_quality_cases": len(scored),
            "recommendation_accuracy": round(sum(bool(r["correct"]) for r in scored) / len(scored), 4) if scored else None,
            "grounding_rate": round(sum(bool(r["evidence_grounded"]) for r in scored) / len(scored), 4) if scored else None,
            "deterministic_policy_compliance": round(sum(bool(r["deterministic_policy_safe"]) for r in scored) / len(scored), 4) if scored else None,
            "correct_human_escalation": round(sum(bool(r["human_escalation_correct"]) for r in scored) / len(scored), 4) if scored else None,
            "mean_latency_ms": round(sum(r["latency_ms"] for r in current) / n, 2) if n else None,
            "mean_llm_calls": round(sum(r["llm_calls"] for r in current) / n, 2) if n else None,
            "mean_tool_calls": round(sum(r["tool_calls"] for r in current) / n, 2) if n else None,
            "mean_model_tool_calls": round(sum(r.get("model_tool_calls", 0) for r in current) / n, 2) if n else None,
        }
    selected_ids = selected_case_ids or sorted({r["case_id"] for r in rows})
    return {"mode": mode, "model": model, "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "case_count_per_architecture": len({r['case_id'] for r in rows}), "interrupted": interrupted, "architectures": groups,
            "selected_case_ids": selected_ids,
            "processed_case_ids_by_architecture": {arch: sorted({r["case_id"] for r in rows if r["architecture"] == arch}) for arch in ("single", "staged")},
            "unprocessed_case_ids_by_architecture": {arch: sorted(set(selected_ids) - {r["case_id"] for r in rows if r["architecture"] == arch}) for arch in ("single", "staged")},
            "interrupted_case_id": next((r["case_id"] for r in rows if r.get("error_category") == "interrupted"), None),
            "results": rows, "interpretation": "Demo metrics are deterministic offline checks, not LLM performance." if mode == "demo" else "Live model results; inspect per-case errors and timestamps."}


def write_results(mode: str, rows: list[dict[str, Any]], model: str | None, interrupted: bool = False,
                  selected_case_ids: list[str] | None = None) -> None:
    prefix = "results_demo" if mode == "demo" else "results_real"
    result = aggregate(rows, mode, model, interrupted=interrupted, selected_case_ids=selected_case_ids)
    names = [prefix]
    if mode == "demo":
        # Keep legacy artifact paths current while labeling their mode in the content.
        names.append("results")
    list_fields = ("evidence_references", "approvals", "risks", "missing_information", "failures",
                   "tool_names", "tool_call_names", "diagnostics", "missing_evidence_tool_names")
    flat_rows = [{**row, **{field: json.dumps(row.get(field, []), ensure_ascii=False) for field in list_fields}}
                 for row in rows]
    summary = [f"# {'Deterministic demo' if mode == 'demo' else 'Real LLM'} comparison results", "",
               f"Mode: `{mode}`", f"Model: `{model or 'none'}`", "",
               "Demo figures are not measurements of real model performance." if mode == "demo" else "Results include live provider calls.", "",
               "| Metric | Single-agent | Staged two-agent |", "|---|---:|---:|"]
    metrics = [("Pass count", "pass_count"), ("Assertion-fail count", "fail_count"), ("Error count", "error_count"),
               ("Gemini API error count", "api_error_count"), ("Vendor service unavailable", "vendor_service_unavailable_count"),
               ("Malformed-response count", "model_response_error_count"), ("Incomplete evidence count", "incomplete_evidence_count"), ("Model refusal count", "model_refusal_count"),
               ("Recommendation accuracy (model-quality cases)", "recommendation_accuracy"), ("Grounding rate", "grounding_rate"),
               ("Policy compliance", "deterministic_policy_compliance"), ("Human escalation", "correct_human_escalation"),
               ("Mean latency (ms)", "mean_latency_ms"), ("Mean LLM calls", "mean_llm_calls"), ("Mean executed tool calls", "mean_tool_calls"), ("Mean model tool requests", "mean_model_tool_calls")]
    for label, key in metrics:
        summary.append(f"| {label} | {result['architectures']['single'][key]} | {result['architectures']['staged'][key]} |")
    for name in names:
        (ROOT / "evals" / f"{name}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        if flat_rows:
            with (ROOT / "evals" / f"{name}.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(flat_rows[0]))
                writer.writeheader()
                writer.writerows(flat_rows)
        (ROOT / "evals" / f"{name}_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare both procurement architectures serially on the same cases.")
    parser.add_argument("--mode", choices=["auto", "demo", "real"], default="real")
    parser.add_argument("--limit", type=int, help="Evaluate only the first N shared cases for both architectures.")
    parser.add_argument("--rate-limit-retries", type=int, help="Override bounded 429 retries (0 disables retries).")
    parser.add_argument("--request-pacing-seconds", type=float, help="Override delay between model request starts; 0 disables pacing.")
    parser.add_argument("--timeout-seconds", type=float, help="Override per-request Gemini timeout for this run.")
    args = parser.parse_args()
    try:
        if args.rate_limit_retries is not None: os.environ["GEMINI_RATE_LIMIT_RETRIES"] = str(args.rate_limit_retries)
        if args.request_pacing_seconds is not None: os.environ["GEMINI_REQUEST_PACING_SECONDS"] = str(args.request_pacing_seconds)
        if args.timeout_seconds is not None: os.environ["GEMINI_TIMEOUT_SECONDS"] = str(args.timeout_seconds)
        if args.limit is not None and args.limit < 1: raise ValueError("--limit must be a positive integer.")
        get_gemini_config()  # validate CLI overrides before any provider request
        result = run(args.mode, limit=args.limit)
    except MissingAPIKeyError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result["architectures"], indent=2))
    if result["interrupted"]: return 130
    return 1 if any(r["status"] != "pass" for r in result["results"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
