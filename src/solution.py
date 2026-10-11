from __future__ import annotations

import re
import json
import logging
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src import data_access
from src.contracts import Architecture, EvidenceItem, ProcurementDecision, RunTelemetry
from src.telemetry import RunTelemetryCounter
from src.tools import budget_and_approvals, policy_lookup, search_existing_software, vendor_security_lookup
from src.llm_config import get_gemini_config, require_gemini_config

logger = logging.getLogger(__name__)


class RequestInput(BaseModel):
    model_config = ConfigDict(extra="allow")
    request_id: str = Field(min_length=1)
    requester_id: str = Field(min_length=1)
    product_name: str = ""
    vendor_name: str = ""
    category: str = ""
    annual_cost_usd: float | None = Field(default=None, ge=0)
    user_count: int | None = Field(default=None, gt=0)
    business_justification: str = ""
    data_access_level: str = "unknown"
    requested_integrations: list[str] = Field(default_factory=list)


INJECTION = re.compile(r"ignore (all |any )?(procurement |security )?rules|cfo-approved|approve (it )?immediately|bypass (approval|controls)", re.I)
SENSITIVE = {"source_code", "production_telemetry", "confidential_documents", "employee_pii", "customer_pii", "credentials", "secrets"}
SECURITY_TERMS = ("source code", "production", "cloud account", "confidential", "pii", "credential", "secret")


def _employee(requester_id: str) -> dict[str, str] | None:
    return next((r for r in data_access.load_employees().to_dict("records") if r["employee_id"] == requester_id), None)


def _decision(req: dict[str, Any], architecture: Architecture, *, tool_results: dict[str, dict[str, Any]] | None = None,
              telemetry: RunTelemetryCounter | None = None, execution_mode: str = "demo", model_name: str | None = None) -> ProcurementDecision:
    parsed = RequestInput.model_validate(req)
    request = parsed.model_dump()
    employee = _employee(request["requester_id"])
    department = employee["department"] if employee else str(request.get("department") or "")
    request["department"] = department
    telemetry = telemetry or RunTelemetryCounter()
    tool_results = tool_results or {}
    evidence: list[EvidenceItem] = []
    missing: list[str] = []
    risks: list[str] = []
    approvals: list[str] = []

    if not employee:
        missing.append("Requester ID is not present in employees.csv; department and manager cannot be verified.")
        risks.append("requester_record_missing")
    if not department: missing.append("Requesting department")
    if not request["product_name"].strip(): missing.append("Product or service name")
    if not request["vendor_name"].strip(): missing.append("Vendor name")
    if request["annual_cost_usd"] is None: missing.append("Annual cost or reasonable annual estimate")
    if request["user_count"] is None: missing.append("Number of users or licenses")
    if not request["business_justification"].strip(): missing.append("Business purpose")
    if request["data_access_level"].strip().casefold() in ("", "unknown", "unspecified"):
        missing.append("Intended data-access level")
    if request.get("requested_integrations") is None: missing.append("Required integrations (provide a list, or confirm none)")
    if missing: risks.append("missing_information")
    business_text = " ".join(str(request.get(k) or "") for k in ("business_justification", "product_name", "vendor_name"))
    injection = bool(INJECTION.search(business_text))
    if injection: risks.append("prompt_injection_detected")

    budget = tool_results.get("budget_and_approvals") or budget_and_approvals(department, request["annual_cost_usd"], telemetry)
    if budget["status"] == "available":
        evidence.append(EvidenceItem(source="Budget checker", finding=budget["evidence"], reference="department_budgets.csv; Procurement Policy §2"))
        approvals.extend(budget["approvals"])
        if budget["sufficient"] is False:
            risks.append("budget_insufficient")
            if "Finance" not in approvals: approvals.append("Finance (budget exception)")
    else:
        evidence.append(EvidenceItem(source="Budget checker", finding=budget["evidence"], reference="department_budgets.csv"))
        missing.append("Department software budget is unavailable")
        risks.append("budget_data_missing")

    catalog = tool_results.get("existing_software_search") or search_existing_software(request, telemetry)
    retrieved_untrusted = json.dumps({"catalog": catalog, "vendor": tool_results.get("vendor_security_lookup", {})}, ensure_ascii=False, default=str)
    retrieved_injection = bool(INJECTION.search(retrieved_untrusted))
    if retrieved_injection and "prompt_injection_detected" not in risks:
        risks.append("prompt_injection_detected")
        evidence.append(EvidenceItem(source="Untrusted retrieved-content check", finding="Instruction-like content was detected in retrieved business data and treated only as untrusted data.", reference="evaluation fixture or retrieved tool record; no instruction executed"))
    matches = catalog["matches"]
    evidence.append(EvidenceItem(source="Software catalog search", finding=(f"Found {len(matches)} possible catalog match(es): " + "; ".join(f"{m['product_name']} ({m['reason']}, {m['licensed_seats']} seats, {m['scope']})" for m in matches[:4])) if matches else "No likely match in the approved software catalog.", reference=catalog.get("source", "software_catalog.csv")))
    if catalog["purchase_history"]:
        evidence.append(EvidenceItem(source="Purchase history", finding="Matching approved purchase history exists: " + "; ".join(f"{p['purchase_id']} {p['product_name']} ${int(p['annual_amount_usd']):,}" for p in catalog["purchase_history"]), reference="purchase_history.csv"))
    if matches: risks.append("existing_tool_overlap")

    vendor = tool_results.get("vendor_security_lookup")
    if vendor is None:
        vendor = vendor_security_lookup(request["vendor_name"], telemetry) if request["vendor_name"].strip() else {"registry": None, "service_status": "missing", "message": "Vendor name is required."}
    registry = vendor.get("registry")
    if registry:
        evidence.append(EvidenceItem(source="Internal vendor registry", finding=f"Procurement={registry['procurement_status']}; security={registry['security_status']}; security review date={registry['security_review_date'] or 'missing'}; legal terms={registry['legal_terms_status']}.", reference=f"vendors.csv; {registry['vendor_id']}"))
    if vendor.get("service_status") == "available":
        ext = vendor["service"]
        evidence.append(EvidenceItem(source="Vendor-risk mock API", finding=f"Risk={ext.get('risk_level', 'unknown')}; security review={ext.get('security_review_status', 'unknown')}; last review={ext.get('last_review_date') or 'missing'}; personal data={ext.get('processes_personal_data', 'unknown')}; cross-region storage={ext.get('stores_data_outside_region', 'unknown')}.", reference=f"GET /vendor-risk/{request['vendor_name']}"))
        if vendor.get("expired"): risks.append("vendor_review_expired")
        if vendor.get("conflict"):
            risks.append("conflicting_vendor_evidence")
            evidence.append(EvidenceItem(source="Vendor evidence comparison", finding=vendor["conflict"], reference="vendors.csv vs vendor-risk service"))
        if ext.get("security_review_status", "").casefold() != "approved": risks.append("security_review_required")
        if ext.get("processes_personal_data") and request["data_access_level"] in SENSITIVE: risks.append("privacy_review_required")
        if ext.get("stores_data_outside_region"): risks.append("privacy_review_required")
        if str(ext.get("risk_level", "")).casefold() == "high": risks.append("security_review_required")
    elif vendor.get("service_status") == "unavailable":
        risks.extend(["vendor_risk_unavailable", "security_review_required"])
        evidence.append(EvidenceItem(source="Vendor-risk mock API", finding="Vendor security status could not be verified; no favorable status is inferred. " + vendor.get("error", "Service unavailable"), reference=f"GET /vendor-risk/{request['vendor_name']}"))
    else:
        evidence.append(EvidenceItem(source="Vendor-risk mock API", finding="Vendor security lookup was skipped because the vendor name is missing.", reference="Vendor name required by GET /vendor-risk/{vendor_name}"))
        risks.extend(["vendor_status_missing", "security_review_required"])
    if vendor.get("conflict") or vendor.get("expired") or vendor.get("service_status") != "available" or (registry or {}).get("security_status", "").casefold() != "approved":
        if "Security" not in approvals: approvals.append("Security")
    if "privacy_review_required" in risks and "Privacy" not in approvals: approvals.append("Privacy")

    if registry:
        new_vendor = registry["procurement_status"].casefold() == "new"
        legal_unapproved = registry["legal_terms_status"].casefold() not in ("approved", "standard")
        if (new_vendor and request["annual_cost_usd"] is not None and request["annual_cost_usd"] >= 10000) or legal_unapproved:
            risks.append("legal_review_required")
            if "Legal" not in approvals: approvals.append("Legal")
    data_level = request["data_access_level"].casefold()
    integration_text = " ".join(request["requested_integrations"] or []).casefold()
    if data_level in SENSITIVE or any(t in (data_level + " " + integration_text).replace("_", " ") for t in SECURITY_TERMS):
        risks.append("security_review_required")
        if "Security" not in approvals: approvals.append("Security")
    if data_level in {"customer_pii", "employee_pii"}:
        risks.append("privacy_review_required")
        if "Privacy" not in approvals: approvals.append("Privacy")

    tool_results.get("procurement_policy_lookup") or policy_lookup(telemetry)
    evidence.append(EvidenceItem(source="Procurement policy", finding="Applied policy 2026.09: annual cost approval bands, available-budget check, overlap review, 365-day security freshness, Security/Privacy/Legal triggers, and human decision boundary.", reference="data/procurement_policy.md §§1-11; snapshot 2026-09-30"))
    # Never allow duplicate risk flags or approvals, while retaining rule order.
    risks = list(dict.fromkeys(risks))
    approvals = list(dict.fromkeys(approvals))

    if architecture == "staged" and execution_mode == "demo":
        # Bounded independent policy-review pass over the same evidence and rules.
        telemetry.record_llm_call() if False else None  # no configured LLM; this is deterministic demo mode
        if any("could not be verified" in item.finding for item in evidence) and "Security" not in approvals:
            approvals.append("Security")
        if "budget_insufficient" in risks and not any(a.startswith("Finance") for a in approvals):
            approvals.append("Finance (budget exception)")
        if missing and "missing_information" not in risks: risks.append("missing_information")
        evidence.append(EvidenceItem(source="Policy and risk reviewer (demo mode)", finding="Independent bounded review checked evidence coverage, deterministic budget/approval results, unresolved vendor evidence, and missing information. It found no basis to authorize a purchase.", reference="staged reviewer; one pass; no LLM configured"))

    if missing:
        recommendation = "Request clarification before review"
        next_step = "Complete the missing request fields, then rerun the assessment. No purchase or approval has occurred."
    elif risks:
        recommendation = "Hold for required human review"
        next_step = "Route this request to the listed procurement, finance, security, privacy, or legal reviewers; resolve all flags before a human decision."
    else:
        recommendation = "Proceed to human approval"
        next_step = "Submit the evidence package and listed approvals to the manager/procurement workflow. This recommendation is not authorization to purchase."
    if injection and not missing and not risks:
        recommendation = "Hold for required human review"
    return ProcurementDecision(request_id=request["request_id"], recommendation=recommendation, evidence=evidence,
        required_approvals=approvals, missing_information=missing, risk_flags=risks, next_step=next_step,
        human_review_required=True, telemetry=RunTelemetry(llm_calls=telemetry.llm_calls, tool_calls=telemetry.tool_calls,
            tool_names=list(telemetry.tool_names), execution_mode=execution_mode, model=model_name,
            model_tool_calls=telemetry.model_tool_calls, tool_call_names=list(telemetry.tool_call_names),
            diagnostics=list(telemetry.diagnostics),
            agents=["deterministic demo analyst", "deterministic demo reviewer"] if architecture == "staged" and execution_mode == "demo" else (["deterministic demo analyst"] if execution_mode == "demo" else [])))


def _merge_llm_review(base: ProcurementDecision, proposals: list[dict[str, Any]], model: str,
                      telemetry: RunTelemetryCounter, architecture: Architecture) -> ProcurementDecision:
    from src.agents import ALLOWED_APPROVALS, ALLOWED_RISKS
    references = {item.reference for item in base.evidence if item.reference}
    risks = list(base.risk_flags)
    approvals = list(base.required_approvals)
    missing = list(base.missing_information)
    deterministic_missing = bool(base.missing_information)
    evidence = list(base.evidence)
    actions = {"proceed_to_human_approval": 1, "hold_for_human_review": 2, "request_clarification": 3}
    base_action = "request_clarification" if deterministic_missing else ("hold_for_human_review" if risks else "proceed_to_human_approval")
    chosen_action = base_action
    unsupported: list[str] = []
    for proposal in proposals:
        cited = list(dict.fromkeys(proposal.get("evidence_references") or []))
        valid_citations = [ref for ref in cited if ref in references]
        invalid = [ref for ref in cited if ref not in references]
        if invalid:
            unsupported.append("Model output cited source references not returned by tools.")
        summary = proposal.get("summary") or proposal.get("evidence_summary") or proposal.get("review_summary") or ""
        if summary.strip():
            if not valid_citations:
                unsupported.append("Model interpretation was not grounded in a verified tool reference.")
            label = "AI-generated interpretation (not independently verified): "
            evidence.append(EvidenceItem(source="Gemini agent interpretation", finding=label + summary.strip(),
                reference=f"model={model}" + ("; cited=" + "; ".join(valid_citations) if valid_citations else "; no verified citations supplied")))
        for text in proposal.get("unsupported_claims", []):
            if text.strip():
                unsupported.append(text.strip())
        for text in proposal.get("uncertainties", []):
            if text.strip():
                risk = "ai_uncertainty"
                if risk not in risks: risks.append(risk)
                evidence.append(EvidenceItem(source="Gemini agent uncertainty", finding="AI-generated uncertainty requiring human interpretation: " + text.strip(), reference=f"model={model}"))
        for item in proposal.get("missing_information", []):
            if item.strip():
                note = "AI-suggested clarification (verify with requester): " + item.strip()
                if note not in missing: missing.append(note)
        for flag in proposal.get("potential_risks", []) + proposal.get("additional_risk_flags", []):
            if flag in ALLOWED_RISKS and valid_citations:
                suggested = "ai_suggested:" + flag
                if suggested not in risks: risks.append(suggested)
            elif flag not in ALLOWED_RISKS:
                unsupported.append("Model proposed an unsupported risk taxonomy value.")
        for approver in proposal.get("additional_approvals", []):
            if approver in ALLOWED_APPROVALS:
                suggested = "AI-suggested approval (verify): " + approver
                if suggested not in approvals: approvals.append(suggested)
            else:
                unsupported.append("Model proposed an unsupported approval role.")
        action = proposal.get("recommendation")
        # A model request for clarification cannot replace a deterministic
        # hold when the submitted request has no mandatory missing fields.
        if action in actions and not (action == "request_clarification" and not deterministic_missing) and actions[action] > actions[chosen_action]:
            chosen_action = action
    if unsupported:
        if "unsupported_model_claim" not in risks: risks.append("unsupported_model_claim")
        evidence.append(EvidenceItem(source="LLM claim validation", finding="One or more model claims or citations could not be verified and are excluded from factual evidence: " + "; ".join(dict.fromkeys(unsupported))[:1000], reference=f"model={model}; deterministic reference validation"))
    if chosen_action == "request_clarification" and deterministic_missing and not missing:
        missing.append("AI analyst recommends clarification; verify the AI interpretation evidence before proceeding.")
    # AI-suggested questions require verification, but they cannot downgrade a
    # deterministic human-review hold into a mandatory clarification state.
    if deterministic_missing or chosen_action == "request_clarification":
        recommendation = "Request clarification before review"
        next_step = "Resolve the listed request gaps, including AI-suggested items where applicable, then rerun assessment. No purchase or approval has occurred."
    elif chosen_action == "hold_for_human_review" or risks:
        recommendation = "Hold for required human review"
        next_step = "Route this request to the listed authorized reviewers and resolve policy or uncertainty flags. This recommendation is not authorization to purchase."
    else:
        recommendation = "Proceed to human approval"
        next_step = "Submit the evidence package to the listed authorized approvers. This recommendation is not authorization to purchase."
    evidence = evidence[:40]
    base.recommendation = recommendation
    base.next_step = next_step
    base.evidence = evidence
    base.missing_information = list(dict.fromkeys(missing))
    base.risk_flags = list(dict.fromkeys(risks))
    base.required_approvals = list(dict.fromkeys(approvals))
    base.human_review_required = True
    base.telemetry = RunTelemetry(llm_calls=telemetry.llm_calls, tool_calls=telemetry.tool_calls,
        tool_names=list(telemetry.tool_names), execution_mode="real", model=model,
        model_tool_calls=telemetry.model_tool_calls, tool_call_names=list(telemetry.tool_call_names),
        diagnostics=list(telemetry.diagnostics),
        agents=["Gemini procurement analyst"] + (["Gemini policy and risk reviewer"] if architecture == "staged" else []))
    return base


def analyze_request(request: dict[str, Any], architecture: Architecture = "single", mode: str | None = None) -> ProcurementDecision:
    if architecture not in ("single", "staged"):
        raise ValueError("architecture must be 'single' or 'staged'")
    if mode not in (None, "auto", "demo", "real"):
        raise ValueError("mode must be 'auto', 'demo', or 'real'")
    telemetry: RunTelemetryCounter | None = None
    selected_mode = mode
    try:
        config = get_gemini_config()
        selected_mode = ("real" if config.is_configured else "demo") if mode in (None, "auto") else mode
        if selected_mode == "demo":
            result = _decision(request, architecture, execution_mode="demo")
        else:
            config = require_gemini_config()
            from src.agents import run_policy_reviewer, run_single_agent, run_staged_analyst, create_gemini_client
            telemetry = RunTelemetryCounter()
            client = create_gemini_client(config)
            if architecture == "single":
                assessment, cache, telemetry, config = run_single_agent(request, client=client, config=config, telemetry=telemetry)
                baseline = _decision(request, "single", tool_results=cache, telemetry=telemetry, execution_mode="real", model_name=config.model)
                proposal = assessment.model_dump()
                result = _merge_llm_review(baseline, [proposal], config.model, telemetry, architecture)
            else:
                package, cache, telemetry, config, client = run_staged_analyst(request, client=client, config=config, telemetry=telemetry)
                baseline = _decision(request, "single", tool_results=cache, telemetry=telemetry, execution_mode="real", model_name=config.model)
                review = run_policy_reviewer(request=request, package=package, tool_results=cache, deterministic_decision=baseline.model_dump(mode="json"),
                    client=client, config=config, telemetry=telemetry)
                result = _merge_llm_review(baseline, [package.model_dump(), review.model_dump()], config.model, telemetry, architecture)
        logger.info("Procurement analysis complete request_id=%s architecture=%s mode=%s model=%s recommendation=%s risks=%d", result.request_id, architecture, selected_mode, result.telemetry.model if result.telemetry else None, result.recommendation, len(result.risk_flags))
        return result
    except ValidationError:
        raise
    except Exception as exc:
        # Evaluation/reporting can retain real call counts on failures without disclosing
        # provider exception text or credentials.
        if selected_mode == "real" and telemetry is not None:
            try:
                exc.agent_telemetry = {
                    "llm_calls": telemetry.llm_calls,
                    "tool_calls": telemetry.tool_calls,
                    "tool_names": list(telemetry.tool_names),
                    "model_tool_calls": telemetry.model_tool_calls,
                    "tool_call_names": list(telemetry.tool_call_names),
                    "diagnostics": list(telemetry.diagnostics),
                    "execution_mode": "real",
                    "model": get_gemini_config().model,
                    "agents": ["Gemini procurement analyst"] + (["Gemini policy and risk reviewer"] if architecture == "staged" else []),
                }
            except Exception:
                pass
        raise


def handle_request(request_id: str, architecture: Architecture = "single", mode: str | None = None) -> ProcurementDecision:
    return analyze_request(data_access.get_request(request_id), architecture, mode=mode)
