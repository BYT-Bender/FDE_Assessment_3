"""Small independently callable procurement tools backed by the supplied snapshot."""
from __future__ import annotations

import csv
import logging
import re
from datetime import date
from pathlib import Path
from typing import Any

from src import data_access
from src.telemetry import RunTelemetryCounter
from src.vendor_client import get_vendor_risk

REFERENCE_DATE = date(2026, 9, 30)
logger = logging.getLogger(__name__)


def _rows(name: str) -> list[dict[str, str]]:
    path = data_access.DATA_DIR / name
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def budget_and_approvals(department: str, annual_cost_usd: float | None, telemetry: RunTelemetryCounter | None = None) -> dict[str, Any]:
    if telemetry: telemetry.record_tool_call("budget_and_approvals")
    rows = [r for r in _rows("department_budgets.csv") if r["department"].casefold() == department.casefold()]
    if not rows:
        return {"status": "missing", "department": department, "available_usd": None, "approvals": [], "evidence": "No matching department budget record."}
    row = rows[0]
    available = int(row["available_usd"])
    approvals: list[str] = []
    if annual_cost_usd is not None:
        if annual_cost_usd <= 1000: approvals = ["Manager"]
        elif annual_cost_usd <= 10000: approvals = ["Department Head", "Procurement"]
        elif annual_cost_usd <= 25000: approvals = ["Department Head", "Finance", "Procurement"]
        else: approvals = ["Department Head", "Finance", "CFO", "Procurement"]
    return {"status": "available", "department": row["department"], "annual_budget_usd": int(row["annual_software_budget_usd"]),
            "committed_usd": int(row["committed_usd"]), "available_usd": available,
            "sufficient": None if annual_cost_usd is None else annual_cost_usd <= available,
            "approvals": approvals, "evidence": f"Available software budget ${available:,.0f} = annual budget ${int(row['annual_software_budget_usd']):,.0f} less committed ${int(row['committed_usd']):,.0f}.",
            "policy": "Procurement Policy §§2, 4"}


def search_existing_software(request: dict[str, Any], telemetry: RunTelemetryCounter | None = None) -> dict[str, Any]:
    if telemetry: telemetry.record_tool_call("existing_software_search")
    terms = set(re.findall(r"[a-z0-9]+", " ".join(str(request.get(k) or "") for k in ("product_name", "category", "business_justification" )).casefold()))
    matches = []
    for row in _rows("software_catalog.csv"):
        text = " ".join(row.values()).casefold()
        overlap = terms & set(re.findall(r"[a-z0-9]+", text))
        same_category = bool(request.get("category")) and row["category"].casefold() == str(request["category"]).casefold()
        exact_vendor = row["vendor_name"].casefold() == str(request.get("vendor_name", "")).casefold()
        if same_category or exact_vendor or len(overlap) >= 2:
            matches.append({"software_id": row["software_id"], "product_name": row["product_name"], "vendor_name": row["vendor_name"],
                            "category": row["category"], "status": row["status"], "licensed_seats": int(row["licensed_seats"]),
                            "scope": row["scope"], "reason": "same category" if same_category else "same vendor" if exact_vendor else "description/capability term match"})
    history = [r for r in _rows("purchase_history.csv") if r["department"].casefold() == str(request.get("department", "")).casefold()
               and r["status"].casefold() == "approved"
               and (r["vendor_name"].casefold() == str(request.get("vendor_name", "")).casefold()
                    or r["product_name"].casefold() == str(request.get("product_name", "")).casefold())]
    return {"matches": matches, "purchase_history": history, "status": "available", "source": "software_catalog.csv; purchase_history.csv"}


def vendor_security_lookup(vendor_name: str, telemetry: RunTelemetryCounter | None = None) -> dict[str, Any]:
    if telemetry: telemetry.record_tool_call("vendor_security_lookup")
    if not vendor_name:
        return {"status": "missing", "message": "Vendor name is required."}
    registry = next((r for r in _rows("vendors.csv") if r["vendor_name"].casefold() == vendor_name.casefold()), None)
    result: dict[str, Any] = {"registry": registry, "registry_source": "vendors.csv"}
    try:
        external = get_vendor_risk(vendor_name, timeout_seconds=2.0)
        result.update({"service": external, "service_status": "available"})
        if registry and registry.get("security_status") and external.get("security_review_status"):
            reg = registry["security_status"].casefold()
            ext = external["security_review_status"].casefold()
            # Registry vocabulary uses Pending where the risk service uses not_completed.
            normalized = {"pending": "not_completed", "approved": "approved", "unknown": "unknown"}
            if normalized.get(reg, reg) != normalized.get(ext, ext):
                result["conflict"] = f"Registry says {registry['security_status']}; risk service says {external['security_review_status']}."
        review_date = external.get("last_review_date")
        if review_date:
            age = (REFERENCE_DATE - date.fromisoformat(review_date)).days
            result["review_age_days"] = age
            result["expired"] = age > 365 or external.get("security_review_status", "").casefold() != "approved"
        else:
            result["expired"] = True
    except Exception as exc:
        logger.warning("Vendor risk lookup failed for %s: %s", vendor_name, type(exc).__name__)
        result.update({"service_status": "unavailable", "error": f"{type(exc).__name__}: {exc}", "expired": True})
    return result


def policy_lookup(telemetry: RunTelemetryCounter | None = None) -> dict[str, Any]:
    if telemetry: telemetry.record_tool_call("procurement_policy_lookup")
    return {"status": "available", "source": "data/procurement_policy.md", "version": "2026.09", "reference_date": "2026-09-30",
            "text": data_access.load_policy_text()}
