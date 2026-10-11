from __future__ import annotations

import json
import os
import unittest
from typing import Any
from types import SimpleNamespace
from unittest.mock import patch

from src.agents import AgentTimeoutError, ToolArgumentError, ToolBudgetExceeded, execute_tool, run_single_agent
from src.llm_config import DEFAULT_MODEL, GeminiConfig, MissingAPIKeyError, get_gemini_config, require_gemini_config
from src.solution import _merge_llm_review, analyze_request, handle_request
from src.telemetry import RunTelemetryCounter


def function_call(name: str, arguments: str, call_id: str):
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=arguments))


def response(message: Any, finish_reason="stop"):
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=finish_reason)])


class FakeChatCompletions:
    def __init__(self, final_json: str, department: str, cost: float, vendor: str):
        self.calls = []
        self.final_json = final_json
        self.department, self.cost, self.vendor = department, cost, vendor

    def create(self, **kwargs):
        self.calls.append(kwargs)
        n = len(self.calls)
        if n <= 4:
            names = [tool["function"]["name"] for tool in kwargs["tools"]]
            name = names[n - 1]
            args = {
                "budget_and_approvals": json.dumps({"department": self.department, "annual_cost_usd": self.cost}),
                "existing_software_search": "{}",
                "vendor_security_lookup": json.dumps({"vendor_name": self.vendor}),
                "procurement_policy_lookup": "{}",
            }[name]
            call = function_call(name, args, f"call-{n}")
            assistant = SimpleNamespace(tool_calls=[call], content=None, refusal=None,
                model_dump=lambda exclude_none=True: {"role":"assistant", "tool_calls":[]})
            return response(assistant, "tool_calls")
        final = SimpleNamespace(tool_calls=None, content=self.final_json, refusal=None)
        return response(final)


class AgentTests(unittest.TestCase):
    def setUp(self):
        from src import data_access
        from src.solution import _employee
        self.request = data_access.get_request("REQ-1001")
        self.department = _employee(self.request["requester_id"])["department"]
        self.config = GeminiConfig("mocked-test-key", "gemini-3.1-flash-lite", 10, request_pacing_seconds=0)

    def assessment(self):
        return json.dumps({"recommendation":"hold_for_human_review", "summary":"SignFlow may overlap an existing tool; keep human review.",
            "evidence_references":["software_catalog.csv"], "missing_information":[], "additional_risk_flags":[], "additional_approvals":[]})

    @staticmethod
    def unavailable_vendor(name, telemetry):
        telemetry.record_tool_call("vendor_security_lookup")
        return {"service_status":"unavailable", "error":"offline"}

    def _patch_real(self, client):
        return (
            patch("src.solution.require_gemini_config", return_value=self.config),
            patch("src.agents.require_gemini_config", return_value=self.config),
            patch("src.agents.create_gemini_client", return_value=client),
            patch("src.agents.vendor_security_lookup", side_effect=self.unavailable_vendor),
        )

    def test_single_agent_uses_real_style_tool_calls_and_structured_response(self):
        client = SimpleNamespace(chat=SimpleNamespace(completions=FakeChatCompletions(
            self.assessment(), self.department, self.request["annual_cost_usd"], self.request["vendor_name"])))
        patches = self._patch_real(client)
        with patches[0], patches[1], patches[2], patches[3]:
            result = analyze_request(self.request, mode="real")
        self.assertEqual(result.telemetry.execution_mode, "real")
        self.assertEqual(result.telemetry.llm_calls, 5)
        self.assertEqual(result.telemetry.tool_calls, 4)
        self.assertIn("vendor_risk_unavailable", result.risk_flags)
        self.assertTrue(result.human_review_required)
        self.assertTrue(any("AI-generated interpretation" in e.finding for e in result.evidence))

    def test_staged_runs_analyst_and_independent_reviewer(self):
        analyst = json.dumps({"business_need":"Digital signature workflow", "evidence_summary":"Catalog evidence reviewed.",
            "evidence_references":["software_catalog.csv"], "missing_information":[], "uncertainties":[], "potential_risks":[]})
        reviewer = json.dumps({"recommendation":"hold_for_human_review", "review_summary":"Overlap warrants procurement review.",
            "evidence_references":["software_catalog.csv"], "unsupported_claims":[], "missing_information":[],
            "additional_risk_flags":[], "additional_approvals":[]})
        class Staged:
            def __init__(inner):
                inner.calls = []
                inner.base = FakeChatCompletions(analyst, self.department, self.request["annual_cost_usd"], self.request["vendor_name"])
            def create(inner, **kwargs):
                inner.calls.append(kwargs)
                n = len(inner.calls)
                if n <= 4:
                    return inner.base.create(**kwargs)
                value = analyst if n == 5 else reviewer
                return response(SimpleNamespace(tool_calls=None, content=value, refusal=None))
        client = SimpleNamespace(chat=SimpleNamespace(completions=Staged()))
        patches = self._patch_real(client)
        with patches[0], patches[1], patches[2], patches[3]:
            result = analyze_request(self.request, architecture="staged", mode="real")
        self.assertEqual(result.telemetry.llm_calls, 6)
        self.assertEqual(result.telemetry.tool_calls, 4)
        self.assertEqual(len(result.telemetry.agents), 2)
        self.assertIn("vendor_risk_unavailable", result.risk_flags)
        self.assertIn("Security", result.required_approvals)
        self.assertTrue(result.human_review_required)

    def test_tool_rejects_altered_verified_department(self):
        with self.assertRaises(ToolArgumentError):
            execute_tool("budget_and_approvals", '{"department":"Sales","annual_cost_usd":950}',
                {"department":"Finance","annual_cost_usd":950}, RunTelemetryCounter(), {})

    def test_model_cannot_remove_deterministic_approvals_or_risks(self):
        base = handle_request("REQ-1004", mode="demo")
        required_before = list(base.required_approvals)
        risks_before = list(base.risk_flags)
        merged = _merge_llm_review(base, [{"recommendation":"proceed_to_human_approval", "summary":"Looks safe.",
            "evidence_references":[], "missing_information":[], "additional_risk_flags":[], "additional_approvals":[]}],
            "mock-model", RunTelemetryCounter(llm_calls=1, tool_calls=4), "single")
        self.assertEqual(merged.required_approvals[:len(required_before)], required_before)
        self.assertEqual(merged.risk_flags[:len(risks_before)], risks_before)
        self.assertEqual(merged.recommendation, "Hold for required human review")
        self.assertTrue(merged.human_review_required)
        self.assertIn("unsupported_model_claim", merged.risk_flags)

    def test_ai_suggested_question_does_not_downgrade_deterministic_hold(self):
        from evals.run_comparison import evaluate_decision
        from src.solution import _decision
        base = _decision(self.request, "single")
        proposal = {"recommendation":"request_clarification", "summary":"Existing company-wide license may overlap.",
            "evidence_references":["software_catalog.csv"],
            "missing_information":["Verify unused seats on the current agreement."],
            "additional_risk_flags":[], "additional_approvals":[]}
        merged = _merge_llm_review(base, [proposal], "mock-model", RunTelemetryCounter(llm_calls=2), "single")
        self.assertEqual(merged.recommendation, "Hold for required human review")
        self.assertTrue(any(x.startswith("AI-suggested clarification") for x in merged.missing_information))
        case = {"expected_action":"hold_for_human_review", "required_approvals":["Manager"],
                "required_risks":["existing_tool_overlap"]}
        self.assertEqual(evaluate_decision(merged, case), [])

    def test_completed_evidence_forces_final_answer_without_more_tools(self):
        names_and_args = [
            ("budget_and_approvals", json.dumps({"department":self.department,"annual_cost_usd":self.request["annual_cost_usd"]})),
            ("existing_software_search", "{}"),
            ("vendor_security_lookup", json.dumps({"vendor_name":self.request["vendor_name"]})),
            ("procurement_policy_lookup", "{}"),
        ]
        class CompleteEvidence:
            calls = []
            def create(inner, **kwargs):
                inner.calls.append(kwargs)
                if len(inner.calls) == 1:
                    calls = [function_call(name,args,f"id-{i}") for i,(name,args) in enumerate(names_and_args)]
                    assistant = SimpleNamespace(tool_calls=calls, content=None, refusal=None,
                        model_dump=lambda exclude_none=True: {"role":"assistant","tool_calls":[]})
                    return response(assistant, "tool_calls")
                return response(SimpleNamespace(tool_calls=None, content=self.assessment(), refusal=None))
        fake = CompleteEvidence()
        client = SimpleNamespace(chat=SimpleNamespace(completions=fake))
        with patch("src.agents.vendor_security_lookup", side_effect=self.unavailable_vendor):
            assessment, _, telemetry, _ = run_single_agent(self.request, client=client, config=self.config)
        self.assertEqual(assessment.recommendation, "hold_for_human_review")
        self.assertEqual(fake.calls[1]["tool_choice"], "none")
        self.assertEqual(telemetry.model_tool_calls, 4)
        self.assertEqual([event["event"] for event in telemetry.diagnostics].count("model_response"), 2)

    def test_missing_evidence_gets_one_bounded_repair_before_incomplete_error(self):
        from src.agents import AgentEvidenceIncompleteError
        names_and_args = [
            ("budget_and_approvals", json.dumps({"department":self.department,"annual_cost_usd":self.request["annual_cost_usd"]})),
            ("existing_software_search", "{}"),
            ("vendor_security_lookup", json.dumps({"vendor_name":self.request["vendor_name"]})),
        ]
        class StopsEarly:
            calls = []
            def create(inner, **kwargs):
                inner.calls.append(kwargs)
                if len(inner.calls) == 1:
                    calls = [function_call(name,args,f"id-{i}") for i,(name,args) in enumerate(names_and_args)]
                    return response(SimpleNamespace(tool_calls=calls, content=None, refusal=None,
                        model_dump=lambda exclude_none=True: {"role":"assistant","tool_calls":[]}), "tool_calls")
                return response(SimpleNamespace(tool_calls=None, content=self.assessment(), refusal=None))
        fake = StopsEarly()
        client = SimpleNamespace(chat=SimpleNamespace(completions=fake))
        with patch("src.agents.vendor_security_lookup", side_effect=self.unavailable_vendor):
            with self.assertRaises(AgentEvidenceIncompleteError) as caught:
                run_single_agent(self.request, client=client, config=self.config)
        self.assertEqual(len(fake.calls), 3)  # initial evidence, one repair, then rejected incomplete final
        self.assertIn("procurement_policy_lookup", caught.exception.missing_tools)
        self.assertEqual(caught.exception.agent_telemetry["diagnostics"][-1]["event"], "evidence_incomplete")

    def test_real_evaluation_requires_key(self):
        from evals.run_comparison import resolve_mode
        with patch("evals.run_comparison.require_gemini_config", side_effect=MissingAPIKeyError("missing")):
            with self.assertRaises(MissingAPIKeyError):
                resolve_mode("real")

    def test_provider_config_default_and_missing_key(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY":"", "GEMINI_MODEL":"", "GEMINI_TIMEOUT_SECONDS":"45"}, clear=False):
            config = get_gemini_config()
        self.assertFalse(config.is_configured)
        self.assertEqual(config.model, DEFAULT_MODEL)
        with patch.dict(os.environ, {"GEMINI_API_KEY":""}, clear=False):
            with self.assertRaises(MissingAPIKeyError):
                require_gemini_config()

    def test_provider_config_override_and_timeout_validation(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY":"test-only", "GEMINI_MODEL":"model-override", "GEMINI_TIMEOUT_SECONDS":"12"}, clear=False):
            config = get_gemini_config()
        self.assertEqual((config.model, config.timeout_seconds), ("model-override", 12.0))
        with patch.dict(os.environ, {"GEMINI_TIMEOUT_SECONDS":"0"}, clear=False):
            with self.assertRaises(ValueError):
                get_gemini_config()

    def test_provider_failure_does_not_fall_back(self):
        class Failed:
            def create(self, **kwargs):
                raise TimeoutError("provider timeout")
        for architecture in ("single", "staged"):
            with self.subTest(architecture=architecture):
                client = SimpleNamespace(chat=SimpleNamespace(completions=Failed()))
                with patch("src.solution.require_gemini_config", return_value=self.config), \
                     patch("src.agents.create_gemini_client", return_value=client):
                    with self.assertRaises(AgentTimeoutError) as caught:
                        analyze_request(self.request, architecture=architecture, mode="real")
                telemetry = caught.exception.agent_telemetry
                self.assertEqual(telemetry["execution_mode"], "real")
                self.assertEqual(telemetry["llm_calls"], 1)
                self.assertEqual(telemetry["tool_calls"], 0)

    def test_staged_reviewer_provider_failure_propagates_without_demo_fallback(self):
        analyst_json = json.dumps({"business_need":"Signature workflow", "evidence_summary":"Sources reviewed.",
            "evidence_references":["software_catalog.csv"], "missing_information":[], "uncertainties":[], "potential_risks":[]})

        class FailedReviewer:
            def __init__(inner):
                inner.calls = 0
                inner.analyst = FakeChatCompletions(analyst_json, self.department,
                    self.request["annual_cost_usd"], self.request["vendor_name"])

            def create(inner, **kwargs):
                inner.calls += 1
                if inner.calls <= 5:  # four tool turns, then the analyst's evidence package
                    return inner.analyst.create(**kwargs)
                raise TimeoutError("reviewer provider timeout")

        client = SimpleNamespace(chat=SimpleNamespace(completions=FailedReviewer()))
        with patch("src.solution.require_gemini_config", return_value=self.config), \
             patch("src.agents.create_gemini_client", return_value=client), \
             patch("src.agents.vendor_security_lookup", side_effect=self.unavailable_vendor):
            with self.assertRaises(AgentTimeoutError) as caught:
                analyze_request(self.request, architecture="staged", mode="real")
        telemetry = caught.exception.agent_telemetry
        self.assertEqual(telemetry["execution_mode"], "real")
        self.assertEqual(telemetry["llm_calls"], 6)
        self.assertEqual(telemetry["tool_calls"], 4)

    def test_tool_call_loop_is_bounded(self):
        class Repeating:
            def __init__(inner): inner.calls = 0
            def create(inner, **kwargs):
                inner.calls += 1
                call = function_call("vendor_security_lookup", '{"vendor_name":"wrong vendor"}', str(inner.calls))
                assistant = SimpleNamespace(tool_calls=[call], content=None, refusal=None,
                    model_dump=lambda exclude_none=True: {"role":"assistant", "tool_calls":[]})
                return response(assistant, "tool_calls")
        client = SimpleNamespace(chat=SimpleNamespace(completions=Repeating()))
        with self.assertRaises(ToolBudgetExceeded) as caught:
            run_single_agent(self.request, client=client, config=self.config, telemetry=RunTelemetryCounter())
        self.assertEqual(caught.exception.termination_reason, "max_tool_calls")


if __name__ == "__main__":
    unittest.main()
