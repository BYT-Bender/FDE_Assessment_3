from __future__ import annotations

import unittest
import csv
import json
import tempfile
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

from evals import run_comparison
from src.agents import (AgentRateLimitError, AgentTimeoutError, ToolBudgetExceeded,
                        AgentAuthenticationError, AgentOutputError, call_gemini)
from src.llm_config import GeminiConfig
from src.telemetry import RunTelemetryCounter


class RetryAndTimeoutTests(unittest.TestCase):
    def setUp(self):
        self.config = GeminiConfig("test-only", "test-model", 10,
                                   request_pacing_seconds=0,
                                   rate_limit_retries=1,
                                   retry_after_cap_seconds=2)

    def test_429_respects_retry_after_with_bounded_cap(self):
        class RateLimitError(Exception):
            status_code = 429
            response = SimpleNamespace(headers={"Retry-After": "9"})

        class Completions:
            calls = 0
            def create(inner, **kwargs):
                inner.calls += 1
                if inner.calls == 1:
                    raise RateLimitError()
                return "ok"

        client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
        telemetry = RunTelemetryCounter()
        with patch("src.agents._pace_model_request"), patch("src.agents.time.sleep") as sleep:
            result = call_gemini(client, config=self.config, telemetry=telemetry, model="test-model")
        self.assertEqual(result, "ok")
        self.assertEqual(client.chat.completions.calls, 2)
        self.assertEqual(telemetry.llm_calls, 2)
        sleep.assert_called_once_with(2)

    def test_429_retries_are_capped_and_classified(self):
        class RateLimitError(Exception):
            status_code = 429
        class Completions:
            calls = 0
            def create(inner, **kwargs):
                inner.calls += 1
                raise RateLimitError()
        client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
        telemetry = RunTelemetryCounter()
        with patch("src.agents._pace_model_request"), patch("src.agents.time.sleep"):
            with self.assertRaises(AgentRateLimitError):
                call_gemini(client, config=self.config, telemetry=telemetry)
        self.assertEqual(client.chat.completions.calls, 2)
        self.assertEqual(telemetry.llm_calls, 2)

    def test_timeout_is_classified_without_retry(self):
        class Completions:
            calls = 0
            def create(inner, **kwargs):
                inner.calls += 1
                raise TimeoutError("simulated timeout")
        client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
        telemetry = RunTelemetryCounter()
        with patch("src.agents._pace_model_request"):
            with self.assertRaises(AgentTimeoutError):
                call_gemini(client, config=self.config, telemetry=telemetry)
        self.assertEqual(client.chat.completions.calls, 1)

    def test_pacing_reserves_request_start_spacing(self):
        import src.agents as agents
        config = GeminiConfig("test-only", "test-model", 10, request_pacing_seconds=2)
        agents._next_request_at = 0
        with patch("src.agents.time.monotonic", side_effect=[10, 10]), \
             patch("src.agents.time.sleep") as sleep:
            agents._pace_model_request(config)
            agents._pace_model_request(config)
        sleep.assert_called_once_with(2)
        agents._next_request_at = 0

    def test_zero_retries_disables_429_retry(self):
        config = GeminiConfig("test-only", "test-model", 10, request_pacing_seconds=0,
                              rate_limit_retries=0, retry_after_cap_seconds=0)
        class RateLimitError(Exception):
            status_code = 429
        class Completions:
            calls = 0
            def create(inner, **kwargs):
                inner.calls += 1
                raise RateLimitError()
        client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
        with patch("src.agents._pace_model_request"):
            with self.assertRaises(AgentRateLimitError):
                call_gemini(client, config=config, telemetry=RunTelemetryCounter())
        self.assertEqual(client.chat.completions.calls, 1)

    def test_interrupted_model_call_preserves_call_counts(self):
        class Completions:
            def create(self, **kwargs):
                raise KeyboardInterrupt()
        client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
        telemetry = RunTelemetryCounter(tool_calls=2, tool_names=["budget", "policy"])
        with patch("src.agents._pace_model_request"):
            with self.assertRaises(KeyboardInterrupt) as caught:
                call_gemini(client, config=self.config, telemetry=telemetry)
        self.assertEqual(caught.exception.agent_telemetry["llm_calls"], 1)
        self.assertEqual(caught.exception.agent_telemetry["tool_calls"], 2)


class EvaluationReliabilityTests(unittest.TestCase):
    def test_csv_checkpoint_accepts_success_and_error_rows_with_json_traces(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            root = Path(directory)
            (root / "evals").mkdir()
            common = {
                "timestamp_utc":"2026-10-11T00:00:00+00:00", "request_id":"REQ-X",
                "execution_mode":"real", "model":"test-model", "latency_ms":5.0,
                "llm_calls":1, "tool_calls":0, "model_tool_calls":0,
                "tool_call_names":[], "tool_names":[], "diagnostics":[],
                "outcome":"Hold for required human review", "expected_action":"hold_for_human_review",
                "correct":True, "evidence_references":[], "evidence_grounded":True,
                "deterministic_policy_safe":True, "human_escalation_correct":True,
                "approvals":[], "risks":[], "missing_information":[], "failures":[],
                "error":None, "status":"pass", "error_category":None,
                "termination_reason":None, "missing_evidence_tool_names":[],
            }
            success = {**common, "case_id":"EVAL-01", "architecture":"single"}
            failed = {**common, "case_id":"EVAL-02", "architecture":"single", "status":"error",
                      "correct":None, "error":"provider failed", "error_category":"provider_error",
                      "diagnostics":[{"event":"model_response", "tool_names":["budget_and_approvals"]}],
                      "tool_call_names":["budget_and_approvals"]}
            with patch.object(run_comparison, "ROOT", root):
                run_comparison.write_results("real", [success, failed], "test-model",
                                             selected_case_ids=["EVAL-01", "EVAL-02"])
            with (root / "evals" / "results_real.csv").open(encoding="utf-8", newline="") as stream:
                output = list(csv.DictReader(stream))
            self.assertEqual(len(output), 2)
            self.assertEqual(json.loads(output[1]["tool_call_names"]), ["budget_and_approvals"])
            self.assertEqual(json.loads(output[1]["diagnostics"])[0]["event"], "model_response")

    def test_per_case_error_is_recorded_and_other_cases_continue(self):
        from src.agents import AgentRateLimitError
        config = GeminiConfig("test-only", "test-model", 5, request_pacing_seconds=0)
        calls = 0

        def evaluate(case, architecture, mode):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise AgentRateLimitError("simulated quota exhausted")
            expected = {
                "request_clarification": "Request clarification before review",
                "hold_for_human_review": "Hold for required human review",
                "proceed_to_human_approval": "Proceed to human approval",
            }[case["expected_action"]]
            if case["case_id"] == "EVAL-02" and architecture == "staged":
                expected = "Hold for required human review"  # a scored assertion failure, not infra
            evidence = [SimpleNamespace(reference="synthetic.csv", source="fixture")]
            telemetry = SimpleNamespace(llm_calls=3, tool_calls=4, model="test-model",
                model_tool_calls=4, tool_call_names=[], tool_names=[], diagnostics=[])
            return SimpleNamespace(recommendation=expected, human_review_required=True,
                evidence=evidence, next_step="Route to human review.",
                required_approvals=case.get("required_approvals", []),
                risk_flags=case.get("required_risks", []), missing_information=[], telemetry=telemetry)

        with patch.object(run_comparison, "resolve_mode", return_value="real"), \
             patch.object(run_comparison, "get_gemini_config", return_value=config), \
             patch.object(run_comparison, "evaluate_case", side_effect=evaluate), \
             patch.object(run_comparison, "write_results"):
            result = run_comparison.run("real", limit=2)

        self.assertEqual(calls, 4)  # both architectures continue through both cases
        self.assertEqual([r["status"] for r in result["results"]], ["error", "pass", "pass", "fail"])
        self.assertEqual(result["results"][0]["error_category"], "rate_limited")
        self.assertEqual(result["architectures"]["single"]["error_count"], 1)
        self.assertEqual(result["architectures"]["single"]["model_quality_cases"], 1)
        self.assertEqual(result["architectures"]["staged"]["fail_count"], 1)
        self.assertEqual(result["architectures"]["staged"]["recommendation_accuracy"], 0.5)

    def test_interrupt_is_checkpointable_and_stops_after_current_case(self):
        config = GeminiConfig("test-only", "test-model", 5)
        with patch.object(run_comparison, "resolve_mode", return_value="real"), \
             patch.object(run_comparison, "get_gemini_config", return_value=config), \
             patch.object(run_comparison, "evaluate_case", side_effect=KeyboardInterrupt), \
             patch.object(run_comparison, "write_results"):
            result = run_comparison.run("real", limit=1)
        self.assertTrue(result["interrupted"])
        self.assertEqual(len(result["results"]), 1)
        self.assertEqual(result["results"][0]["error_category"], "interrupted")
        self.assertEqual(result["interrupted_case_id"], "EVAL-01")
        self.assertEqual(result["unprocessed_case_ids_by_architecture"]["staged"], ["EVAL-01"])

    def test_tool_loop_termination_reason_is_preserved(self):
        error = ToolBudgetExceeded("Loop stopped", reason="max_model_calls")
        category, reason = run_comparison.classify_error(error)
        self.assertEqual(category, "tool_loop_terminated")
        self.assertEqual(reason, "max_model_calls")

    def test_auth_and_malformed_responses_have_distinct_categories(self):
        from src.agents import AgentEvidenceIncompleteError
        self.assertEqual(run_comparison.classify_error(AgentAuthenticationError("auth"))[0], "authentication_error")
        self.assertEqual(run_comparison.classify_error(AgentOutputError("bad JSON"))[0], "malformed_response")
        self.assertEqual(run_comparison.classify_error(AgentEvidenceIncompleteError("tools missing", ["policy_lookup"]))[0], "incomplete_evidence")


if __name__ == "__main__":
    unittest.main()
