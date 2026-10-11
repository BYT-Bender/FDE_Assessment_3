from __future__ import annotations
import unittest
from unittest.mock import patch
from pydantic import ValidationError
from src.solution import analyze_request, handle_request
from src.tools import budget_and_approvals, search_existing_software, vendor_security_lookup

class WorkflowTests(unittest.TestCase):
    def test_budget_math_and_thresholds(self):
        low=budget_and_approvals("Finance",950)
        mid=budget_and_approvals("Finance",1000.01)
        high=budget_and_approvals("Finance",10000.01)
        top=budget_and_approvals("Finance",25000.01)
        self.assertEqual(low["available_usd"],29000)
        self.assertEqual(low["approvals"],["Manager"])
        self.assertEqual(mid["approvals"],["Department Head","Procurement"])
        self.assertIn("Finance",high["approvals"])
        self.assertIn("CFO",top["approvals"])
        self.assertEqual(budget_and_approvals("unknown",100)["status"],"missing")
        self.assertFalse(budget_and_approvals("Sales",22000)["sufficient"])

    def test_catalog_overlap(self):
        matches=search_existing_software({"product_name":"TaskFlow Pro","vendor_name":"TaskFlow","category":"Project Management","business_justification":"Marketing wants a task tracker for campaign launches.","department":"Marketing"})
        self.assertTrue(any(m["product_name"]=="TaskFlow" for m in matches["matches"]))

    def test_vendor_expired_and_conflicting(self):
        record={"vendor_name":"SignalWatch","risk_level":"medium","security_review_status":"expired","last_review_date":"2025-07-01","processes_personal_data":False,"stores_data_outside_region":False}
        with patch("src.tools.get_vendor_risk",return_value=record): result=vendor_security_lookup("SignalWatch")
        self.assertTrue(result["expired"])
        self.assertTrue(result["conflict"])

    def test_equivalent_vendor_status_labels_are_not_conflict(self):
        record={"vendor_name":"BrandBoard","risk_level":"medium","security_review_status":"not_completed","last_review_date":None,"processes_personal_data":True,"stores_data_outside_region":False}
        with patch("src.tools.get_vendor_risk",return_value=record): result=vendor_security_lookup("BrandBoard")
        self.assertNotIn("conflict",result)

    def test_vendor_outage_is_not_approved(self):
        result=vendor_security_lookup("NimbusAI")
        self.assertEqual(result["service_status"],"unavailable")
        self.assertTrue(result["expired"])

    def test_injection_and_missing_information(self):
        result=handle_request("REQ-1006", mode="demo")
        self.assertIn("prompt_injection_detected",result.risk_flags)
        self.assertTrue(result.missing_information)
        self.assertTrue(result.human_review_required)
        self.assertNotIn("CFO-approved",result.recommendation)

    def test_sensitive_data_escalates(self):
        result=handle_request("REQ-1004", mode="demo")
        self.assertIn("Security",result.required_approvals)
        self.assertIn("Privacy",result.required_approvals)

    def test_staged_review_adds_trace_and_keeps_tool_contract(self):
        single=handle_request("REQ-1001","single", mode="demo")
        staged=handle_request("REQ-1001","staged", mode="demo")
        self.assertEqual(single.telemetry.tool_calls,4)
        self.assertEqual(staged.telemetry.tool_calls,4)
        self.assertEqual(staged.telemetry.llm_calls,0)
        self.assertTrue(any(e.source.startswith("Policy and risk reviewer") for e in staged.evidence))

    def test_invalid_request_rejected(self):
        with self.assertRaises(ValidationError): analyze_request({"request_id":"x"}, mode="demo")

    def test_outage_downgrades_request_to_review(self):
        request={"request_id":"TEST","requester_id":"E004","product_name":"SafeTool","vendor_name":"SafeVendor","category":"x","annual_cost_usd":100,"user_count":1,"business_justification":"ordinary purpose","data_access_level":"none","requested_integrations":[]}
        with patch("src.tools.get_vendor_risk",side_effect=TimeoutError("offline")):
            result=analyze_request(request, mode="demo")
        self.assertIn("vendor_risk_unavailable",result.risk_flags)
        self.assertIn("Security",result.required_approvals)
        self.assertTrue(result.human_review_required)

if __name__=="__main__": unittest.main()
