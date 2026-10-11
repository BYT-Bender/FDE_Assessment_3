from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.data_access import load_employees, load_requests
from src.llm_config import get_gemini_config
from src.solution import analyze_request

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Procurement Copilot", page_icon="◈", layout="wide")
st.markdown("""<style>
.block-container{padding-top:2rem;max-width:1400px} .hero{padding:1.4rem 1.6rem;border-radius:14px;background:linear-gradient(120deg,#10263d,#1a4962);color:white;margin-bottom:1rem}
.hero p{color:#d3e4ed;margin-bottom:0}.tag{display:inline-block;padding:.25rem .65rem;border-radius:99px;background:#eaf1f7;color:#173d5a;font-size:.78rem;font-weight:650}
div[data-testid="stMetric"]{background:#f5f8fb;padding:12px 16px;border-radius:10px;border:1px solid #e3eaf0}
</style>""", unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>Procurement Request Copilot</h1><p>Evidence-led assessment for software requests · Recommendations support human decisions and never authorize a purchase.</p></div>', unsafe_allow_html=True)
st.caption("Synthetic assessment data · Policy snapshot 30 Sep 2026 · Human approval remains mandatory")
llm_config = get_gemini_config()

employees = load_employees().to_dict("records")
employee_map = {f"{e['name']} · {e['department']} ({e['employee_id']})": e for e in employees}
samples = load_requests()
sample_map = {f"{r['request_id']} · {r['product_name']}": r for r in samples}

with st.sidebar:
    st.subheader("Assessment setup")
    architecture_label = st.radio("Workflow", ["Single-agent baseline", "Staged two-agent review"], help="Both paths share the same tools and deterministic policy rules.")
    architecture = "single" if architecture_label.startswith("Single") else "staged"
    if llm_config.is_configured:
        mode_options = [f"Real LLM · Google Gemini · {llm_config.model}", "Deterministic demo · explicit opt-in"]
        mode_label = st.radio("Execution mode", mode_options, index=0)
        execution_mode = "real" if mode_label.startswith("Real LLM") else "demo"
        st.info(f"Gemini API key is present in local configuration. Google will validate it on request. Model: {llm_config.model}.")
    else:
        mode_options = ["Deterministic demo · GEMINI_API_KEY not configured", "Real LLM · requires GEMINI_API_KEY"]
        mode_label = st.radio("Execution mode", mode_options, index=0)
        execution_mode = "real" if mode_label.startswith("Real LLM") else "demo"
        st.warning("Real LLM mode is unavailable until you add GEMINI_API_KEY=your_key to the project-root .env file and restart. Current results are labelled deterministic demo.")
    st.markdown("**Workflow health**")
    st.success("Local data sources ready")
    st.info("Vendor-risk API is checked during analysis. Failures are shown as unverified evidence.")
    st.markdown("**Human authority**")
    st.caption("All results remain advisory. Required approvers must decide in the organization’s actual approval workflow.")

form_col, result_col = st.columns([1, 1.15], gap="large")
with form_col:
    st.subheader("Request details")
    st.caption("Fields marked * are needed for a complete assessment. You can submit a partial request; missing facts will be called out.")
    selected = st.selectbox("Start from a sample request", ["New request"] + list(sample_map), index=0)
    base = sample_map.get(selected, {})
    with st.form("request_form"):
        who_labels = list(employee_map)
        current_employee = next((k for k, v in employee_map.items() if v["employee_id"] == base.get("requester_id")), who_labels[0])
        requester_label = st.selectbox("Requester *", who_labels, index=who_labels.index(current_employee))
        product = st.text_input("Product or service *", value=base.get("product_name", ""))
        vendor = st.text_input("Vendor *", value=base.get("vendor_name", ""))
        category = st.text_input("Category", value=base.get("category", ""))
        left, right = st.columns(2)
        cost = left.number_input("Annual cost (USD) *", min_value=0.0, value=float(base.get("annual_cost_usd") or 0), step=100.0)
        users = right.number_input("Users/licenses *", min_value=0, value=int(base.get("user_count") or 0), step=1)
        purpose = st.text_area("Business purpose *", value=base.get("business_justification", ""), height=100)
        data_level_options = ["unknown", "none", "internal_documents", "internal_marketing", "source_code", "production_telemetry", "confidential_documents", "employee_pii", "customer_pii", "credentials"]
        raw_level = base.get("data_access_level", "unknown")
        data_level = st.selectbox("Data access level *", data_level_options, index=data_level_options.index(raw_level) if raw_level in data_level_options else 0)
        integration_text = st.text_input("Required integrations (comma separated)", value=", ".join(base.get("requested_integrations", [])))
        submitted = st.form_submit_button("Analyze request", type="primary", use_container_width=True)
    if submitted:
        request = {"request_id": base.get("request_id", "UI-" + str(__import__("time").time_ns())[-8:]),
                       "requester_id": employee_map[requester_label]["employee_id"], "product_name": product.strip(),
                       "vendor_name": vendor.strip(), "category": category.strip(), "annual_cost_usd": cost if cost > 0 else None,
                       "user_count": users if users > 0 else None, "business_justification": purpose.strip(), "data_access_level": data_level,
                       "requested_integrations": [x.strip() for x in integration_text.split(",") if x.strip()]}
        try:
            spinner = "Calling Gemini agents and gathering evidence…" if execution_mode == "real" else "Gathering deterministic demo evidence…"
            with st.spinner(spinner):
                st.session_state["decision"] = analyze_request(request, architecture, mode=execution_mode)
                st.session_state["request"] = request
        except Exception as exc:
            st.session_state.pop("decision", None)
            st.error(f"Analysis could not be completed: {type(exc).__name__}: {exc}")
            if execution_mode == "real":
                st.info("Real LLM execution failed. No deterministic demo fallback was used. Check .env, model access, network, and vendor service, then retry.")
            else:
                st.info("Demo analysis failed. Check the local data and vendor-risk service, then retry. A service failure never implies vendor approval.")

with result_col:
    st.subheader("Assessment result")
    decision = st.session_state.get("decision")
    if not decision:
        st.info("Submit a request to see its evidence and recommended next step. Missing required facts will be listed for clarification.")
        st.markdown("#### The assessment returns")
        st.markdown("Recommendation · Evidence and sources · Required approvals · Missing information · Risk flags · Next step")
    else:
        st.markdown(f"### {decision.recommendation}")
        st.warning("Human decision required · This is a recommendation, not an approval or purchase authorization.")
        st.markdown(f"**Next step:** {decision.next_step}")
        a, b, c = st.columns(3)
        a.metric("Evidence sources", len(decision.evidence))
        b.metric("Approvals", len(decision.required_approvals))
        c.metric("Risk flags", len(decision.risk_flags))
        with st.expander("Approvals required", expanded=True):
            if decision.required_approvals:
                for item in decision.required_approvals: st.markdown(f"- **{item}**")
            else: st.write("No additional reviewer identified by the snapshot rules; human approval still applies.")
        with st.expander("Missing information", expanded=True):
            if decision.missing_information:
                for item in decision.missing_information: st.markdown(f"- {item}")
            else: st.success("No required information is missing from the submitted request.")
        with st.expander("Risk flags", expanded=True):
            if decision.risk_flags:
                for item in decision.risk_flags: st.markdown(f"- `{item}`")
            else: st.success("No risk flags were identified in the available snapshot.")
        st.markdown("#### Evidence")
        for item in decision.evidence:
            with st.container(border=True):
                st.markdown(f"**{item.source}**")
                st.write(item.finding)
                if item.reference: st.caption(f"Source: {item.reference}")
        telemetry = decision.telemetry
        if telemetry:
            provider_label = f"Google Gemini · {telemetry.model}" if telemetry.execution_mode == "real" else "Deterministic demo (no LLM)"
            st.caption(f"Workflow: {architecture_label} · Mode: {provider_label} · Tool calls: {telemetry.tool_calls} · LLM calls: {telemetry.llm_calls}")
            if telemetry.tool_names:
                with st.expander("Tool activity"):
                    for name in telemetry.tool_names: st.markdown(f"- `{name}`")
            if telemetry.agents:
                st.caption("Agents: " + " → ".join(telemetry.agents))
        st.caption("No reviewer decision is recorded by this application. Use your authorized procurement workflow for approval and rationale.")
