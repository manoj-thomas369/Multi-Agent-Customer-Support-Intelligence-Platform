import os

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL_DEFAULT = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="Support Intelligence Platform", page_icon="🤖", layout="wide")

if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.title("🤖 Multi-Agent Support")
    st.caption("E-commerce customer support intelligence platform")
    api_base_url = st.text_input("API base URL", value=API_BASE_URL_DEFAULT)
    if st.button("Reset conversation", use_container_width=True):
        st.session_state.history = []
        st.rerun()

    try:
        health = requests.get(f"{api_base_url}/health", timeout=3)
        if health.ok:
            st.success("Backend connected")
        else:
            st.error("Backend reachable but unhealthy")
    except requests.RequestException:
        st.error("Backend not reachable. Start it with:\n\nuvicorn src.api.main:app --reload")

st.title("Customer Support Ticket Assistant")
st.caption("Submit a ticket the way a customer would, and watch each agent's decision unfold.")

_PRIORITY_COLOR = {"High": "🔴", "Medium": "🟠", "Low": "🟢"}
_SENTIMENT_EMOJI = {"negative": "😠", "neutral": "😐", "positive": "🙂"}


def _submit_ticket(text: str, api_base_url: str) -> dict:
    create_resp = requests.post(f"{api_base_url}/ticket", json={"text": text}, timeout=10)
    create_resp.raise_for_status()
    ticket_id = create_resp.json()["ticket_id"]

    process_resp = requests.post(f"{api_base_url}/process", json={"ticket_id": ticket_id}, timeout=30)
    process_resp.raise_for_status()
    return process_resp.json()


def _render_assistant_turn(result: dict) -> None:
    st.markdown(result["response"])

    cols = st.columns(4)
    cols[0].metric("Category", result["category"] or "-")
    cols[1].metric("Priority", f"{_PRIORITY_COLOR.get(result['priority'], '')} {result['priority']}")
    cols[2].metric("Sentiment", f"{_SENTIMENT_EMOJI.get(result['sentiment'], '')} {result['sentiment']}")
    cols[3].metric("Confidence", f"{result['category_confidence']:.0%}")

    if result["escalate"]:
        st.warning(f"🚨 Escalated to a human specialist — reason: {result['escalation_reason']}")

    with st.expander("🔍 Agent decision log"):
        for entry in result["logs"]:
            st.markdown(f"**{entry['agent']}** → `{entry['decision']}`")
            if entry["detail"]:
                st.json(entry["detail"], expanded=False)


for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        if turn["role"] == "assistant":
            _render_assistant_turn(turn["result"])
        else:
            st.markdown(turn["content"])

ticket_text = st.chat_input("Describe your issue (e.g. order number, product, what went wrong)...")

if ticket_text:
    st.session_state.history.append({"role": "user", "content": ticket_text})
    with st.chat_message("user"):
        st.markdown(ticket_text)

    with st.chat_message("assistant"):
        with st.spinner("Agents are working on your ticket..."):
            try:
                result = _submit_ticket(ticket_text, api_base_url)
            except requests.RequestException as exc:
                st.error(f"Couldn't reach the backend: {exc}")
                result = None

        if result is not None:
            _render_assistant_turn(result)
            st.session_state.history.append({"role": "assistant", "result": result})
