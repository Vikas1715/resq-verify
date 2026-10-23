"""Shared UI helpers: CSS, banners, KPI cards, error display."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

import streamlit as st

from frontend.api_client import ApiError

NO_CONCLUSION = "No experimental conclusion available yet."
SIM_TEXT = "SIMULATED LLM (mock) — results are NOT experimental results. Set LLM_PROVIDER=gemini with your key for real runs."
DEMO_TEXT = "DEMO DATA — NOT EXPERIMENTAL RESULT"
METHOD_LABELS = {"llm_only": "LLM-only", "self_critique": "Self-critique", "resq_verify": "ResQ-Verify", "resample_verify": "Resample + Verify",
                 "ortools": "OR-Tools CP-SAT"}
METHOD_COLORS = {"llm_only": "#9aa5b1", "self_critique": "#e0a030", "resq_verify": "#1f6feb", "resample_verify": "#a371f7", "ortools": "#2da44e"}


def load_css() -> None:
    css = (Path(__file__).parent / "style.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def banner(text: str, kind: str = "info") -> None:
    st.markdown(f'<div class="rv-banner {kind}">{text}</div>', unsafe_allow_html=True)


def sim_banner() -> None:
    banner(f"<b>{SIM_TEXT}</b>", "sim")


def no_results() -> None:
    banner(f"<b>{NO_CONCLUSION}</b> Run an experiment in the Experiment Lab first.", "none")


def kpi(label: str, value: Any) -> None:
    st.markdown(f'<div class="rv-card"><div class="rv-kpi-label">{label}</div><div class="rv-kpi">{value}</div></div>', unsafe_allow_html=True)


def pill(text: str, kind: str) -> str:
    return f'<span class="rv-pill {kind}">{text}</span>'


def fmt_pct(x: Optional[float], d: int = 1) -> str:
    return "—" if x is None else f"{x * 100:.{d}f}%"


def fmt_num(x: Optional[float], d: int = 3) -> str:
    return "—" if x is None else f"{x:.{d}f}"


def safe(fn: Callable[[], Any], default: Any = None) -> Any:
    """Run an API call; show a readable error instead of a traceback."""
    try:
        return fn()
    except ApiError as e:
        st.error(e.message)
    except Exception as e:  # noqa: BLE001 - never show a raw traceback to the user
        st.error(f"Unexpected error: {type(e).__name__}: {e}")
    return default


def violations_html(report: dict) -> str:
    if not report:
        return ""
    if report.get("valid"):
        return pill("VALID — no hard-constraint violations", "ok")
    out = [pill(f"{report['n_violations']} violation(s)", "bad")]
    for v in report["violations"]:
        out.append(f'<div class="rv-viol"><b>{v["rule"]} · {v["type"]}</b> · {v["entity"]}<br>{v["message"]}</div>')
    return "".join(out)
