"""Streamlit and Plotly cannot be assumed in the test environment's UI sense, so this runs every page's render()
against a recording fake `st` and the real in-process API. It catches typos, bad keys and crashes, not layout."""
import sys
import types
from unittest.mock import MagicMock

import pytest

pytest.importorskip("pandas")
pytest.importorskip("fastapi")


class FakeSt(types.ModuleType):
    def __init__(self):
        super().__init__("streamlit")
        self.session_state = {}
        self.calls = []

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)

        def f(*a, **k):
            self.calls.append(name)
            if name == "columns":
                n = a[0] if isinstance(a[0], int) else len(a[0])
                return [MagicMock() for _ in range(n)]
            if name == "tabs":
                return [MagicMock() for _ in a[0]]
            if name in ("selectbox", "radio"):
                opts = a[1] if len(a) > 1 else k.get("options")
                return opts[k.get("index", 0)] if opts else None
            if name == "multiselect":
                return a[2] if len(a) > 2 else k.get("default", [])
            if name in ("button", "form_submit_button", "checkbox", "download_button"):
                return bool(k.get("value", a[1] if name == "checkbox" and len(a) > 1 else False))
            if name == "number_input":
                return a[2] if len(a) > 2 else k.get("value", 0)
            if name in ("text_input",):
                return a[1] if len(a) > 1 else ""
            if name == "text_area":
                return ""
            return MagicMock()
        return f


@pytest.fixture()
def fake_streamlit(monkeypatch):
    fake = FakeSt()
    monkeypatch.setitem(sys.modules, "streamlit", fake)
    for m in [m for m in sys.modules if m.startswith("frontend")]:
        monkeypatch.delitem(sys.modules, m)
    return fake


def monkeypatch_plotly(_):
    m = MagicMock()
    sys.modules["plotly"] = m
    sys.modules["plotly.graph_objects"] = m.graph_objects


def test_every_page_renders_without_crashing(fake_streamlit, settings):
    try:
        import plotly  # noqa: F401
    except ImportError:   # chart code is still executed, against a recording stand-in
        monkeypatch_plotly(fake_streamlit)
    from fastapi.testclient import TestClient
    from backend.app.api.app import create_app
    from frontend.api_client import Api
    from frontend.views import (experiment_details, experiment_lab, export_page, overview, planning_lab, results, scenario_lab, verification_lab)

    app = create_app(settings)
    tc = TestClient(app)

    class InProc(Api):
        def __init__(self): super().__init__("http://test")
        def _req(self, method, path, *, raw=False, timeout=None, **kw):
            r = tc.request(method, path, **{k: v for k, v in kw.items() if k in ("json", "params")})
            if r.status_code >= 400:
                from frontend.api_client import ApiError
                raise ApiError(r.json()["error"]["message"], status=r.status_code)
            return r.content if raw else r.json()

    api = InProc()
    pages = (overview, scenario_lab, planning_lab, verification_lab, experiment_lab, results, experiment_details, export_page)
    for p in pages:                      # empty state first: nothing generated, nothing run
        p.render(api)
    api.generate_scenario("small", "high", 1)
    e = api.run_experiment({"sizes": ["small"], "scarcity": ["high"], "num_scenarios": 2})
    app.state.ctx.manager.join(e["experiment_id"], 60)
    api.plan("verify", "small-high-s1")
    for p in pages:                      # then with data
        p.render(api)
    assert "plotly_chart" in fake_streamlit.calls
    sys.modules.pop("plotly", None) if isinstance(sys.modules.get("plotly"), MagicMock) else None
