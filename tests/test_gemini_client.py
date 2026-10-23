"""The Gemini client is tested against a local fake HTTP server that speaks the generateContent protocol."""
import dataclasses
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
import requests

from backend.app.core.errors import LLMCallError, LLMNotConfiguredError
from backend.app.services.llm.gemini import GeminiClient

SECRET = "SECRET-KEY-123"


class Server:
    def __init__(self, script):
        self.script, self.requests = list(script), []
        outer = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append({"path": self.path, "headers": dict(self.headers), "body": body})
                status, payload, hdrs = outer.script.pop(0) if outer.script else (500, {"error": {"message": "script exhausted"}}, {})
                data = json.dumps(payload).encode()
                self.send_response(status)
                for k, v in hdrs.items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a): pass

        self.httpd = HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.httpd.server_port}/v1beta"

    def close(self): self.httpd.shutdown()


OK = {"candidates": [{"content": {"parts": [{"thought": True, "text": "hidden"}, {"text": '{"a": 1}'}]}, "finishReason": "STOP"}],
      "usageMetadata": {"promptTokenCount": 11, "candidatesTokenCount": 5, "thoughtsTokenCount": 3, "totalTokenCount": 19},
      "modelVersion": "gemini-test-001"}


def client(settings, server, **kw):
    s = dataclasses.replace(settings, llm_provider="gemini", gemini_api_key=SECRET, gemini_model="gemini-test", gemini_api_base=server.base,
                            llm_max_retries=kw.pop("retries", 3))
    sess = requests.Session()
    sess.trust_env = False   # never route the loopback test server through a proxy
    sleeps = []
    return GeminiClient(s, session=sess, sleep=sleeps.append), sleeps


def test_success_extracts_text_usage_and_version(settings):
    srv = Server([(200, OK, {})])
    c, _ = client(settings, srv)
    r = c.generate("hello", temperature=0.2, max_output_tokens=100)
    srv.close()
    assert r.text == '{"a": 1}' and "hidden" not in r.text          # reasoning ("thought") parts are skipped
    assert (r.input_tokens, r.output_tokens, r.thoughts_tokens, r.total_tokens) == (11, 5, 3, 19)
    assert r.model_version == "gemini-test-001" and r.finish_reason == "STOP" and r.retry_count == 0
    req = srv.requests[0]
    assert req["path"].endswith("/models/gemini-test:generateContent")
    assert req["headers"].get("x-goog-api-key") == SECRET and SECRET not in req["path"]   # key in header, never in URL
    assert req["body"]["generationConfig"]["temperature"] == 0.2


def test_retries_on_503_and_honours_retry_after(settings):
    srv = Server([(503, {"error": {"status": "UNAVAILABLE", "message": "busy"}}, {"Retry-After": "2"}), (429, {"error": {"message": "slow"}}, {}), (200, OK, {})])
    c, sleeps = client(settings, srv)
    r = c.generate("x")
    srv.close()
    assert r.retry_count == 2 and len(r.errors) == 2 and sleeps[0] == 2.0 and len(sleeps) == 2


def test_gives_up_after_max_retries(settings):
    srv = Server([(500, {"error": {"message": "boom"}}, {})] * 5)
    c, _ = client(settings, srv, retries=2)
    with pytest.raises(LLMCallError) as ei:
        c.generate("x")
    srv.close()
    assert len(srv.requests) == 3 and ei.value.retry_count == 2


def test_client_error_is_not_retried_and_key_is_not_leaked(settings):
    srv = Server([(400, {"error": {"status": "INVALID_ARGUMENT", "message": f"bad key {SECRET}"}}, {})])
    c, _ = client(settings, srv)
    with pytest.raises(LLMCallError) as ei:
        c.generate("x")
    srv.close()
    assert len(srv.requests) == 1 and SECRET not in str(ei.value) and SECRET not in " ".join(ei.value.errors)


def test_blocked_response_raises_without_retry(settings):
    srv = Server([(200, {"candidates": [], "promptFeedback": {"blockReason": "SAFETY"}}, {})])
    c, _ = client(settings, srv)
    with pytest.raises(LLMCallError):
        c.generate("x")
    srv.close()
    assert len(srv.requests) == 1


def test_missing_key_or_model_is_a_configuration_error(settings):
    with pytest.raises(LLMNotConfiguredError):
        GeminiClient(dataclasses.replace(settings, gemini_api_key=None, gemini_model="m"))
    with pytest.raises(LLMNotConfiguredError):
        GeminiClient(dataclasses.replace(settings, gemini_api_key="k", gemini_model=None))
