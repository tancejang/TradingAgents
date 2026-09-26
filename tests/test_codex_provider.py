"""Codex OAuth bridge tests: no credentials or model requests required."""

import queue
import sys
import time
from collections import deque
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel

from tradingagents.llm_clients import codex_client as codex
from tradingagents.llm_clients.factory import build_llm_kwargs, create_llm_client


class Report(BaseModel):
    summary: str


@pytest.fixture
def server(monkeypatch):
    class FakeServer:
        events = []
        instances = []
        auth_error = False

        def __init__(self, timeout):
            self.timeout = timeout
            self.calls = []
            self.pending = deque(self.events)
            self.directory = SimpleNamespace(name="/empty")
            self.closed = False
            self.instances.append(self)

        def initialize(self):
            if self.auth_error:
                raise RuntimeError("login required")

        def rpc(self, method, params):
            self.calls.append((method, params))
            return {"thread": {"id": "thread"}, "turn": {"id": "turn"}}

        def receive(self):
            raise TimeoutError("no events")

        def send(self, message):
            self.calls.append(("send", message))

        def close(self):
            self.closed = True

    monkeypatch.setattr(codex, "_AppServer", FakeServer)
    return FakeServer


def completed(text, status="completed"):
    return [
        {"method": "item/completed", "params": {
            "threadId": "thread", "turnId": "turn",
            "item": {"type": "agentMessage", "text": text, "phase": "final_answer"},
        }},
        {"method": "turn/completed", "params": {
            "threadId": "thread", "turn": {"id": "turn", "status": status,
                                           "error": {"message": "quota reached"}},
        }},
    ]


def tool_event(name="prices", arguments=None):
    return {"id": 100, "method": "item/tool/call", "params": {
        "threadId": "thread", "turnId": "turn", "callId": "call1",
        "tool": name, "arguments": {"ticker": "AAPL"} if arguments is None else arguments,
    }}


TOOL = {"name": "prices", "description": "Fetch prices", "parameters": {
    "type": "object", "properties": {"ticker": {"type": "string"}}, "required": ["ticker"],
}}


def test_factory_and_config():
    config = {"llm_provider": "codex", "openai_reasoning_effort": "low", "codex_timeout": "42"}
    llm = create_llm_client("CODEX", "account-model", **build_llm_kwargs(config)).get_llm()
    assert isinstance(llm, codex.CodexChatModel)
    assert llm.timeout == 42
    assert llm.reasoning_effort == "low"


def test_cli_registration():
    from cli.prompts import _llm_provider_table, ensure_api_key
    from tradingagents.llm_clients.model_catalog import get_model_options

    assert any(key == "codex" and url is None for _, key, url in _llm_provider_table())
    assert ensure_api_key("codex") is None
    assert get_model_options("codex", "quick") == [("Custom model ID", "custom")]


def test_text_and_transcript_replay(server):
    server.events = completed("analysis")
    messages = [SystemMessage("Analyze only the supplied evidence"), HumanMessage("AAPL"),
                AIMessage("", tool_calls=[{"id": "old", "name": "prices", "args": {"ticker": "AAPL"}}]),
                ToolMessage("123.45", tool_call_id="old")]
    result = codex.CodexChatModel(model="account-model", reasoning_effort="low").invoke(messages)
    assert result.content == "analysis"
    instance = server.instances[-1]
    assert instance.closed
    start, injected, turn = [params for _, params in instance.calls]
    assert start["ephemeral"] and start["environments"] == []
    assert start["modelProvider"] == "openai"
    assert start["approvalPolicy"] == "never"
    assert turn["effort"] == "low"
    transcript = injected["items"]
    assert transcript[0]["role"] == "developer"
    assert transcript[-1]["call_id"] == "old"
    assert transcript[-2]["call_id"] == "old"
    assert transcript[-1]["type"] == "function_call_output"
    assert transcript[-1]["output"] == "123.45"


def test_tool_handoff(server):
    server.events = [tool_event()]
    result = codex.CodexChatModel(model="account-model").bind_tools([TOOL]).invoke("Find prices")
    assert result.tool_calls == [{"name": "prices", "args": {"ticker": "AAPL"},
                                  "id": "call1", "type": "tool_call"}]
    instance = server.instances[-1]
    assert instance.closed
    assert instance.calls[0][1]["dynamicTools"][0]["inputSchema"] == TOOL["parameters"]


@pytest.mark.parametrize("event", [tool_event("unregistered"), tool_event(arguments="bad")])
def test_invalid_tool_requests_fail(server, event):
    server.events = [event]
    with pytest.raises(RuntimeError, match="tool"):
        codex.CodexChatModel(model="account-model").bind_tools([TOOL]).invoke("test")
    assert server.instances[-1].closed


def test_structured_output(server):
    server.events = completed('{"summary":"hold"}')
    llm = codex.CodexChatModel(model="account-model")
    assert llm.with_structured_output(Report).invoke("report") == Report(summary="hold")
    schema = server.instances[-1].calls[2][1]["outputSchema"]
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["summary"]
    raw = llm.with_structured_output(Report, include_raw=True).invoke("report")
    assert isinstance(raw["raw"], AIMessage)
    assert raw["parsed"] == Report(summary="hold") and raw["parsing_error"] is None


def test_structured_parse_failure(server):
    server.events = completed("not json")
    result = codex.CodexChatModel(model="m").with_structured_output(Report, include_raw=True).invoke("test")
    assert result["parsed"] is None and result["parsing_error"] is not None


@pytest.mark.parametrize("events,error", [(completed("", "failed"), "quota reached"),
                                         (completed(""), "without an assistant"), ([], "no events")])
def test_failure_and_timeout_cleanup(server, events, error):
    server.events = events
    with pytest.raises((RuntimeError, TimeoutError), match=error):
        codex.CodexChatModel(model="m").invoke("test")
    assert server.instances[-1].closed


def test_login_failure_cleanup(server):
    server.auth_error = True
    with pytest.raises(RuntimeError, match="login"):
        codex.CodexChatModel(model="m").invoke("test")
    assert server.instances[-1].closed


@pytest.mark.parametrize("account", [None, {"type": "apiKey"}, {"type": "amazonBedrock"}])
def test_requires_chatgpt_account(account):
    transport = object.__new__(codex._AppServer)
    transport.rpc = lambda *args: {"account": account}
    transport.send = lambda *args: None
    with pytest.raises(RuntimeError, match="ChatGPT login required"):
        transport.initialize()


def test_isolated_home_and_no_api_fallback(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADINGAGENTS_CODEX_HOME", str(tmp_path / "auth"))
    for key in ("OPENAI_API_KEY", "CODEX_API_KEY", "CODEX_ACCESS_TOKEN"):
        monkeypatch.setenv(key, "do-not-forward")
    env = codex._environment()
    assert env["CODEX_HOME"] == str((tmp_path / "auth").resolve())
    assert not {"OPENAI_API_KEY", "CODEX_API_KEY", "CODEX_ACCESS_TOKEN"} & env.keys()


def test_rpc_preserves_early_events():
    transport = object.__new__(codex._AppServer)
    transport.next_id = 0
    transport.pending = deque()
    transport.incoming = queue.Queue()
    transport.deadline = time.monotonic() + 5
    transport.send = lambda *args: None
    early = tool_event()
    transport.incoming.put(early)
    transport.incoming.put({"id": 1, "result": {"ok": True}})
    assert transport.rpc("turn/start", {}) == {"ok": True}
    assert list(transport.pending) == [early]


def test_transport_deadline_and_eof():
    transport = object.__new__(codex._AppServer)
    transport.incoming = queue.Queue()
    transport.deadline = time.monotonic() - 1
    with pytest.raises(TimeoutError):
        transport.receive()
    transport.deadline = time.monotonic() + 5
    transport.incoming.put(None)
    with pytest.raises(RuntimeError, match="exited unexpectedly"):
        transport.receive()


@pytest.mark.parametrize("action,extra,expected", [
    ("login", [], ["login"]), ("login", ["--device-auth"], ["login", "--device-auth"]),
    ("status", [], ["login", "status"]), ("logout", [], ["logout"]),
])
def test_login_commands(monkeypatch, tmp_path, action, extra, expected):
    monkeypatch.setenv("TRADINGAGENTS_CODEX_HOME", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["codex_client", action, *extra])
    monkeypatch.setattr(codex, "_command", lambda: "codex")
    calls = []
    monkeypatch.setattr(codex.subprocess, "call", lambda cmd, **kw: calls.append((cmd, kw)) or 0)
    with pytest.raises(SystemExit) as exc:
        codex.main()
    assert exc.value.code == 0
    assert calls[0][0][-len(expected):] == expected
    assert calls[0][1]["env"]["CODEX_HOME"] == str(tmp_path.resolve())


@pytest.mark.parametrize("kwargs", [{"base_url": "https://example.com"}, {"temperature": 0.2},
                                   {"max_tokens": 100}, {"max_retries": 2}])
def test_unsupported_settings_fail(kwargs):
    with pytest.raises(ValueError):
        create_llm_client("codex", "m", **kwargs).get_llm()


def test_missing_executable(monkeypatch):
    monkeypatch.setattr(codex.shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError, match="Install the Codex CLI"):
        codex._command()
