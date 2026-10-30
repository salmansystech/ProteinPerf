import json
import os

import pytest

from proteinperf.assistant import AssistantError, ResultsTools, ask, load_env

@pytest.fixture
def results(tmp_path):
    rows = [{"model": "150M", "variant": "compile", "speedup_vs_eager": 1.23456789},
            {"model": "650M", "variant": "compile", "speedup_vs_eager": 1.1}]
    (tmp_path / "optimizations.json").write_text(json.dumps({"metadata": {"settings": {"dtype": "fp16"}}, "results": rows}))
    (tmp_path / "profiles").mkdir()
    (tmp_path / "profiles" / "esm2.txt").write_text("ESM-2 150M (fp16)\nGPU time per forward pass: 1.00 ms\n")
    return tmp_path

def _call(tool, **arguments):
    return {"function": {"name": tool, "arguments": arguments}}

class FakeModel:
    """Replays scripted assistant messages and records what it was sent"""
    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests = []

    def __call__(self, messages, tools):
        self.requests.append([dict(m) for m in messages])
        return self.replies.pop(0)

def test_list_results(results):
    listing = json.loads(ResultsTools(str(results)).list_results())
    assert list(listing["results"]) == ["optimizations.json"]
    assert listing["profiles"] == ["esm2.txt"]

def test_read_results_filters_and_rounds(results):
    data = json.loads(ResultsTools(str(results)).read_results("optimizations.json", model="150M"))
    assert data["settings"] == {"dtype": "fp16"}
    assert data["rows"] == [{"model": "150M", "variant": "compile", "speedup_vs_eager": 1.2346}]

def test_read_results_rejects_unknown_and_missing_files(results):
    tools = ResultsTools(str(results))
    assert tools.read_results("../secrets.json").startswith("Unknown results file")
    assert tools.read_results("batching.json") == "batching.json has not been generated yet."

def test_read_profile_stays_in_profiles_folder(results):
    tools = ResultsTools(str(results))
    assert tools.read_profile("esm2.txt").startswith("ESM-2 150M")
    assert tools.read_profile("../optimizations.json") == "No profile named '../optimizations.json'."

def test_bad_tool_calls_return_errors_to_the_model(results):
    tools = ResultsTools(str(results))
    assert tools.call("delete_everything", {}) == "Unknown tool 'delete_everything'"
    assert tools.call("read_results", {"wrong": 1}).startswith("Invalid arguments for read_results")

def test_ask_runs_tools_then_answers(results):
    model = FakeModel({"content": "", "tool_calls": [_call("read_results", name="optimizations.json", model="650M")]},
                      {"content": "compile is 1.1x faster for 650M."})
    seen = []
    answer = ask("How fast is compile?", str(results), chat=model, on_tool_call=lambda n, a: seen.append(n))

    assert answer == "compile is 1.1x faster for 650M."
    assert seen == ["read_results"]
    first, second = model.requests
    assert first[0]["role"] == "system" and first[1] == {"role": "user", "content": "How fast is compile?"}
    tool_message = second[-1]
    assert tool_message["role"] == "tool" and tool_message["tool_name"] == "read_results"
    assert '"650M"' in tool_message["content"] and '"150M"' not in tool_message["content"]

def test_ask_accepts_arguments_as_json_string(results):
    model = FakeModel({"tool_calls": [{"function": {"name": "read_profile", "arguments": '{"name": "esm2.txt"}'}}]},
                      {"content": "done"})
    assert ask("q", str(results), chat=model) == "done"
    assert model.requests[1][-1]["content"].startswith("ESM-2 150M")

def test_ask_stops_after_too_many_tool_rounds(results):
    model = FakeModel(*[{"tool_calls": [_call("list_results")]}] * 20)
    with pytest.raises(AssistantError, match="No answer"):
        ask("q", str(results), chat=model)

def test_load_env_does_not_override(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# comment\nPP_TEST_A='from file'\nPP_TEST_B=from file\n")
    monkeypatch.setenv("PP_TEST_B", "already set")
    monkeypatch.delenv("PP_TEST_A", raising=False)
    load_env(str(env))
    assert os.environ["PP_TEST_A"] == "from file"
    assert os.environ["PP_TEST_B"] == "already set"
    monkeypatch.delenv("PP_TEST_A")
