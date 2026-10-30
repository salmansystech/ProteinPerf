"""Profiling assistant: an LLM that answers questions about the results by reading them through tools.

Uses Ollama's chat API, so it runs with a free local model or with Ollama Cloud. The model
never sees results it hasn't asked for, and it is told to answer only from what the tools return.
"""
from pathlib import Path
from typing import Callable, Dict, List, Optional
import json
import os
import urllib.error
import urllib.request

from proteinperf.findings import RULES, collect_findings

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "gemma4:31b"
MAX_TOOL_ROUNDS = 8
MAX_ROWS = 60

SYSTEM_PROMPT = """You are a GPU performance engineer reviewing ProteinPerf results: inference benchmarks,
profiles, roofline analysis and optimizations of ESM-2 protein language models.

Use the tools to read the results before answering. Start with get_findings, then read the raw results
you need. Answer only from what the tools return: quote the measured numbers, name the model, precision,
batch size and length they apply to, and say when results are flagged as noisy or missing. Never invent
measurements. When you suggest an optimization, explain which measurement shows it would help.
Keep the answer short and in plain language."""

DESCRIPTIONS = {
    "baseline.json": "Latency, throughput and memory for each model, precision, batch size and length (phase 1)",
    "roofline.json": "FLOPs, bytes, arithmetic intensity, measured time and roofline efficiency of each layer operation (phase 2)",
    "optimizations.json": "Speed-up of torch.compile, CUDA graphs and attention backends over eager mode (phase 3)",
    "batching.json": "Time, padding efficiency and memory of each batching strategy on real proteins (phase 3)",
    "kernels.json": "Custom Triton kernels vs PyTorch eager and torch.compile, in isolation (phase 4)",
    "kernel_variants.json": "Whole-model speed-up with the Triton kernels swapped in (phase 4)",
}

TOOLS = [
    {"type": "function", "function": {
        "name": "list_results",
        "description": "List the results files and profile reports that exist, with what each one contains.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "get_findings",
        "description": "Main conclusions computed from all results by fixed rules: speed-ups, bottlenecks, noise warnings.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "read_results",
        "description": "Rows of one results file, optionally only for one model.",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string", "enum": list(DESCRIPTIONS), "description": "Results file name"},
            "model": {"type": "string", "description": "Optional model size such as 150M or 650M"},
        }, "required": ["name"]},
    }},
    {"type": "function", "function": {
        "name": "read_profile",
        "description": "A profiler report: GPU time by operator category and the slowest operators.",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string", "description": "Profile file name from list_results"},
        }, "required": ["name"]},
    }},
]

class AssistantError(Exception):
    pass

def load_env(path: str = ".env"):
    """Read KEY=value lines into the environment, without overriding variables already set"""
    env_file = Path(path)
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

class ResultsTools:
    """The tools the model can call, reading from one results directory"""

    def __init__(self, results_dir: str = "results"):
        self.directory = Path(results_dir)

    def list_results(self) -> str:
        files = {name: description for name, description in DESCRIPTIONS.items() if (self.directory / name).exists()}
        profiles = [p.name for p in sorted((self.directory / "profiles").glob("*.txt"))]
        return json.dumps({"results": files, "profiles": profiles}, indent=1)

    def get_findings(self) -> str:
        findings = collect_findings(str(self.directory))
        return "\n".join(f"- {f}" for f in findings) if findings else "No results found."

    def read_results(self, name: str, model: Optional[str] = None) -> str:
        if name not in DESCRIPTIONS:
            return f"Unknown results file '{name}'. Choose from: {', '.join(DESCRIPTIONS)}"
        path = self.directory / name
        if not path.exists():
            return f"{name} has not been generated yet."
        data = json.loads(path.read_text())
        rows = [r for r in data["results"] if model is None or r.get("model") == model]
        note = f" (showing {MAX_ROWS} of {len(rows)} rows; filter by model for the rest)" if len(rows) > MAX_ROWS else ""
        settings = data.get("metadata", {}).get("settings", {})
        rounded = [{k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()} for r in rows[:MAX_ROWS]]
        return json.dumps({"settings": settings, "rows": rounded}) + note

    def read_profile(self, name: str) -> str:
        path = self.directory / "profiles" / Path(name).name    # never read outside the profiles folder
        return path.read_text(encoding="utf-8") if path.exists() else f"No profile named '{name}'."

    def call(self, name: str, arguments: Dict) -> str:
        functions = {"list_results": self.list_results, "get_findings": self.get_findings,
                     "read_results": self.read_results, "read_profile": self.read_profile}
        if name not in functions:
            return f"Unknown tool '{name}'"
        try:
            return functions[name](**(arguments or {}))
        except TypeError as e:
            return f"Invalid arguments for {name}: {e}"

def ollama_chat(messages: List[Dict], tools: List[Dict]) -> Dict:
    """One request to Ollama's chat API; returns the assistant message"""
    base_url = os.getenv("OLLAMA_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    body = {"model": os.getenv("OLLAMA_MODEL", DEFAULT_MODEL), "messages": messages, "tools": tools, "stream": False}
    headers = {"Content-Type": "application/json"}
    if os.getenv("OLLAMA_API_KEY"):
        headers["Authorization"] = f"Bearer {os.environ['OLLAMA_API_KEY']}"
    request = urllib.request.Request(f"{base_url}/api/chat", data=json.dumps(body).encode(), headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return json.loads(response.read())["message"]
    except urllib.error.HTTPError as e:
        if e.code == 401:
            raise AssistantError("Ollama rejected the API key: check OLLAMA_API_KEY") from e
        raise AssistantError(f"Ollama API error ({e.code}): {e.read().decode(errors='replace')[:200]}") from e
    except urllib.error.URLError as e:
        raise AssistantError(f"Could not reach Ollama at {base_url}: {e.reason}") from e

def ask(question: str, results_dir: str = "results", chat: Callable[[List[Dict], List[Dict]], Dict] = ollama_chat,
        on_tool_call: Optional[Callable[[str, Dict], None]] = None) -> str:
    """Answer a question about the results, letting the model call tools until it has what it needs"""
    tools = ResultsTools(results_dir)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}]
    for _ in range(MAX_TOOL_ROUNDS):
        message = chat(messages, TOOLS)
        calls = message.get("tool_calls") or []
        messages.append({"role": "assistant", "content": message.get("content", ""), "tool_calls": calls})
        if not calls:
            return message.get("content", "").strip()
        for call in calls:
            name = call["function"]["name"]
            arguments = call["function"].get("arguments") or {}
            if isinstance(arguments, str):           # some models send arguments as a JSON string
                arguments = json.loads(arguments or "{}")
            if on_tool_call:
                on_tool_call(name, arguments)
            messages.append({"role": "tool", "tool_name": name, "content": tools.call(name, arguments)})
    raise AssistantError(f"No answer after {MAX_TOOL_ROUNDS} rounds of tool calls")
