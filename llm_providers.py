"""
LLM Provider abstraction layer.
Supports: Ollama (local), OpenAI (GPT), Anthropic (Claude).

The goal: the engine.py doesn't care which provider is used.
All providers return the same message format:
    {"role": "assistant", "content": "...", "tool_calls": [...]}

Usage:
    from llm_providers import get_provider
    llm = get_provider("ollama")           # or "openai", "claude"
    msg = llm.chat(messages, tool_specs)
"""
import json
import os
import urllib.error
import urllib.request


# ============================================================
# Base class
# ============================================================
class LLMProvider:
    """All providers must implement .chat(messages, tools) -> dict."""

    def chat(self, messages: list, tools: list) -> dict:
        raise NotImplementedError

    @staticmethod
    def _normalize_tool_calls(calls: list) -> list:
        """Convert any provider's tool_calls to a unified format:
        [{"function": {"name": str, "arguments": dict}}, ...]"""
        normalized = []
        for c in calls or []:
            # Some providers give objects, some give dicts
            if hasattr(c, "function"):
                fn = c.function
                name = fn.name
                args = fn.arguments
            else:
                fn = c.get("function", {})
                name = fn.get("name")
                args = fn.get("arguments", {})
            # Arguments might be a JSON string; parse it
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            normalized.append({"function": {"name": name, "arguments": args}})
        return normalized


# ============================================================
# Ollama (local, free, private)
# ============================================================
class OllamaProvider(LLMProvider):
    def __init__(self, model: str = "qwen3.5:4b",
                 url: str = "http://localhost:11434/api/chat",
                 num_ctx: int = 4096, temperature: float = 0.2):
        self.model = model
        self.url = url
        self.num_ctx = num_ctx
        self.temperature = temperature
        # Ignore system proxy for localhost
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def chat(self, messages: list, tools: list) -> dict:
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "stream": False,
            "think": False,
            "options": {"num_ctx": self.num_ctx, "temperature": self.temperature},
        }
        req = urllib.request.Request(
            self.url, data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with self._opener.open(req, timeout=600) as resp:
                msg = json.loads(resp.read().decode("utf-8"))["message"]
        except urllib.error.HTTPError as e:
            raise SystemExit(f"Ollama error {e.code}: {e.read().decode('utf-8', 'replace')}")
        except urllib.error.URLError as e:
            raise SystemExit(f"Cannot reach Ollama at {self.url}: {e}\n"
                             "Make sure the Ollama app is running.")
        # Normalize tool_calls
        msg["tool_calls"] = self._normalize_tool_calls(msg.get("tool_calls"))
        return msg


# ============================================================
# OpenAI (GPT)
# ============================================================
class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str = None, model: str = "gpt-4o-mini",
                 temperature: float = 0.2):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise SystemExit("OPENAI_API_KEY is not set.")
        self.model = model
        self.temperature = temperature

    def chat(self, messages: list, tools: list) -> dict:
        from openai import OpenAI  # imported lazily so stdlib users don't need it
        client = OpenAI(api_key=self.api_key)

        # OpenAI wants tools wrapped differently
        openai_tools = [{"type": "function", "function": t["function"]} for t in tools]

        resp = client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=openai_tools,
            temperature=self.temperature,
        )
        msg = resp.choices[0].message
        return {
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": self._normalize_tool_calls(msg.tool_calls),
        }


# ============================================================
# Anthropic (Claude)
# ============================================================
class ClaudeProvider(LLMProvider):
    def __init__(self, api_key: str = None,
                 model: str = "claude-3-5-sonnet-20241022",
                 temperature: float = 0.2):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise SystemExit("ANTHROPIC_API_KEY is not set.")
        self.model = model
        self.temperature = temperature

    def chat(self, messages: list, tools: list) -> dict:
        import anthropic
        client = anthropic.Anthropic(api_key=self.api_key)

        # Anthropic separates system from messages
        system_msg = ""
        chat_messages = []
        for m in messages:
            if m["role"] == "system":
                system_msg += m["content"] + "\n"
            elif m["role"] == "tool":
                # Anthropic expects tool results as user messages with tool_result blocks
                chat_messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": m.get("tool_call_id", "unknown"),
                        "content": m["content"],
                    }],
                })
            else:
                chat_messages.append({"role": m["role"], "content": m["content"]})

        # Anthropic tool format
        anthropic_tools = [{
            "name": t["function"]["name"],
            "description": t["function"]["description"],
            "input_schema": t["function"]["parameters"],
        } for t in tools]

        resp = client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system_msg.strip(),
            messages=chat_messages,
            tools=anthropic_tools,
            temperature=self.temperature,
        )

        # Convert Anthropic response to our unified format
        content = ""
        tool_calls = []
        for block in resp.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                tool_calls.append({"function": {"name": block.name, "arguments": block.input}})
        return {"role": "assistant", "content": content, "tool_calls": tool_calls}


# ============================================================
# Factory
# ============================================================
def get_provider(name: str = None, **kwargs) -> LLMProvider:
    """
    Get an LLM provider by name.
    If name is None, reads from env var LLM_PROVIDER (default: ollama).

    Examples:
        get_provider("ollama", model="qwen3.5:4b")
        get_provider("openai", model="gpt-4o-mini")
        get_provider("claude", model="claude-3-5-sonnet-20241022")
    """
    name = (name or os.getenv("LLM_PROVIDER", "ollama")).lower()
    if name == "ollama":
        return OllamaProvider(**kwargs)
    if name == "openai":
        return OpenAIProvider(**kwargs)
    if name in ("claude", "anthropic"):
        return ClaudeProvider(**kwargs)
    raise ValueError(f"Unknown provider: {name}. Use ollama, openai, or claude.")