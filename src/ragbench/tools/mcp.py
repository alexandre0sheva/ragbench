"""MCP tools: the tools of a Model Context Protocol server, offered to an agent like any other tool.

    tools:
      - {type: mcp, server: {command: [python, price_server.py]}, allow: [lookup_price]}

The server is started (or reached) once per run, when the first system that lists it is built, and shared by every question; its tool list
is read then, so the agent is told the server's own descriptions and argument schemas. Only the tools under `allow` are offered. Under
`--mock` no server is started: each allowed tool answers with a fixed placeholder, so a config with MCP tools runs offline.

The SDK (extra `mcp`) is imported only when a server is actually started. Its calls are asynchronous; each server gets one event loop in a
daemon thread, and a tool call from any question thread is a request on that loop.
"""

from __future__ import annotations

import asyncio
import atexit
import concurrent.futures
import difflib
import json
import shlex
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Literal, Protocol
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from ragbench.tools.base import SideEffects, ToolContext, ToolResult, ToolSpec
from ragbench.tools.custom import NAME_PATTERN

INSTALL_HINT = "MCP tools need the MCP SDK: pip install 'ragbench[mcp]'"
STARTUP_TIMEOUT_S = 30.0
MAX_PAGES = 100


def is_mcp_ref(ref: Any) -> bool:
    return isinstance(ref, dict) and ref.get("type") == "mcp"


class McpServerConfig(BaseModel):
    """Where the server is: a `command` to start (stdio) or the `url` of a Streamable HTTP endpoint. `${VAR}` in `env` and `headers`
    is filled from the environment when the server is started, so secrets are never written to a config or a run directory."""

    model_config = ConfigDict(extra="forbid")

    command: str | list[str] | None = Field(default=None, description="Command that starts the server on stdio; a string is split like a shell would.")
    env: dict[str, str] = Field(default_factory=dict, description="Extra environment for the command; `${VAR}` is filled from the environment.")
    cwd: str | None = Field(default=None, description="Working directory of the command.")
    url: str | None = Field(default=None, description="Streamable HTTP endpoint of a running server.")
    headers: dict[str, str] = Field(default_factory=dict, description="Extra HTTP headers (e.g. `Authorization: Bearer ${TOKEN}`).")

    @model_validator(mode="after")
    def _one_transport(self) -> McpServerConfig:
        if (self.command is None) == (self.url is None):
            raise ValueError("server needs exactly one of `command` or `url`")
        if self.url is None and self.headers:
            raise ValueError("`headers` only apply to a `url` server")
        if self.command is None and (self.env or self.cwd):
            raise ValueError("`env` and `cwd` only apply to a `command` server")
        if self.command is not None and not self.argv():
            raise ValueError("server `command` is empty")
        return self

    def argv(self) -> list[str]:
        if self.command is None:
            return []
        return shlex.split(self.command) if isinstance(self.command, str) else [str(part) for part in self.command]

    @property
    def side_effects(self) -> SideEffects:
        return "network" if self.url is not None else "filesystem"  # a started process can do anything on this machine

    @property
    def label(self) -> str:
        """How errors name the server: never a secret (no environment, headers, query or userinfo)."""
        if self.url is not None:
            parts = urlsplit(self.url)
            return f"{parts.scheme}://{parts.hostname or ''}{':' + str(parts.port) if parts.port else ''}{parts.path}"
        return " ".join(self.argv())


class McpRef(BaseModel):
    """One `{type: mcp, ...}` entry of a system's `tools:` list."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["mcp"]
    server: McpServerConfig
    allow: list[str] = Field(min_length=1, description="The server's tools the agent may call, by the server's own names. Nothing else is offered.")
    prefix: str = Field(default="", description="Put before each tool's name for the agent (to avoid clashes with other tools).")
    cost_usd: float | dict[str, float] = Field(default=0.0, description="Price of one call, for every tool or per tool name; MCP reports no cost itself.")
    timeout_s: float | None = Field(default=None, gt=0, description="Per-call timeout; default `tools.timeout_s`.")

    @model_validator(mode="after")
    def _names_and_prices(self) -> McpRef:
        for name in self.allow:
            if not NAME_PATTERN.match(self.prefix + name):
                raise ValueError(f"tool name {self.prefix + name!r} must be 1-64 letters, digits, '_' or '-'; set `prefix` or pick another tool")
        if len(set(self.allow)) != len(self.allow):
            raise ValueError("`allow` lists a tool twice")
        prices = self.cost_usd if isinstance(self.cost_usd, dict) else {}
        if unknown := sorted(set(prices) - set(self.allow)):
            raise ValueError(f"`cost_usd` prices {', '.join(unknown)}, which `allow` does not list")
        if any(price < 0 for price in (self.cost_usd.values() if isinstance(self.cost_usd, dict) else [self.cost_usd])):
            raise ValueError("`cost_usd` cannot be negative")
        return self

    def price(self, tool: str) -> float:
        return float(self.cost_usd.get(tool, 0.0)) if isinstance(self.cost_usd, dict) else float(self.cost_usd)


def parse_mcp_ref(ref: dict[str, Any]) -> McpRef:
    try:
        return McpRef.model_validate(ref)
    except ValidationError as exc:
        parts = []
        for error in exc.errors():
            where = ".".join(str(part) for part in error["loc"])
            parts.append(f"{where}: {error['msg'].removeprefix('Value error, ')}" if where else error["msg"].removeprefix("Value error, "))
        raise ValueError("mcp tools: " + "; ".join(parts)) from None


def mcp_tool_names(ref: dict[str, Any]) -> list[str]:
    """The names the agent sees for an `mcp` ref (the ref is checked, the server is not contacted)."""
    parsed = parse_mcp_ref(ref)
    return [parsed.prefix + name for name in parsed.allow]


# -- the connection ---------------------------------------------------------------------------------------------------------


class McpClient(Protocol):
    async def list_tools(self, *, cursor: str | None = None) -> Any: ...

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None, read_timeout_seconds: float | None = None) -> Any: ...


@asynccontextmanager
async def open_client(config: McpServerConfig) -> AsyncIterator[McpClient]:
    """Connect to the server with the SDK. Replaced in tests by a fake."""
    try:
        from mcp import Client, StdioServerParameters
        from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client
    except ImportError:
        raise ValueError(INSTALL_HINT) from None
    from ragbench.rag_systems.external_http_rag import fill_env

    if config.command is not None:
        argv = config.argv()
        env = {name: fill_env(value, "an MCP server") for name, value in config.env.items()}
        async with Client(StdioServerParameters(command=argv[0], args=argv[1:], env=env or None, cwd=config.cwd)) as client:
            yield client
        return
    assert config.url is not None
    headers = {name: fill_env(value, "an MCP server") for name, value in config.headers.items()}
    async with create_mcp_http_client(headers=headers or None) as http:
        async with Client(streamable_http_client(fill_env(config.url, "an MCP server"), http_client=http)) as client:
            yield client


class McpServer:
    """One server connection, kept open for the run: an event loop in a daemon thread holds the client, and any thread can call it."""

    def __init__(self, config: McpServerConfig):
        self.config = config
        self._loop: asyncio.AbstractEventLoop | None = None
        self._client: McpClient | None = None
        self._stop: asyncio.Event | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self._tools: dict[str, Any] | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        """Connect (once). Raises `ValueError` saying what failed."""
        with self._lock:
            if self._thread is None:
                self._thread = threading.Thread(target=lambda: asyncio.run(self._main()), name="ragbench-mcp", daemon=True)
                self._thread.start()
        if not self._ready.wait(STARTUP_TIMEOUT_S):
            raise ValueError(f"MCP server {self.config.label!r} did not start within {STARTUP_TIMEOUT_S:g} s")
        if self._error is not None:
            if isinstance(self._error, ValueError):
                raise self._error
            raise ValueError(f"cannot start MCP server {self.config.label!r}: {type(self._error).__name__}: {self._error}") from None

    async def _main(self) -> None:
        self._stop = asyncio.Event()
        try:
            async with open_client(self.config) as client:
                self._client = client
                self._loop = asyncio.get_running_loop()
                self._ready.set()
                await self._stop.wait()
        except BaseException as exc:  # noqa: BLE001 (reported by `start`, or by the calls that follow a server that died)
            self._error = exc
            self._ready.set()

    def _run(self, coroutine: Any, timeout: float) -> Any:
        if self._loop is None or self._error is not None:
            coroutine.close()
            raise ConnectionError(f"MCP server {self.config.label!r} is not running" + (f" ({type(self._error).__name__}: {self._error})" if self._error else ""))
        future = asyncio.run_coroutine_threadsafe(coroutine, self._loop)
        try:
            return future.result(timeout)
        except concurrent.futures.TimeoutError:
            future.cancel()
            raise TimeoutError(f"no reply within {timeout:g} s") from None

    def tools(self) -> dict[str, Any]:
        """The server's tools by name (every page), read once."""
        with self._lock:
            known = self._tools
        if known is not None:
            return known
        self.start()
        found: dict[str, Any] = {}

        async def read() -> None:
            assert self._client is not None
            cursor: str | None = None
            for _ in range(MAX_PAGES):
                page = await self._client.list_tools(cursor=cursor)
                found.update({tool.name: tool for tool in page.tools})
                cursor = getattr(page, "next_cursor", None)
                if not cursor:
                    return

        try:
            self._run(read(), STARTUP_TIMEOUT_S)
        except Exception as exc:
            raise ValueError(f"cannot list the tools of MCP server {self.config.label!r}: {type(exc).__name__}: {exc}") from None
        with self._lock:
            self._tools = found
        return found

    def call(self, name: str, arguments: dict[str, Any], timeout: float) -> Any:
        async def invoke() -> Any:
            assert self._client is not None
            return await self._client.call_tool(name, arguments, read_timeout_seconds=timeout)

        return self._run(invoke(), timeout)

    def close(self) -> None:
        if self._loop is not None and self._stop is not None:
            self._loop.call_soon_threadsafe(self._stop.set)
        if self._thread is not None:
            self._thread.join(5)


_servers: dict[str, McpServer] = {}
_servers_lock = threading.Lock()


def server_for(config: McpServerConfig) -> McpServer:
    """The shared connection for this server config: systems and routes that name the same server use one."""
    key = config.model_dump_json()
    with _servers_lock:
        if key not in _servers:
            _servers[key] = McpServer(config)
        return _servers[key]


def close_all_servers() -> None:
    with _servers_lock:
        servers = list(_servers.values())
        _servers.clear()
    for server in servers:
        server.close()


atexit.register(close_all_servers)


# -- the tool ---------------------------------------------------------------------------------------------------------------

FREE_FORM = {"type": "object", "additionalProperties": True}


def _result(raw: Any, cost_usd: float) -> ToolResult:
    parts: list[str] = []
    for block in getattr(raw, "content", None) or []:
        kind = getattr(block, "type", "")
        if kind == "text":
            parts.append(str(block.text))
        else:
            parts.append(f"[{kind} {getattr(block, 'mime_type', '') or ''}".rstrip() + " omitted]")  # an image or audio clip is not text the model can read
    body = "\n".join(parts)
    if getattr(raw, "is_error", False):
        return ToolResult(text=body, error=body or "the tool reported an error", cost_usd=cost_usd)
    data = getattr(raw, "structured_content", None)
    if isinstance(data, dict) and list(data) == ["result"] and isinstance(data["result"], list):
        data = data["result"]  # the SDK wraps a tool's list in an object, which older protocol versions require
    if data is None and body[:1] in ("[", "{"):  # a tool that prints JSON gives the trace its doc ids all the same
        try:
            data = json.loads(body)
        except ValueError:
            data = None
    return ToolResult(text=body, data=data, cost_usd=cost_usd)


class McpTool:
    """One of a server's tools, as the agent calls it. `server` is None under `--mock` and in a config check (no connection is made)."""

    def __init__(self, ref: McpRef, remote_name: str, server: McpServer | None, info: Any | None, *, mock: bool, timeout_s: float):
        self.remote_name = remote_name
        self.server = server
        self.mock = mock
        self.side_effects: SideEffects = ref.server.side_effects
        self.timeout_s = ref.timeout_s or timeout_s  # also what `ToolBox` waits, so the call and the wait for it end together
        self.cost_usd = ref.price(remote_name)
        name = ref.prefix + remote_name
        if info is not None:
            description = " ".join((getattr(info, "description", None) or f"MCP tool {remote_name}.").split())
            self.spec = ToolSpec(name=name, description=description, parameters=dict(getattr(info, "input_schema", None) or FREE_FORM))
        else:
            self.spec = ToolSpec(name=name, description=f"MCP tool {remote_name} (--mock: the server is not started).", parameters=FREE_FORM)

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        if self.mock or self.server is None:
            return ToolResult.ok(f"[mock mcp] {self.remote_name}({json.dumps(args, sort_keys=True, default=str)})", cost_usd=self.cost_usd)
        try:
            raw = self.server.call(self.remote_name, args, self.timeout_s)
        except TimeoutError:
            return ToolResult.fail(f"{self.spec.name} timed out after {self.timeout_s:g} s")
        except Exception as exc:  # noqa: BLE001 (a protocol error, a vanished server: the agent reads it, the run goes on)
            return ToolResult.fail(f"{self.spec.name} failed: {type(exc).__name__}: {exc}")
        return _result(raw, self.cost_usd)


def build_mcp_tools(ref: dict[str, Any], *, discover: bool, mock: bool, timeout_s: float = 10.0) -> list[McpTool]:
    """The tools of an `mcp` ref. With `discover` (and not `mock`) the server is started and the allowed tools must exist on it;
    otherwise the tools are placeholders that carry only a name, which is all a config check needs."""
    parsed = parse_mcp_ref(ref)
    if mock or not discover:
        return [McpTool(parsed, name, None, None, mock=mock, timeout_s=timeout_s) for name in parsed.allow]
    server = server_for(parsed.server)
    try:
        offered = server.tools()
    except ValueError:
        with _servers_lock:  # a server that never came up is tried again next time, not remembered as broken
            _servers.pop(parsed.server.model_dump_json(), None)
        raise
    tools = []
    for name in parsed.allow:
        if name not in offered:
            hint = difflib.get_close_matches(name, list(offered), n=1)
            raise ValueError(
                f"MCP server {parsed.server.label!r} does not offer {name!r}." + (f" Did you mean {hint[0]!r}?" if hint else "") + f" It offers: {', '.join(offered) or 'no tools'}"
            )
        tools.append(McpTool(parsed, name, server, offered[name], mock=False, timeout_s=timeout_s))
    return tools
