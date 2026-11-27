"""Вызов search_text на встроенном MCP-сервере через stdio."""

import argparse
import asyncio
import json
from pathlib import Path
import sys

from . import __version__
from .mcp_server import PROTOCOL_VERSION


class ProtocolError(Exception):
    pass


async def _exchange(process, arguments: dict) -> dict:
    async def send(method: str, params: dict, request_id: int | None = None):
        message = {"jsonrpc": "2.0", "method": method, "params": params}
        if request_id is not None:
            message["id"] = request_id
        process.stdin.write(json.dumps(message).encode("ascii") + b"\n")
        await process.stdin.drain()
        if request_id is None:
            return
        reply = json.loads((await process.stdout.readline()).decode("utf-8"))
        if (not isinstance(reply, dict) or reply.get("jsonrpc") != "2.0"
                or type(reply.get("id")) is not int or reply["id"] != request_id
                or not isinstance(reply.get("result"), dict)):
            raise ProtocolError
        return reply["result"]

    hello = await send("initialize", {"protocolVersion": PROTOCOL_VERSION, "capabilities": {},
                       "clientInfo": {"name": "assistant-lab", "version": __version__}}, 1)
    if (hello.get("protocolVersion") != PROTOCOL_VERSION
            or not isinstance(hello.get("capabilities"), dict) or "tools" not in hello["capabilities"]):
        raise ProtocolError
    await send("notifications/initialized", {})
    catalog = await send("tools/list", {}, 2)
    tools = catalog.get("tools")
    if not isinstance(tools, list) or not any(
            isinstance(tool, dict) and tool.get("name") == "search_text" for tool in tools):
        raise ProtocolError
    result = await send("tools/call", {"name": "search_text", "arguments": arguments}, 3)
    outcome = result.get("structuredContent")
    if (not isinstance(outcome, dict) or type(outcome.get("ok")) is not bool
            or result.get("isError", False) != (not outcome["ok"])):
        raise ProtocolError
    return outcome


async def _search(project: Path, arguments: dict) -> dict:
    # Import our installed/bundled code from its own directory. The project is
    # data passed as an argument and never the working directory for imports.
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-E", "-s", "-B", "-m", "assistant_lab.mcp_server",
        "--project", str(project.absolute()), cwd=Path(__file__).resolve().parent.parent,
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL, limit=2 * 1024 * 1024,
    )
    try:
        return await asyncio.wait_for(_exchange(process, arguments), timeout=20)
    finally:
        process.stdin.close()
        try:
            await asyncio.wait_for(process.wait(), timeout=2)
        except TimeoutError:
            process.kill()
            await process.wait()


def search_project(project: Path, arguments: dict) -> dict:
    try:
        return asyncio.run(_search(project, arguments))
    except TimeoutError:
        return {"ok": False, "error": "Истекло время ожидания поиска через MCP."}
    except (OSError, ValueError, ProtocolError, RecursionError):
        return {"ok": False, "error": "Не удалось выполнить запрос к локальному MCP-серверу."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("query", help="Однострочная фраза с учётом регистра")
    parser.add_argument("--path", default=".", help="Файл или подкаталог внутри проекта")
    args = parser.parse_args()
    result = search_project(args.project, {"query": args.query, "path": args.path})
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
