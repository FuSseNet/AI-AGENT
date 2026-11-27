"""Локальный MCP-сервер поиска по проекту, транспорт stdio."""

import argparse
import json
from pathlib import Path
import sys

from . import __version__
from .tools import FileTools


PROTOCOL_VERSION = "2025-11-25"
MAX_REQUEST_BYTES = 65536


def error(request_id, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


class SearchServer:
    def __init__(self, project: Path):
        self.files = FileTools(project, report_path=None, project_mode=True)
        self.phase = "new"

    def handle(self, request: object) -> dict | None:
        if (not isinstance(request, dict) or request.get("jsonrpc") != "2.0"
                or not isinstance(request.get("method"), str)):
            return error(None, -32600, "Invalid request")
        method = request["method"]
        params = request.get("params", {})
        if "id" not in request:
            if method == "notifications/initialized" and self.phase == "initializing" and isinstance(params, dict):
                self.phase = "ready"
            return None
        request_id = request["id"]
        if type(request_id) not in (int, str):
            return error(None, -32600, "Invalid request ID")
        if not isinstance(params, dict):
            return error(request_id, -32602, "Expected object parameters")
        if method == "ping":
            result = {}
        elif method == "initialize":
            if self.phase != "new":
                return error(request_id, -32600, "Already initialized")
            client = params.get("clientInfo")
            if (not isinstance(params.get("protocolVersion"), str)
                    or not isinstance(params.get("capabilities"), dict)
                    or not isinstance(client, dict)
                    or not all(isinstance(client.get(key), str) for key in ("name", "version"))):
                return error(request_id, -32602, "Invalid initialization parameters")
            self.phase = "initializing"
            result = {"protocolVersion": PROTOCOL_VERSION, "capabilities": {"tools": {}},
                      "serverInfo": {"name": "assistant-lab-search", "version": __version__}}
        elif self.phase != "ready":
            return error(request_id, -32002, "Initialization required")
        elif method == "tools/list":
            if params.get("cursor") is not None:
                return error(request_id, -32602, "Unknown cursor")
            definition = next(item["function"] for item in self.files.definitions
                              if item["function"]["name"] == "search_text")
            result = {"tools": [{"name": definition["name"], "description": definition["description"],
                                 "inputSchema": definition["parameters"], "annotations": {"readOnlyHint": True}}]}
        elif method == "tools/call":
            if params.get("name") != "search_text":
                return error(request_id, -32602, "Unknown tool")
            outcome = self.files.execute("search_text", params.get("arguments", {}))
            result = {"content": [{"type": "text", "text": json.dumps(outcome, ensure_ascii=False)}],
                      "structuredContent": outcome, "isError": not outcome["ok"]}
        else:
            return error(request_id, -32601, "Method not found")
        return {"jsonrpc": "2.0", "id": request_id, "result": result}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, type=Path, help="Корень доступного проекта")
    server = SearchServer(parser.parse_args().project)
    while line := sys.stdin.buffer.readline(MAX_REQUEST_BYTES + 1):
        oversized = len(line) > MAX_REQUEST_BYTES
        if oversized:
            response = error(None, -32600, "Request too large")
        else:
            try:
                request = json.loads(line.decode("utf-8"))
            except (UnicodeError, ValueError, RecursionError):
                response = error(None, -32700, "Parse error")
            else:
                response = server.handle(request)
        if response is not None:
            sys.stdout.buffer.write(json.dumps(response, ensure_ascii=True).encode("ascii") + b"\n")
            sys.stdout.buffer.flush()
        if oversized:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
