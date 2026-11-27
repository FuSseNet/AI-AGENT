"""Проверки MCP на границе процессов с временным учебным проектом."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from assistant_lab.mcp_client import search_project
from assistant_lab.mcp_server import MAX_REQUEST_BYTES, PROTOCOL_VERSION


class McpSearchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.project = Path(self.temp.name)
        (self.project / "guide.md").write_bytes("Модель\nБаза SQLite.\n".encode())

    def server(self, wire):
        return subprocess.run(
            [sys.executable, "-m", "assistant_lab.mcp_server", "--project", str(self.project)],
            input=wire, capture_output=True, timeout=10,
        )

    def test_stdio_lifecycle_catalog_and_read_only_dispatch(self):
        calls = [
            ("tools/list", {}),
            ("initialize", {"protocolVersion": "other", "clientInfo": {"name": "test", "version": "1"},
                            "capabilities": {}}),
            ("tools/list", {}),
            ("notifications/initialized", {}),
            ("ping", {}),
            ("tools/list", {}),
            ("tools/call", {"name": "save_report", "arguments": {"content": "x"}}),
            ("tools/call", {"name": "search_text", "arguments": {"query": "SQLite"}}),
            ("tools/call", {"name": "search_text", "arguments": {"query": ""}}),
        ]
        wire = bytearray(b"!\n")
        for index, (method, params) in enumerate(calls):
            message = {"jsonrpc": "2.0", "method": method, "params": params}
            if method != "notifications/initialized":
                message["id"] = index
            wire.extend(json.dumps(message).encode() + b"\n")
        completed = self.server(wire)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        replies = [json.loads(line) for line in completed.stdout.splitlines()]
        self.assertEqual([r["id"] for r in replies], [None, 0, 1, 2, 4, 5, 6, 7, 8])
        by_id = {reply["id"]: reply for reply in replies}
        self.assertEqual(by_id[None]["error"]["code"], -32700)
        self.assertIn("error", by_id[0])
        self.assertIn("error", by_id[2])
        self.assertEqual(by_id[1]["result"]["protocolVersion"], PROTOCOL_VERSION)
        self.assertEqual(by_id[4]["result"], {})
        catalog = by_id[5]["result"]["tools"]
        self.assertEqual([tool["name"] for tool in catalog], ["search_text"])
        self.assertEqual(catalog[0]["inputSchema"]["required"], ["query"])
        self.assertIn("error", by_id[6])
        result = by_id[7]["result"]
        self.assertFalse(result["isError"])
        self.assertEqual(json.loads(result["content"][0]["text"]), result["structuredContent"])
        self.assertEqual(result["structuredContent"]["matches"], [{"path": "guide.md", "line": 2}])
        self.assertTrue(by_id[8]["result"]["isError"])
        self.assertEqual([entry.name for entry in self.project.iterdir()], ["guide.md"])

    def test_oversized_frame_terminates_session(self):
        completed = self.server(b"x" * (MAX_REQUEST_BYTES + 1))
        self.assertNotEqual(completed.returncode, 0)
        self.assertEqual(json.loads(completed.stdout)["error"]["code"], -32600)

    def test_client_propagates_server_error_and_distinguishes_no_matches(self):
        self.assertFalse(search_project(self.project, {"path": "../outside", "query": "x"})["ok"])
        missing = search_project(self.project, {"query": "absent"})
        self.assertTrue(missing["ok"], missing)
        self.assertEqual(missing["matches"], [])
        self.assertFalse(missing["truncated"])


if __name__ == "__main__":
    unittest.main()
