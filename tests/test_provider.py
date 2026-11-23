from __future__ import annotations

import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request

from assistant_lab.config import ConfigError, load_env, local_base_url
from assistant_lab.providers import LocalChatModel, ModelError, _NoRedirects


class ProviderTests(unittest.TestCase):
    def setUp(self):
        self.model = LocalChatModel("http://127.0.0.1:8000/v1", "test-model")

    def test_chat_payload_and_tool_call_response(self):
        message = {"role": "assistant", "content": None, "tool_calls": [{
            "id": "call-1", "type": "function", "function": {
                "name": "read_file", "arguments": '{"path":"README.md"}'}}]}
        response = {"choices": [{"finish_reason": "tool_calls", "message": message}]}
        with patch.object(self.model._opener, "open", return_value=io.BytesIO(
                json.dumps(response).encode())) as request:
            actual = self.model.complete([{"role": "user", "content": "Прочитай"}], [])
        self.assertEqual(actual, message)
        sent = request.call_args.args[0]
        payload = json.loads(sent.data)
        self.assertEqual(sent.full_url, "http://127.0.0.1:8000/v1/chat/completions")
        self.assertEqual(payload["model"], "test-model")
        self.assertEqual(payload["messages"][0]["content"], "Прочитай")
        self.assertFalse(payload["stream"])
        self.assertEqual(request.call_args.kwargs["timeout"], 60)

    def test_server_errors_do_not_reflect_secrets(self):
        error = HTTPError("http://localhost/v1", 401, "secret-value", {},
                          io.BytesIO(b'{"error":"secret-value"}'))
        with patch.object(self.model._opener, "open", side_effect=error):
            with self.assertRaises(ModelError) as context:
                self.model.list_models()
        self.assertNotIn("secret-value", str(context.exception))
        self.assertIn("401", str(context.exception))

    def test_invalid_json_and_truncated_generations_fail(self):
        payloads = [b"<html>not JSON</html>", json.dumps({"choices": [{
            "finish_reason": "length", "message": {"role": "assistant", "content": "partial"}
        }]}).encode()]
        for payload in payloads:
            with self.subTest(payload=payload):
                with patch.object(self.model._opener, "open", return_value=io.BytesIO(payload)):
                    with self.assertRaises(ModelError):
                        self.model.complete([], [])

    def test_response_size_limit(self):
        with patch.object(self.model._opener, "open", return_value=io.BytesIO(b"x" * 524_289)):
            with self.assertRaises(ModelError):
                self.model.list_models()

    def test_redirects_cannot_send_local_context_elsewhere(self):
        with self.assertRaises(ModelError):
            _NoRedirects().redirect_request(Request("http://localhost/v1"), None,
                                           302, "", {}, "https://example.com")

    def test_only_loopback_urls_are_accepted(self):
        for url in ["https://example.com/v1", "http://192.168.1.2/v1",
                    "http://localhost.evil.test/v1", "file:///tmp/model",
                    "http://user:password@localhost/v1", "http://localhost/v1?key=secret",
                    "http://localhost:0/v1", "http://localhost/v1/chat/completions"]:
            with self.subTest(url=url), self.assertRaises(ConfigError):
                local_base_url(url)
        self.assertEqual(local_base_url("http://localhost:8000/"), "http://localhost:8000/v1")
        self.assertEqual(local_base_url("http://[::1]:8000/v1/"), "http://[::1]:8000/v1")

    def test_env_is_literal_and_environment_wins(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text('ASSISTANT_MODEL="file-model"\nASSISTANT_API_KEY=$(not-executed)\n',
                            encoding="utf-8")
            with patch.dict("os.environ", {"ASSISTANT_MODEL": "env-model"}, clear=True):
                values = load_env(path)
            self.assertEqual(values["ASSISTANT_MODEL"], "env-model")
            self.assertEqual(values["ASSISTANT_API_KEY"], "$(not-executed)")


if __name__ == "__main__":
    unittest.main()
