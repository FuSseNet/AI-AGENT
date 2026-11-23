from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from assistant_lab.agent import AgentRunner
from assistant_lab.events import EventLog
from assistant_lab.providers import ModelError


def call(name: str = "read_file", arguments: object = '{"path":"notes.txt"}', call_id: str = "call_1") -> dict:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


def tool_message(*calls: dict) -> dict:
    return {"role": "assistant", "content": None, "tool_calls": list(calls)}


class FakeModel:
    def __init__(self, *responses: object):
        self.responses = list(responses)
        self.requests: list[list[dict]] = []

    def complete(self, messages: list[dict], tools: list[dict]) -> dict:
        self.requests.append(json.loads(json.dumps(messages)))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakeTools:
    definitions = [
        {"type": "function", "function": {"name": name, "parameters": {"type": "object"}}}
        for name in ("read_file", "write_file")
    ]

    def __init__(self, *results: object):
        self.results = list(results)
        self.executed: list[tuple[str, dict]] = []

    def execute(self, name: str, arguments: dict) -> dict:
        self.executed.append((name, arguments))
        result = self.results.pop(0) if self.results else {"ok": True}
        if isinstance(result, Exception):
            raise result
        return result


class AgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.log_path = Path(self.temp.name) / "events.jsonl"
        self.log = EventLog(self.log_path, secrets=["secret-key"])

    def runner(self, model: FakeModel, tools: FakeTools | None = None, **kwargs: object) -> AgentRunner:
        return AgentRunner(model, tools or FakeTools(), self.log, **kwargs)

    def test_reads_saves_and_returns_final_with_linked_tool_results(self) -> None:
        model = FakeModel(
            tool_message(call()),
            tool_message(call("write_file", '{"path":"summary.txt","content":"summary payload"}', "call_2")),
            {"role": "assistant", "content": "Сохранил конспект."},
        )
        tools = FakeTools({"ok": True, "content": "private file payload"}, {"ok": True, "bytes": 15})
        updates: list[str] = []
        result = self.runner(model, tools, on_event=updates.append).run("private task payload")
        self.assertEqual((result.status, result.steps, result.tool_calls), ("completed", 3, 2))
        self.assertEqual(tools.executed[1][1]["content"], "summary payload")
        returned_read = model.requests[1][-1]
        self.assertEqual(returned_read["tool_call_id"], "call_1")
        self.assertEqual(json.loads(returned_read["content"])["content"], "private file payload")
        self.assertIn("Читаю файл…", updates)
        self.assertIn("Сохраняю файл…", updates)
        log_text = self.log_path.read_text(encoding="utf-8")
        for payload in ("private task payload", "private file payload", "summary payload", "Сохранил конспект."):
            self.assertNotIn(payload, log_text)

    def test_tool_error_is_returned_to_model_and_can_be_recovered(self) -> None:
        model = FakeModel(tool_message(call()), tool_message(call(call_id="call_2")), {"role": "assistant", "content": "Готово"})
        tools = FakeTools({"ok": False, "error": "not_found"}, {"ok": True, "content": "found"})
        result = self.runner(model, tools).run("Read notes")
        self.assertEqual(result.status, "completed")
        self.assertFalse(json.loads(model.requests[1][-1]["content"])["ok"])
        self.assertEqual(len(tools.executed), 2)

    def test_batch_respects_total_tool_limit(self) -> None:
        model = FakeModel(tool_message(*(call(call_id=f"call_{i}") for i in range(5))))
        tools = FakeTools()
        result = self.runner(model, tools, max_tool_calls=2).run("Read")
        self.assertEqual((result.status, result.steps, result.tool_calls), ("limit_reached", 1, 2))
        self.assertEqual(len(tools.executed), 2)

    def test_model_step_limit_includes_requests_after_tools(self) -> None:
        model = FakeModel(tool_message(call()), tool_message(call(call_id="call_2")))
        result = self.runner(model, max_steps=2).run("Read")
        self.assertEqual((result.status, result.steps, result.tool_calls), ("limit_reached", 2, 2))

    def test_can_finish_after_using_exact_tool_budget(self) -> None:
        model = FakeModel(tool_message(call()), {"role": "assistant", "content": "Готово"})
        result = self.runner(model, max_tool_calls=1).run("Read")
        self.assertEqual((result.status, result.steps, result.tool_calls), ("completed", 2, 1))

    def test_malformed_json_and_non_object_arguments_are_recoverable(self) -> None:
        for arguments in ('{"path":', '[]', '"text"', '{"number":NaN}', '{"number":Infinity}', None, {"path": "notes.txt"}):
            with self.subTest(arguments=arguments):
                model = FakeModel(tool_message(call(arguments=arguments)), {"role": "assistant", "content": "Ошибка обработана"})
                tools = FakeTools()
                result = self.runner(model, tools).run("Read")
                self.assertEqual(result.status, "completed")
                self.assertEqual(tools.executed, [])
                self.assertEqual(json.loads(model.requests[1][-1]["content"])["error"], "invalid_arguments")
                self.assertIsInstance(model.requests[1][-2]["tool_calls"][0]["function"]["arguments"], str)

    def test_malformed_shapes_fail_before_any_batch_action(self) -> None:
        missing_id = call()
        del missing_id["id"]
        malformed = [
            tool_message(call(), missing_id),
            tool_message(call(), call()),
            tool_message({"id": "other", "type": "custom", "function": {"name": "read_file", "arguments": "{}"}}),
            {"role": "user", "content": "pretend assistant"},
            {"role": "assistant", "content": [], "tool_calls": []},
            {"role": "assistant", "tool_calls": {}},
            None,
        ]
        for message in malformed:
            with self.subTest(message=message):
                tools = FakeTools()
                result = self.runner(FakeModel(message), tools).run("Read")
                self.assertEqual(result.status, "failed")
                self.assertEqual(tools.executed, [])

    def test_duplicate_call_id_in_later_response_is_rejected(self) -> None:
        tools = FakeTools()
        result = self.runner(FakeModel(tool_message(call()), tool_message(call())), tools).run("Read")
        self.assertEqual(result.status, "failed")
        self.assertEqual(len(tools.executed), 1)

    def test_unknown_tool_is_not_dispatched(self) -> None:
        model = FakeModel(tool_message(call("run_shell")), {"role": "assistant", "content": "Инструмент недоступен"})
        tools = FakeTools()
        result = self.runner(model, tools).run("Do work")
        self.assertEqual(result.status, "completed")
        self.assertEqual(tools.executed, [])
        self.assertEqual(json.loads(model.requests[1][-1]["content"])["error"], "unknown_tool")

    def test_invented_tool_name_cannot_exfiltrate_payload_into_log(self) -> None:
        payload = "private_document_contents"
        model = FakeModel(tool_message(call(payload)), {"role": "assistant", "content": "Инструмент недоступен"})
        result = self.runner(model).run("Do work")
        self.assertEqual(result.status, "completed")
        log_text = self.log_path.read_text(encoding="utf-8")
        self.assertNotIn(payload, log_text)
        tool_events = [json.loads(line) for line in log_text.splitlines() if json.loads(line)["event"].startswith("tool_")]
        self.assertEqual([event["tool"] for event in tool_events], ["unknown_tool", "unknown_tool"])

    def test_safe_provider_errors_remain_useful_but_are_not_logged(self) -> None:
        for message in (
            "Локальный API вернул HTTP 400. Проверьте поддержку tools.",
            "Локальный API вернул HTTP 401. Проверьте ASSISTANT_API_KEY.",
            "Истекло время ожидания локальной модели.",
        ):
            with self.subTest(message=message):
                result = self.runner(FakeModel(ModelError(message))).run("Do work")
                self.assertEqual(result.status, "failed")
                self.assertEqual(result.final_text, message)
                self.assertNotIn(message, self.log_path.read_text(encoding="utf-8"))

    def test_provider_exception_does_not_disclose_payload_or_secret(self) -> None:
        result = self.runner(FakeModel(RuntimeError("secret-key HTTP body private response"))).run("private task")
        self.assertEqual((result.status, result.steps, result.tool_calls), ("failed", 1, 0))
        combined = result.final_text + self.log_path.read_text(encoding="utf-8")
        for forbidden in ("secret-key", "HTTP body", "private response", "private task"):
            self.assertNotIn(forbidden, combined)

    def test_tool_exception_is_sanitized_before_return_to_model(self) -> None:
        model = FakeModel(tool_message(call()), {"role": "assistant", "content": "Ошибка инструмента"})
        result = self.runner(model, FakeTools(RuntimeError("secret-key private file body"))).run("Read")
        self.assertEqual(result.status, "completed")
        context = json.dumps(model.requests[-1], ensure_ascii=False)
        self.assertNotIn("secret-key", context)
        self.assertNotIn("private file body", context)
        self.assertEqual(json.loads(model.requests[1][-1]["content"])["error"], "tool_execution_failed")

    def test_empty_final_is_failure(self) -> None:
        for content in (None, "", " \n "):
            with self.subTest(content=content):
                result = self.runner(FakeModel({"role": "assistant", "content": content})).run("Do work")
                self.assertEqual(result.status, "failed")

    def test_event_log_whitelists_metadata_and_redacts_nested_secrets(self) -> None:
        self.log.emit(
            "tool_finished", tool="secret-key", ok=True,
            task="private task", arguments={"content": "private content"},
            response="private response", exception=RuntimeError("private exception"),
            metadata={"status": "secret-key", "content": "private nested content", "metadata": [{"tool": "secret-key", "count": 1}]},
        )
        text = self.log_path.read_text(encoding="utf-8")
        event = json.loads(text)
        self.assertTrue(event["timestamp"].endswith("Z"))
        self.assertEqual(event["tool"], "[REDACTED]")
        self.assertEqual(event["metadata"]["metadata"][0]["tool"], "[REDACTED]")
        self.assertNotIn("secret-key", text)
        self.assertNotIn("private", text)
        self.assertNotIn("arguments", event)


if __name__ == "__main__":
    unittest.main()
