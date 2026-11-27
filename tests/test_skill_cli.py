"""CLI и API-протокол скиллов: решения модели заданы сценарием теста."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from assistant_lab.__main__ import main


def call(identifier, tool_name, **arguments):
    return {"role": "assistant", "content": None, "tool_calls": [{
        "id": identifier, "type": "function", "function": {
            "name": tool_name, "arguments": json.dumps(arguments, ensure_ascii=False)}}]}


class SkillCliTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.project = self.root / "project"
        self.project.mkdir()
        (self.project / "README.md").write_bytes(b"Study project.\n")
        (self.project / "old.md").write_bytes(b"Store tasks in JSON.\n")
        (self.project / "new.md").write_bytes(b"Store tasks in SQLite.\nExport CSV.\n")
        self.env = self.root / ".env"
        self.env.write_text("ASSISTANT_BASE_URL=http://localhost:8000/v1\nASSISTANT_MODEL=test\n",
                            encoding="utf-8")

    def run_script(self, arguments, responses):
        pending = list(responses)
        observed = []
        output = io.StringIO()

        def respond(request, timeout):
            observed.append(json.loads(request.data))
            message = pending.pop(0)
            return io.BytesIO(json.dumps({"choices": [{"message": message,
                "finish_reason": "tool_calls" if message.get("tool_calls") else "stop"}]}).encode())

        with patch("assistant_lab.__main__.ROOT", self.root), patch("os.environ", {}), \
                patch("urllib.request.OpenerDirector.open", side_effect=respond), \
                contextlib.redirect_stdout(output):
            code = main([*arguments, "--project", str(self.project), "--env-file", str(self.env)])
        self.assertFalse(pending, "Не все ответы сценария были использованы")
        return code, output.getvalue(), observed

    def test_natural_project_goal_activates_skill_then_saves_report(self):
        code, output, requests = self.run_script(["Опиши учебный проект и сохрани обзор."], [
            call("skill", "load_skill", name="project-review"),
            call("list", "list_files"),
            call("read", "read_file", path="README.md"),
            call("save", "save_report", content="Краткий обзор прочитанного проекта."),
            {"role": "assistant", "content": "Обзор сохранён."},
        ])
        self.assertEqual(code, 0, output)
        self.assertIn("Выбранный скилл: project-review", output)
        self.assertEqual({tool["function"]["name"] for tool in requests[0]["tools"]}, {"load_skill"})
        self.assertEqual({tool["function"]["name"] for tool in requests[1]["tools"]},
                         {"list_files", "read_file", "search_text", "save_report"})
        for request in requests[1:]:
            self.assertTrue(json.loads(request["messages"][-1]["content"])["ok"])
        self.assertTrue(list(self.root.glob("workspace/reports/*/report.md")))

    def test_natural_comparison_goal_reads_both_documents(self):
        code, output, requests = self.run_script(["Сравни old.md и new.md."], [
            call("skill", "load_skill", name="compare-documents"),
            call("list", "list_files"),
            call("old", "read_file", path="old.md"),
            call("new", "read_file", path="new.md"),
            call("save", "save_report", content="JSON заменён на SQLite, добавлен CSV."),
            {"role": "assistant", "content": "Сравнение сохранено."},
        ])
        self.assertEqual(code, 0, output)
        self.assertIn("Выбранный скилл: compare-documents", output)
        report = next(self.root.glob("workspace/reports/*/report.md")).read_text(encoding="utf-8")
        self.assertIn("`old.md`", report)
        self.assertIn("`new.md`", report)
        self.assertEqual(requests[1]["messages"][-1]["tool_call_id"], "skill")

    def test_explicit_skill_skips_model_selection_request(self):
        code, output, requests = self.run_script(["--skill", "project-review", "Опиши проект."], [
            call("list", "list_files"), call("read", "read_file", path="README.md"),
            call("save", "save_report", content="Обзор."),
            {"role": "assistant", "content": "Готово."},
        ])
        self.assertEqual(code, 0, output)
        self.assertNotIn("load_skill", {tool["function"]["name"] for tool in requests[0]["tools"]})
        self.assertIn("project-review", requests[0]["messages"][0]["content"])

    def test_clarification_prints_question_without_fake_report(self):
        code, output, _ = self.run_script(["Сравни документы."], [
            call("skill", "load_skill", name="compare-documents"),
            {"role": "assistant", "content": "Назови два относительных пути к документам."},
        ])
        self.assertEqual(code, 1)
        self.assertIn("Назови два относительных пути", output)
        self.assertFalse(list(self.root.glob("workspace/reports/*/report.md")))

    def test_no_skill_selection_cannot_be_reported_as_success(self):
        code, output, _ = self.run_script(["Сделай обзор."], [
            {"role": "assistant", "content": "Я всё сделал!"},
        ])
        self.assertEqual(code, 1)
        self.assertIn("Выбранный скилл: не выбран", output)
        self.assertFalse(list(self.root.glob("workspace/reports/*/report.md")))

    def test_offline_search_demo_confirms_a_hit_before_writing_the_report(self):
        code, output, _ = self.run_script(["--demo-search", "CSV"], [])
        self.assertEqual(code, 0, output)
        report = next(self.root.glob("workspace/reports/*/report.md")).read_text(encoding="utf-8")
        self.assertIn("> Export CSV.", report)
        self.assertIn("`new.md`: строки 2–2", report)
        self.assertNotIn("Store tasks", report)
        log = next(self.root.glob("runs/*.jsonl")).read_text(encoding="utf-8")
        calls = [event["tool"] for event in map(json.loads, log.splitlines())
                 if event["event"] == "tool_finished" and event["ok"]]
        self.assertEqual(calls, ["list_files", "search_text", "read_file", "save_report"])
        self.assertNotIn("CSV", log)

    def test_offline_search_demo_miss_does_not_create_report(self):
        code, output, _ = self.run_script(["--demo-search", "absent"], [])
        self.assertEqual(code, 1, output)
        self.assertIn("Совпадений", output)
        self.assertFalse(list(self.root.glob("workspace/reports/*/report.md")))

    def test_list_skills_does_not_read_env_or_call_api(self):
        self.env.write_text("BROKEN LINE WITHOUT EQUALS\n", encoding="utf-8")
        output = io.StringIO()
        with patch("assistant_lab.__main__.load_env", side_effect=AssertionError("env прочитан")), \
                patch("urllib.request.OpenerDirector.open", side_effect=AssertionError("API вызван")), \
                contextlib.redirect_stdout(output):
            code = main(["--list-skills", "--env-file", str(self.env)])
        self.assertEqual(code, 0)
        self.assertIn("project-review:", output.getvalue())
        self.assertIn("compare-documents:", output.getvalue())


if __name__ == "__main__":
    unittest.main()
