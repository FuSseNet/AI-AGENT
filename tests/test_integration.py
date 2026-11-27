"""Проверка всего цикла с имитацией HTTP-транспорта, без настоящей модели."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from assistant_lab.__main__ import main


class IntegrationTests(unittest.TestCase):
    def test_java_review_reads_manifest_and_source_fragments_with_cli_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "java-project"
            source = project / "src" / "App.java"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"class App {\n  int count = 1;\n}\n")
            project.joinpath("pom.xml").write_text("<project />\n", encoding="utf-8")
            env = root / ".env"
            env.write_text("ASSISTANT_BASE_URL=http://localhost:8000/v1\n"
                           "ASSISTANT_MODEL=test\nASSISTANT_READ_BUDGET=1\n", encoding="utf-8")
            script = [
                ("list", "list_files", {"path": ".", "limit": 10}),
                ("manifest", "read_file", {"path": "pom.xml"}),
                ("find", "search_text", {"query": "int count"}),
                ("source", "read_file", {}),
                ("save", "save_report", {"content": "Java-проект; изучена только одна строка исходника."}),
            ]
            responses = 0

            def respond(request, timeout):
                nonlocal responses
                data = json.loads(request.data)
                index = responses
                responses += 1
                if index:
                    previous = json.loads(data["messages"][-1]["content"])
                    self.assertTrue(previous["ok"], previous)
                    if index == 1:
                        self.assertEqual(previous["remaining_read_bytes"], 128)
                    if index == 3:
                        self.assertEqual(previous["matches"], [{"path": "src/App.java", "line": 2}])
                        hit = previous["matches"][0]
                        script[index][2].update(path=hit["path"], start_line=hit["line"], max_lines=1)
                    if index == 4:
                        self.assertEqual(previous["content"], "  int count = 1;\n")
                        self.assertEqual(previous["next_line"], 3)
                if index == len(script):
                    message = {"role": "assistant", "content": "Готово."}
                else:
                    call_id, name, arguments = script[index]
                    message = {"role": "assistant", "content": None, "tool_calls": [{
                        "id": call_id, "type": "function", "function": {
                            "name": name, "arguments": json.dumps(arguments, ensure_ascii=False)}}]}
                return io.BytesIO(json.dumps({"choices": [{"message": message,
                    "finish_reason": "stop" if index == len(script) else "tool_calls"}]}).encode())

            with patch("assistant_lab.__main__.ROOT", root), patch("os.environ", {}), \
                    patch("urllib.request.OpenerDirector.open", side_effect=respond), \
                    contextlib.redirect_stdout(io.StringIO()):
                code = main(["--describe-project", "--project", str(project),
                             "--env-file", str(env), "--read-budget", "128"])
            self.assertEqual(code, 0)
            report = next(root.glob("workspace/reports/*/report.md")).read_text(encoding="utf-8")
            self.assertIn("`src/App.java`: строки 2–2", report)
            log = next(root.glob("runs/*.jsonl")).read_text(encoding="utf-8")
            self.assertIn('"tool": "search_text"', log)
            self.assertNotIn("int count", log)
            self.assertEqual(source.read_text(encoding="utf-8"), "class App {\n  int count = 1;\n}\n")

    def test_project_review_lists_reads_source_and_saves_without_modifying_project(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "project"
            project.mkdir()
            cmake = "add_executable(example worker.cpp)\n"
            source = "int main() { return 0; }\n"
            project.joinpath("CMakeLists.txt").write_text(cmake, encoding="utf-8")
            project.joinpath("worker.cpp").write_text(source, encoding="utf-8")
            env = root / ".env"
            env.write_text("ASSISTANT_BASE_URL=http://localhost:8000/v1\nASSISTANT_MODEL=test\n",
                           encoding="utf-8")
            requests = []

            def respond(request, timeout):
                data = json.loads(request.data)
                requests.append(data)
                names = {tool["function"]["name"] for tool in data["tools"]}
                self.assertIn("list_files", names)
                script = [
                    ("list-1", "list_files", {}),
                    ("cmake-1", "read_file", {"path": "CMakeLists.txt"}),
                    ("source-1", "read_file", {"path": "worker.cpp"}),
                    ("save-1", "save_report", {"content": "# Обзор\nПростая программа C++. Запуск не проверялся."}),
                ]
                index = len(requests) - 1
                if index:
                    tool_result = json.loads(data["messages"][-1]["content"])
                    self.assertTrue(tool_result["ok"], tool_result)
                    self.assertEqual(data["messages"][-1]["tool_call_id"], script[index - 1][0])
                if index == len(script):
                    message = {"role": "assistant", "content": "Обзор сохранён."}
                    reason = "stop"
                else:
                    call_id, name, arguments = script[index]
                    message = {"role": "assistant", "content": None, "tool_calls": [{
                        "id": call_id, "type": "function", "function": {
                            "name": name, "arguments": json.dumps(arguments, ensure_ascii=False)}}]}
                    reason = "tool_calls"
                return io.BytesIO(json.dumps({"choices": [{"finish_reason": reason, "message": message}]}).encode())

            output = io.StringIO()
            with patch("assistant_lab.__main__.ROOT", root), patch("os.environ", {}), \
                    patch("urllib.request.OpenerDirector.open", side_effect=respond), \
                    contextlib.redirect_stdout(output):
                code = main(["--describe-project", "--project", str(project), "--env-file", str(env)])
            self.assertEqual(code, 0)
            self.assertEqual(len(requests), 5)
            report = next(root.glob("workspace/reports/*/report.md")).read_text(encoding="utf-8")
            self.assertIn("worker.cpp", report)
            self.assertIn("CMakeLists.txt", report)
            self.assertIn("worker.cpp", output.getvalue())
            self.assertEqual(project.joinpath("worker.cpp").read_text(encoding="utf-8"), source)
            self.assertEqual(project.joinpath("CMakeLists.txt").read_text(encoding="utf-8"), cmake)

    def test_check_lists_models_without_model_setting_or_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env = root / ".env"
            env.write_text("ASSISTANT_BASE_URL=http://localhost:8000/v1\n", encoding="utf-8")
            output = io.StringIO()
            with patch("assistant_lab.__main__.ROOT", root), \
                    patch("os.environ", {}), \
                    patch("urllib.request.OpenerDirector.open", return_value=io.BytesIO(
                        b'{"data":[{"id":"local-test-model"}]}')) as request, \
                    contextlib.redirect_stdout(output):
                code = main(["--check", "--env-file", str(env)])
            self.assertEqual(code, 0)
            self.assertIn("local-test-model", output.getvalue())
            self.assertEqual(request.call_args.args[0].get_method(), "GET")
            self.assertEqual(request.call_args.args[0].full_url, "http://localhost:8000/v1/models")
            self.assertFalse((root / "workspace").exists())

    def test_api_roundtrip_reads_writes_and_links_tool_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "workspace" / "project"
            project.mkdir(parents=True)
            project.joinpath("README.md").write_text("Сервис хранит учебные задания.", encoding="utf-8")
            root.joinpath(".env").write_text(
                "ASSISTANT_BASE_URL=http://localhost:8000/v1\nASSISTANT_MODEL=test\n",
                encoding="utf-8")
            requests = []

            def respond(request, timeout):
                data = json.loads(request.data)
                requests.append(data)
                messages = data["messages"]
                if len(requests) == 1:
                    name, arguments, call_id = "read_file", {"path": "README.md"}, "read-1"
                elif len(requests) == 2:
                    self.assertEqual(messages[-1]["tool_call_id"], "read-1")
                    result = json.loads(messages[-1]["content"])
                    self.assertIn("учебные задания", result["content"])
                    name, arguments, call_id = "save_report", {"content": "# Отчёт\nХранит учебные задания."}, "save-1"
                else:
                    self.assertEqual(messages[-1]["tool_call_id"], "save-1")
                    self.assertTrue(json.loads(messages[-1]["content"])["ok"])
                    return io.BytesIO(json.dumps({"choices": [{"finish_reason": "stop", "message": {
                        "role": "assistant", "content": "Отчёт сохранён."}}]}).encode())
                return io.BytesIO(json.dumps({"choices": [{"finish_reason": "tool_calls", "message": {
                    "role": "assistant", "content": None, "tool_calls": [{"id": call_id,
                    "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}]
                }}]}).encode())

            with patch("assistant_lab.__main__.ROOT", root), \
                    patch("os.environ", {}), \
                    patch("urllib.request.OpenerDirector.open", side_effect=respond), \
                    contextlib.redirect_stdout(io.StringIO()):
                code = main(["--env-file", str(root / ".env"), "--project", str(project)])
            self.assertEqual(code, 0)
            self.assertEqual(len(requests), 3)
            reports = list(root.glob("workspace/reports/*/report.md"))
            self.assertEqual(len(reports), 1)
            self.assertIn("Хранит учебные задания", reports[0].read_text(encoding="utf-8"))
            log = next(root.glob("runs/*.jsonl")).read_text(encoding="utf-8")
            self.assertNotIn("Сервис хранит", log)
            self.assertNotIn("# Отчёт", log)

    def test_claim_of_success_without_report_is_not_cli_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "project"
            project.mkdir()
            with patch("assistant_lab.__main__.ROOT", root), \
                    patch("os.environ", {}), \
                    patch("assistant_lab.providers.DemoModel.complete", return_value={
                        "role": "assistant", "content": "Я всё сохранил!"}), \
                    contextlib.redirect_stdout(io.StringIO()):
                code = main(["--demo", "--env-file", str(root / ".env"), "--project", str(project)])
            self.assertEqual(code, 1)
            self.assertFalse(list(root.glob("workspace/reports/*/report.md")))

    def test_saving_without_reading_cannot_complete_standard_task(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "project"
            project.mkdir()
            project.joinpath("README.md").write_text("Настоящий исходник", encoding="utf-8")
            scripted = [{"role": "assistant", "content": None, "tool_calls": [{
                "id": "fabricated-report", "type": "function", "function": {
                    "name": "save_report", "arguments": '{"content":"Invented report"}'}}]},
                {"role": "assistant", "content": "Я всё сохранил!"}]
            with patch("assistant_lab.__main__.ROOT", root), \
                    patch("os.environ", {}), \
                    patch("assistant_lab.providers.DemoModel.complete", side_effect=scripted), \
                    contextlib.redirect_stdout(io.StringIO()):
                code = main(["--demo", "--env-file", str(root / ".env"), "--project", str(project)])
            self.assertEqual(code, 1)
            self.assertFalse(list(root.glob("workspace/reports/*/report.md")))

    def test_demo_missing_source_fails_without_a_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "project"
            project.mkdir()
            with patch("assistant_lab.__main__.ROOT", root), \
                    patch("os.environ", {}), contextlib.redirect_stdout(io.StringIO()):
                code = main(["--demo", "--env-file", str(root / ".env"), "--project", str(project)])
            self.assertEqual(code, 1)
            self.assertFalse(list(root.glob("workspace/reports/*/report.md")))


if __name__ == "__main__":
    unittest.main()
