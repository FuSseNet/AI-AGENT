"""Model planning and execution through scripted HTTP, without a real model.

The responses prescribe the model's choices. These tests verify CLI boundaries,
artifact routing and protocol separation rather than semantic planning quality.
"""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from assistant_lab.__main__ import main


def message(value):
    return {"role": "assistant", "content": json.dumps(value, ensure_ascii=False)}


def call(identifier, tool_name, **arguments):
    return {"role": "assistant", "content": None, "tool_calls": [{
        "id": identifier, "type": "function", "function": {
            "name": tool_name, "arguments": json.dumps(arguments, ensure_ascii=False),
        },
    }]}


def review(report):
    return [
        call("list", "list_files"),
        call("read", "read_file", path="README.md"),
        call("save", "save_report", content=report),
        {"role": "assistant", "content": "Обзор сохранён."},
    ]


def source_step(alias="project-1", task="Составь обзор и сохрани отчёт."):
    return {"skill": "project-review", "task": task, "project": alias, "depends_on": []}


class PlannerCliTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="private-planner-root-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.projects = []
        for name in ("alpha", "beta"):
            project = self.root / "owner-projects" / name
            project.mkdir(parents=True)
            (project / "README.md").write_text(f"PRIVATE_SOURCE_{name}\n", encoding="utf-8")
            self.projects.append(project)
        self.default_project = self.root / "workspace" / "project"
        self.default_project.mkdir(parents=True)
        (self.default_project / "README.md").write_text("PRIVATE_DEFAULT_SOURCE\n", encoding="utf-8")
        self.env = self.root / "test.settings"
        self.env.write_text(
            "ASSISTANT_BASE_URL=http://localhost:8000/v1\n"
            "ASSISTANT_MODEL=planner-and-worker-test\n",
            encoding="utf-8",
        )

    def arguments(self, projects=None, goal="Опиши проекты, затем сопоставь их возможности."):
        arguments = ["--auto-plan", "--env-file", str(self.env)]
        if projects is not None:
            for project in projects:
                arguments.extend(["--project", str(project)])
        if goal is not None:
            arguments.append(goal)
        return arguments

    def invoke(self, arguments, responses):
        pending = list(responses)
        observed = []
        stdout, stderr = io.StringIO(), io.StringIO()

        def respond(request, timeout):
            self.assertEqual(request.get_method(), "POST")
            self.assertEqual(request.full_url, "http://localhost:8000/v1/chat/completions")
            data = json.loads(request.data)
            observed.append(data)
            self.assertEqual(data["model"], "planner-and-worker-test")
            self.assertTrue(pending, "Unexpected extra model request")
            response = pending.pop(0)
            return io.BytesIO(json.dumps({"choices": [{
                "message": response,
                "finish_reason": "tool_calls" if response.get("tool_calls") else "stop",
            }]}).encode())

        with patch("assistant_lab.__main__.ROOT", self.root), patch("os.environ", {}), \
                patch("urllib.request.OpenerDirector.open", side_effect=respond), \
                contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                code = main(arguments)
            except SystemExit as stopped:
                code = stopped.code
        self.assertFalse(pending, "The script contains unused responses")
        return code, stdout.getvalue(), stderr.getvalue(), observed

    def run_dir(self):
        folders = list(self.root.glob("workspace/reports/*"))
        self.assertEqual(len(folders), 1)
        return folders[0]

    def assert_planning_request(self, request, projects):
        self.assertEqual(request["tools"], [])
        self.assertEqual([item["role"] for item in request["messages"]], ["system", "user"])
        text = "\n".join(item["content"] for item in request["messages"])
        self.assertIn("project-review", text)
        self.assertIn("compare-materials", text)
        self.assertNotIn("manage-tasks", text)
        for index, project in enumerate(projects, 1):
            self.assertIn(f"project-{index}", text)
            self.assertIn(project.name, text)
        self.assertNotIn(self.root.name, text, "Planner must not receive absolute source paths")
        self.assertNotIn("PRIVATE_SOURCE_", text)
        self.assertNotIn("PRIVATE_DEFAULT_SOURCE", text)

    def assert_fresh_execution(self, request, planning_response, *, review=True):
        self.assertEqual([item["role"] for item in request["messages"]], ["system", "user"])
        expected_tools = {"list_files", "read_file", "save_report"}
        if review:
            expected_tools.add("search_text")
        self.assertEqual({tool["function"]["name"] for tool in request["tools"]}, expected_tools)
        self.assertNotIn(planning_response["content"],
                         "\n".join(item["content"] for item in request["messages"]))

    def test_model_plan_reviews_two_roots_then_compares_exact_dependency_snapshots(self):
        plan = {"steps": [source_step(), source_step("project-2"), {
            "skill": "compare-materials", "task": "Сравни оба переданных обзора.",
            "project": None, "depends_on": [1, 2],
        }]}
        planned = message(plan)
        responses = [planned, *review("ALPHA_REPORT: возможности по README."),
                     *review("BETA_REPORT: возможности по README."),
                     call("list", "list_files"),
                     call("first", "read_file", path="step-1.md"),
                     call("second", "read_file", path="step-2.md"),
                     call("save", "save_report", content="Сопоставлены два частичных обзора."),
                     {"role": "assistant", "content": "Сопоставление сохранено."}]
        originals = {project: (project / "README.md").read_bytes() for project in self.projects}
        code, output, errors, requests = self.invoke(self.arguments(self.projects), responses)
        self.assertEqual(code, 0, output + errors)
        self.assertEqual(len(requests), 14)
        self.assertIn("Запросов к модели на планирование: 1.", output)
        self.assertIn("Всего запросов к модели: 14; вызовов инструментов: 10.", output)
        self.assert_planning_request(requests[0], self.projects)
        for first_request in (1, 5):
            self.assert_fresh_execution(requests[first_request], planned)
        self.assert_fresh_execution(requests[9], planned, review=False)
        run_dir = self.run_dir()
        saved_plan = json.loads((run_dir / "plan.json").read_text(encoding="utf-8"))
        self.assertEqual([step["id"] for step in saved_plan["steps"]], ["step-1", "step-2", "step-3"])
        self.assertEqual([step["project"] for step in saved_plan["steps"]], ["project-1", "project-2", None])
        self.assertEqual(saved_plan["steps"][2]["depends_on"], ["step-1", "step-2"])
        self.assertEqual(saved_plan["steps"][2]["skill"], "compare-materials")
        self.assertNotIn(self.root.name, json.dumps(saved_plan))
        self.assertIn("alpha", json.dumps(saved_plan))
        self.assertIn("beta", json.dumps(saved_plan))
        for index, project in enumerate(self.projects, 1):
            report = run_dir / f"step-{index}" / "report.md"
            copied = run_dir / "step-3" / "inputs" / f"step-{index}.md"
            self.assertEqual(copied.read_bytes(), report.read_bytes())
            self.assertIn(str(report), output)
            self.assertEqual((project / "README.md").read_bytes(), originals[project])
        self.assertEqual(sorted(item.name for item in (run_dir / "step-3" / "inputs").iterdir()),
                         ["step-1.md", "step-2.md"])
        final = run_dir / "step-3" / "report.md"
        self.assertIn("`step-1.md`", final.read_text(encoding="utf-8"))
        self.assertIn("`step-2.md`", final.read_text(encoding="utf-8"))
        self.assertIn(str(final), output)
        log = next(self.root.glob("runs/*.jsonl")).read_text(encoding="utf-8")
        for value in (self.root.name, "PRIVATE_SOURCE_", "ALPHA_REPORT", "BETA_REPORT", planned["content"]):
            self.assertNotIn(value, log)

    def test_default_root_allows_one_overview_and_counts_planning_separately(self):
        planned = message({"steps": [source_step(task="Опиши учебный проект и сохрани обзор.")]})
        code, output, errors, requests = self.invoke(
            self.arguments(goal="Опиши учебный проект и сохрани обзор."),
            [planned, *review("Обзор проекта по прочитанному README.")],
        )
        self.assertEqual(code, 0, output + errors)
        self.assertEqual(len(requests), 5)
        self.assert_planning_request(requests[0], [self.default_project])
        self.assert_fresh_execution(requests[1], planned)
        self.assertIn("Запросов к модели на планирование: 1.", output)
        self.assertIn("Всего запросов к модели: 5; вызовов инструментов: 3.", output)
        self.assertTrue((self.run_dir() / "step-1" / "report.md").is_file())

    def test_independent_terminal_reports_are_all_displayed_without_claiming_comparison(self):
        planned = message({"steps": [source_step(), source_step("project-2")]})
        code, output, errors, requests = self.invoke(
            self.arguments(self.projects, "Подготовь два отдельных обзора, без сравнения."),
            [planned, *review("Первый отдельный обзор."), *review("Второй отдельный обзор.")],
        )
        self.assertEqual(code, 0, output + errors)
        self.assertEqual(len(requests), 9)
        run_dir = self.run_dir()
        for index in (1, 2):
            report = run_dir / f"step-{index}" / "report.md"
            self.assertTrue(report.is_file())
            self.assertIn(str(report), output)
        self.assertNotIn("Итоговое сравнение:", output)
        self.assertFalse(list(run_dir.glob("*/inputs")))

    def test_planner_clarification_stops_before_any_execution_or_report(self):
        question = "Какие два документа требуется сопоставить?"
        code, output, errors, requests = self.invoke(
            self.arguments([self.projects[0]], "Сопоставь документы."),
            [message({"clarification": question})],
        )
        self.assertEqual(code, 1, output + errors)
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0]["tools"], [])
        self.assertIn(question, output)
        self.assertIn("Запросов к модели на планирование: 1.", output)
        self.assertFalse(list(self.root.glob("workspace/reports/*/*/report.md")))

    def test_invalid_late_alias_exhausts_repair_without_executing_earlier_valid_step(self):
        invalid = {"steps": [source_step(), source_step("project-99")]}
        code, output, errors, requests = self.invoke(
            self.arguments(self.projects), [message(invalid), message(invalid)],
        )
        self.assertEqual(code, 1, output + errors)
        self.assertEqual(len(requests), 2)
        self.assertTrue(all(request["tools"] == [] for request in requests))
        self.assertIn("Запросов к модели на планирование: 2.", output)
        self.assertFalse(list(self.root.glob("workspace/reports/*/*/report.md")))
        self.assertFalse(list(self.root.glob("workspace/reports/*/step-*")))

    def test_invalid_json_can_be_repaired_once_before_execution_with_fresh_history(self):
        planned = message({"steps": [source_step()]})
        code, output, errors, requests = self.invoke(
            self.arguments([self.projects[0]], "Сохрани обзор проекта."),
            [{"role": "assistant", "content": "This is not JSON."}, planned, *review("Готовый обзор.")],
        )
        self.assertEqual(code, 0, output + errors)
        self.assertEqual(len(requests), 6)
        self.assertEqual(requests[0]["tools"], [])
        self.assertEqual(requests[1]["tools"], [])
        self.assert_fresh_execution(requests[2], planned)
        self.assertNotIn("This is not JSON.", json.dumps(requests[2]))
        self.assertIn("Запросов к модели на планирование: 2.", output)
        self.assertIn("Всего запросов к модели: 6; вызовов инструментов: 3.", output)

    def test_invalid_cli_goal_roots_and_mode_combinations_do_not_call_model(self):
        extra = []
        for index in range(3):
            project = self.root / f"extra-{index}"
            project.mkdir()
            extra.append(project)
        cases = [
            self.arguments(self.projects, goal=None),
            self.arguments(self.projects, goal="   "),
            self.arguments([*self.projects, *extra]),
            self.arguments([self.projects[0], self.projects[0]]),
            self.arguments([self.root / "missing-project"]),
            self.arguments([self.projects[0] / "README.md"]),
        ]
        for mode in (["--demo"], ["--check"], ["--describe-project"],
                     ["--skill", "project-review"], ["--list-skills"], ["--compare-projects"]):
            cases.append([*self.arguments([self.projects[0]]), *mode])
        for arguments in cases:
            with self.subTest(arguments=arguments):
                code, output, errors, requests = self.invoke(arguments, [])
                self.assertEqual(code, 2, output + errors)
                self.assertEqual(requests, [])
        self.assertFalse(list(self.root.glob("workspace/reports/*/*/report.md")))


if __name__ == "__main__":
    unittest.main()
