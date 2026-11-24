"""CLI workflows through scripted HTTP, with temporary projects and no model."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from assistant_lab.__main__ import main, parser


class WorkflowCliTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.env = self.root / "test.settings"
        self.env.write_text(
            "ASSISTANT_BASE_URL=http://localhost:8000/v1\n"
            "ASSISTANT_MODEL=workflow-test\nASSISTANT_READ_BUDGET=1\n",
            encoding="utf-8",
        )

    def projects(self, count=2):
        roots = []
        for name in ("alpha", "beta", "gamma", "delta", "epsilon")[:count]:
            project = self.root / "owner-projects" / name
            project.mkdir(parents=True)
            project.joinpath("README.md").write_text(f"PRIVATE-SOURCE-{name}\n", encoding="utf-8")
            roots.append(project)
        return roots

    @staticmethod
    def action(stage, name=None, arguments=None, *, text="Этап завершён.", ok=True):
        return {"stage": stage, "name": name, "arguments": arguments or {}, "text": text, "ok": ok}

    def review_script(self, project, index):
        stage = f"project-{index}"
        return [
            self.action(stage, "list_files", {"path": ".", "limit": 100}),
            self.action(stage, "read_file", {"path": "README.md"}),
            self.action(stage, "save_report", {"content": f"# Обзор {project.name}\nИзучен README; запуск не проверялся."}),
            self.action(stage),
        ]

    def comparison_script(self, count):
        stage = "comparison"
        return [
            self.action(stage, "list_files", {"path": "."}),
            *(self.action(stage, "read_file", {"path": f"project-{index}.md"})
              for index in range(1, count + 1)),
            self.action(stage, "save_report", {"content":
                "# Сопоставление обзоров\n| Критерий | Проекты |\n| --- | --- |\n"
                "| Покрытие | Частичные обзоры README |\n"}),
            self.action(stage, text="Сопоставление сохранено."),
        ]

    def arguments(self, projects, goal="Сравни структуру и ограничения без ранжирования."):
        args = ["--compare-projects", "--env-file", str(self.env), "--read-budget", "64"]
        for project in projects:
            args.extend(["--project", str(project)])
        if goal:
            args.append(goal)
        return args

    def invoke(self, arguments, script):
        requests = []
        stdout, stderr = io.StringIO(), io.StringIO()

        def respond(request, timeout):
            self.assertEqual(request.get_method(), "POST")
            self.assertEqual(request.full_url, "http://localhost:8000/v1/chat/completions")
            index = len(requests)
            self.assertLess(index, len(script), "Unexpected extra model request")
            action = script[index]
            data = json.loads(request.data)
            requests.append(data)
            self.assertEqual(data["model"], "workflow-test")
            self.assertEqual(
                {tool["function"]["name"] for tool in data["tools"]},
                {"list_files", "read_file", "save_report"},
                "Workflow skills must be preloaded without a load_skill request",
            )
            skill = "compare-materials" if action["stage"] == "comparison" else "project-review"
            self.assertIn("выбран скилл " + skill, data["messages"][0]["content"])
            if not index or script[index - 1]["stage"] != action["stage"]:
                self.assertEqual([message["role"] for message in data["messages"]], ["system", "user"])
            else:
                previous = script[index - 1]
                self.assertIsNotNone(previous["name"])
                result_message = data["messages"][-1]
                self.assertEqual(result_message["role"], "tool")
                self.assertEqual(result_message["tool_call_id"], f"request-{index - 1}")
                result = json.loads(result_message["content"])
                self.assertEqual(result["ok"], previous["ok"], result)
                if previous["name"] == "list_files" and previous["stage"] != "comparison":
                    self.assertEqual(result["remaining_read_bytes"], 64)
                if previous["name"] == "read_file" and previous["stage"] != "comparison":
                    self.assertFalse(result["truncated"])
            if action["name"] is None:
                message = {"role": "assistant", "content": action["text"]}
                reason = "stop"
            else:
                message = {"role": "assistant", "content": None, "tool_calls": [{
                    "id": f"request-{index}", "type": "function", "function": {
                        "name": action["name"],
                        "arguments": json.dumps(action["arguments"], ensure_ascii=False),
                    },
                }]}
                reason = "tool_calls"
            return io.BytesIO(json.dumps({"choices": [{"message": message, "finish_reason": reason}]}).encode())

        with patch("assistant_lab.__main__.ROOT", self.root), patch("os.environ", {}), \
                patch("urllib.request.OpenerDirector.open", side_effect=respond) as transport, \
                contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = main(arguments)
        self.assertEqual(transport.call_count, len(script))
        return code, stdout.getvalue(), stderr.getvalue(), requests

    def assert_successful_workflow(self, projects, goal, arguments):
        script = [action for index, project in enumerate(projects, 1)
                  for action in self.review_script(project, index)]
        script.extend(self.comparison_script(len(projects)))
        source_bytes = {project: project.joinpath("README.md").read_bytes() for project in projects}
        code, output, errors, requests = self.invoke(arguments, script)
        self.assertEqual(code, 0, errors + output)
        self.assertEqual(errors, "")
        self.assertEqual(len(requests), 5 * len(projects) + 3)
        self.assertIn(f"Всего запросов к модели: {5 * len(projects) + 3}; "
                      f"вызовов инструментов: {4 * len(projects) + 2}.", output)
        run_dirs = list(self.root.glob("workspace/reports/*"))
        self.assertEqual(len(run_dirs), 1)
        run_dir = run_dirs[0]
        for index, project in enumerate(projects, 1):
            stage_report = run_dir / f"project-{index}" / "report.md"
            copied_report = run_dir / "comparison" / "inputs" / f"project-{index}.md"
            self.assertTrue(stage_report.is_file())
            self.assertEqual(copied_report.read_bytes(), stage_report.read_bytes())
            self.assertIn(str(stage_report), output)
            self.assertIn(f"Этап project-{index}: completed", output)
            self.assertIn(project.name, requests[(index - 1) * 4]["messages"][1]["content"])
            self.assertEqual(project.joinpath("README.md").read_bytes(), source_bytes[project])
            self.assertEqual([item.name for item in project.iterdir()], ["README.md"])
        comparison = run_dir / "comparison" / "report.md"
        self.assertIn("| Критерий |", comparison.read_text(encoding="utf-8"))
        self.assertIn("Итоговое сравнение: " + str(comparison), output)
        comparison_prompt = requests[4 * len(projects)]["messages"][1]["content"]
        self.assertIn("Соответствие файлов и имён папок", comparison_prompt)
        for index, project in enumerate(projects, 1):
            self.assertIn(json.dumps(f"project-{index}.md") + ": " + json.dumps(project.name), comparison_prompt)
            self.assertIn(f"project-{index}.md", comparison.read_text(encoding="utf-8"))
        for request in requests:
            self.assertIn(goal, request["messages"][1]["content"])
        self.assertNotIn("PRIVATE-SOURCE", json.dumps(requests[4 * len(projects)]["messages"]))
        events = [json.loads(line) for line in next(self.root.glob("runs/*.jsonl")).read_text(encoding="utf-8").splitlines()]
        finished = [event for event in events if event["event"] == "workflow_finished"]
        self.assertEqual(len(finished), 1)
        self.assertEqual((finished[0]["status"], finished[0]["steps"], finished[0]["tool_calls"]),
                         ("completed", 5 * len(projects) + 3, 4 * len(projects) + 2))

    def test_two_project_workflow_passes_owner_criteria_and_prints_real_reports(self):
        projects = self.projects()
        goal = "Сравни пригодность для занятий; учитывай структуру и ограничения."
        self.assert_successful_workflow(projects, goal, self.arguments(projects, goal))

    def test_three_projects_and_owner_goal_between_project_options(self):
        projects = self.projects(3)
        goal = "Выдели общие компоненты и неизвестные свойства."
        arguments = ["--compare-projects", "--project", str(projects[0]), goal,
                     "--project", str(projects[1]), "--project", str(projects[2]),
                     "--env-file", str(self.env), "--read-budget", "64"]
        self.assert_successful_workflow(projects, goal, arguments)

    def test_owner_goal_can_precede_or_follow_project_options(self):
        projects = self.projects()
        options = ["--compare-projects", "--project", str(projects[0]), "--project", str(projects[1])]
        for arguments in (["Критерии владельца", *options], [*options, "Критерии владельца"]):
            with self.subTest(arguments=arguments):
                args = parser().parse_args(arguments)
                self.assertEqual(args.task, "Критерии владельца")
                self.assertEqual(args.project, projects)

    def test_repeated_project_without_comparison_is_parser_error(self):
        projects = self.projects()
        with patch("assistant_lab.__main__.ROOT", self.root), patch("os.environ", {}), \
                patch("urllib.request.OpenerDirector.open") as transport, \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as stopped:
            main(["--project", str(projects[0]), "--project", str(projects[1]),
                  "--env-file", str(self.env)])
        self.assertEqual(stopped.exception.code, 2)
        transport.assert_not_called()
        self.assertFalse((self.root / "workspace").exists())

    def test_comparison_is_mutually_exclusive_with_other_modes(self):
        for mode in (["--demo"], ["--check"], ["--describe-project"],
                     ["--skill", "project-review"], ["--list-skills"]):
            with self.subTest(mode=mode), patch("urllib.request.OpenerDirector.open") as transport, \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as stopped:
                main(["--compare-projects", *mode])
            self.assertEqual(stopped.exception.code, 2)
            transport.assert_not_called()

    def test_invalid_root_counts_duplicates_missing_and_file_roots_fail_without_model(self):
        projects = self.projects(5)
        cases = [[], [projects[0]], projects,
                 [projects[0], projects[0]],
                 [projects[0], projects[0] / ".." / projects[0].name],
                 [projects[0], self.root / "missing-project"],
                 [projects[0], projects[1] / "README.md"]]
        for roots in cases:
            with self.subTest(roots=roots):
                code, output, errors, requests = self.invoke(self.arguments(roots), [])
                self.assertEqual(code, 2, output + errors)
                self.assertIn("Ошибка:", errors)
                self.assertEqual(requests, [])
                self.assertNotIn("Итоговое сравнение:", output)
        self.assertFalse(list(self.root.glob("workspace/reports/*/*/report.md")))

    def test_failed_review_stops_before_comparison_despite_model_success_claim(self):
        projects = self.projects()
        script = self.review_script(projects[0], 1)
        script.append(self.action("project-2", text="Всё сохранено, сравнение готово!"))
        code, output, errors, requests = self.invoke(self.arguments(projects), script)
        self.assertEqual(code, 1, errors + output)
        self.assertIn("Цепочка остановлена на этапе project-2", output)
        self.assertIn("Этап project-2: failed", output)
        self.assertNotIn("Итоговое сравнение:", output)
        self.assertEqual(len(requests), 5)
        run_dir = next(self.root.glob("workspace/reports/*"))
        self.assertTrue((run_dir / "project-1" / "report.md").is_file())
        self.assertFalse((run_dir / "project-2" / "report.md").exists())
        self.assertFalse((run_dir / "comparison").exists())

    def test_comparison_cannot_save_after_reading_only_one_dependency(self):
        projects = self.projects()
        script = [action for index, project in enumerate(projects, 1)
                  for action in self.review_script(project, index)]
        script.extend([
            self.action("comparison", "list_files", {}),
            self.action("comparison", "read_file", {"path": "project-1.md"}),
            self.action("comparison", "save_report", {"content": "Invented final comparison"}, ok=False),
            self.action("comparison", text="Сравнение сохранено!"),
        ])
        code, output, errors, _ = self.invoke(self.arguments(projects), script)
        self.assertEqual(code, 1, errors + output)
        self.assertIn("Этап comparison: failed", output)
        self.assertNotIn("Итоговое сравнение:", output)
        run_dir = next(self.root.glob("workspace/reports/*"))
        self.assertTrue((run_dir / "project-1" / "report.md").is_file())
        self.assertTrue((run_dir / "project-2" / "report.md").is_file())
        self.assertFalse((run_dir / "comparison" / "report.md").exists())


if __name__ == "__main__":
    unittest.main()
