from contextlib import ExitStack, redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from assistant_lab.__main__ import main
from assistant_lab.config import ConfigError
from assistant_lab.idle import IdleDemoModel
from assistant_lab.providers import DemoModel
from assistant_lab.task_queue import DEMO_TASK, PAGE_SIZE, TaskQueue
from assistant_lab.task_tools import TaskTools


class TaskQueueTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.project = self.root / "workspace" / "project"
        self.project.mkdir(parents=True)
        (self.project / "README.md").write_text("Fixture for a queued assignment.", encoding="utf-8")
        self.stack.enter_context(patch("assistant_lab.__main__.ROOT", self.root))
        self.stack.enter_context(redirect_stdout(io.StringIO()))
        self.queue = TaskQueue(self.root)

    def cli(self, *args, setup="", code=0):
        script = ("import os, sys\nfrom pathlib import Path\nimport assistant_lab.__main__ as cli\n"
                  "cli.ROOT = Path(sys.argv[1])\n" + setup + "\nraise SystemExit(cli.main(sys.argv[2:]))")
        env = {k: v for k, v in os.environ.items() if not k.startswith("ASSISTANT_")}
        result = subprocess.run([sys.executable, "-c", script, str(self.root), *args], capture_output=True,
                                encoding="utf-8", timeout=15, env=env | {"PYTHONIOENCODING": "utf-8"})
        self.assertEqual(result.returncode, code, result.stderr + result.stdout)
        return result.stdout

    def test_separate_cli_processes_execute_queue_before_building_next_view(self):
        self.cli("--add-task", "Сравни документы", "--due", "2026-11-27")
        self.cli("--add-demo-task")
        self.assertIn("#2 pending", self.cli("--list-tasks", "--status", "pending"))
        catalog = self.root / "workspace" / "idle-demo" / "views"
        self.assertIn("Поручение #2: done", self.cli("--demo-idle"))
        self.assertFalse(catalog.exists())
        real, demo = self.queue.list_tasks()["tasks"]
        self.assertEqual(real["status"], "pending")
        self.assertIn("Fixture for a queued assignment.", Path(demo["report"]).read_text(encoding="utf-8"))
        self.assertIn(demo["report"], self.cli("--list-tasks", "--status", "done"))
        self.cli("--demo-idle")
        self.assertTrue((catalog / "upcoming" / "view.json").is_file())

    def test_worker_recovers_published_receipt_after_abrupt_process_exit(self):
        self.cli("--add-demo-task")
        crash = "from assistant_lab.task_queue import TaskQueue\nTaskQueue.finish = lambda *args: os._exit(73)"
        self.cli("--demo-idle", setup=crash, code=73)
        self.assertEqual(self.queue.list_tasks()["tasks"][0]["status"], "running")
        original, = self.root.glob("workspace/reports/*/report.md")
        self.cli("--demo-idle")
        self.assertEqual(self.queue.list_tasks()["tasks"][0]["report"], str(original))
        self.assertEqual(list(self.root.glob("workspace/reports/*/report.md")), [original])

    def test_interrupted_draft_is_retried_in_a_new_run(self):
        self.cli("--add-demo-task")
        crash = ("from assistant_lab.providers import DemoModel\noriginal = DemoModel.complete\n"
                 "def complete(self, messages, tools):\n"
                 " if messages[-1].get('tool_call_id') == 'demo_save': os._exit(74)\n"
                 " return original(self, messages, tools)\nDemoModel.complete = complete")
        self.cli("--demo-idle", setup=crash, code=74)
        draft, = self.root.glob("workspace/reports/*/.draft.md")
        self.assertFalse(draft.with_name("report.md").exists())
        self.cli("--demo-idle")
        report = Path(self.queue.list_tasks()["tasks"][0]["report"])
        self.assertTrue(report.is_file())
        self.assertNotEqual(report.parent, draft.parent)

    def test_failed_job_and_retry_do_not_block_other_tasks(self):
        self.queue.add(DEMO_TASK, self.root / "missing", demo=True)
        self.queue.add(DEMO_TASK, self.project, demo=True)
        self.assertEqual(main(["--demo-idle"]), 1)
        failed, = self.queue.list_tasks("failed")["tasks"]
        self.assertIsNone(failed["report"])
        self.assertTrue(Path(failed["log"]).is_file())
        self.assertEqual(main(["--demo-idle"]), 0)
        self.assertEqual(main(["--retry-task", str(failed["id"])]), 0)
        self.assertEqual(self.queue.list_tasks("pending")["tasks"][0]["id"], failed["id"])
        self.assertEqual(main(["--complete-task", str(failed["id"])]), 0)
        self.assertIsNone(self.queue.list_tasks("done")["tasks"][0]["report"])

    def test_arrival_before_idle_publication_serves_owner_first(self):
        original = IdleDemoModel.complete

        def interrupt(model, messages, tools):
            answer = original(model, messages, tools)
            if messages[-1].get("tool_call_id") == "propose":
                self.queue.add(DEMO_TASK, self.project, demo=True)
            return answer

        with patch.object(IdleDemoModel, "complete", interrupt):
            self.assertEqual(main(["--demo-idle"]), 0)
        self.assertEqual(self.queue.list_tasks()["tasks"][0]["status"], "done")
        self.assertFalse((self.root / "workspace/idle-demo/views/upcoming").exists())

    def test_deadline_order_pages_and_manual_state_changes(self):
        late = self.queue.add("later", self.project, "2026-12-02")
        soon = self.queue.add("soon", self.project, "2026-12-01")
        same = self.queue.add("same day", self.project, "2026-12-01")
        for i in range(PAGE_SIZE):
            self.queue.add(f"undated {i}", self.project)
        page = self.queue.list_tasks()
        self.assertEqual([t["id"] for t in page["tasks"][:3]], [soon, same, late])
        self.assertEqual(len(self.queue.list_tasks(offset=page["next_offset"])["tasks"]), 3)
        self.assertEqual([t["id"] for t in self.queue.list_tasks(due="2026-12-01")["tasks"]], [soon, same])
        self.assertEqual(self.queue.claim_next(False)["id"], soon)
        with self.assertRaises(ConfigError):
            self.queue.change(soon)
        self.queue.change(same)
        self.queue.change(same)
        self.assertEqual(len(self.queue.list_tasks("active", due="2026-12-01")["tasks"]), 1)
        for bad in ("2026-02-30", "20261127", False):
            with self.assertRaises(ConfigError):
                self.queue.add("invalid", self.project, bad)

    def test_management_tool_validates_arguments_and_deduplicates_repeated_model_calls(self):
        tools = TaskTools(self.queue, self.project)
        for args in ({"task": ""}, {"task": "x", "project": "/other"}, {"task": "a", "due": []}):
            self.assertFalse(tools.execute("create_task", args)["ok"])
        first = tools.execute("create_task", {"task": "Реферат"})
        self.assertEqual(first, tools.execute("create_task", {"task": "Реферат"}))
        self.assertEqual(len(self.queue.list_tasks()["tasks"]), 1)
        self.assertFalse(tools.execute("complete_task", {"id": True})["ok"])
        self.assertFalse(tools.last_call_ok)

    def test_owner_can_look_up_due_tasks_and_close_one_by_id(self):
        saved = self.queue.add("Подготовить обзор JUnit", self.project, "2026-11-27")
        self.queue.add("Сравнить версии", self.project, "2026-11-28")
        tools = TaskTools(TaskQueue(self.root), self.project)
        result = tools.execute("list_tasks", {"due": "2026-11-27"})
        self.assertEqual([task["id"] for task in result["tasks"]], [saved])
        self.assertTrue(tools.execute("complete_task", {"id": saved})["ok"])
        self.assertEqual(tools.execute("list_tasks", {"due": "2026-11-27"})["tasks"], [])
        done, = tools.execute("list_tasks", {"status": "done"})["tasks"]
        self.assertIsNone(done["report"])

    def test_error_before_model_call_marks_failed_and_allows_next_assignment(self):
        self.queue.add(DEMO_TASK, self.project, demo=True)
        with patch("assistant_lab.idle.run_assignment", side_effect=PermissionError):
            self.assertEqual(main(["--demo-idle"]), 1)
        entry, = self.queue.list_tasks("failed")["tasks"]
        self.assertIsNone(entry["report"])
        self.assertIsNone(entry["log"])
        self.cli("--retry-task", str(entry["id"]))
        self.cli("--demo-idle")
        self.assertEqual(self.queue.list_tasks("done")["tasks"][0]["id"], entry["id"])

    def test_model_routes_to_queue_and_worker_uses_the_saved_project(self):
        env = self.root / ".env"
        env.write_text("ASSISTANT_BASE_URL=http://localhost:8000/v1\nASSISTANT_MODEL=test\n", encoding="utf-8")
        task = "Опиши назначение проекта по README и сохрани отчёт"
        call = DemoModel._call
        final = {"role": "assistant", "content": "Готово."}
        replies = [call("skill", "load_skill", name="manage-tasks"),
                   call("add", "create_task", task=task, due="2026-11-27"), final,
                   call("skill", "load_skill", name="project-review"), call("list", "list_files"),
                   call("read", "read_file", path="README.md"), call("save", "save_report", content="Обзор"), final]
        requests = []

        def respond(request, timeout):
            requests.append(json.loads(request.data))
            if requests[-1]["messages"][-1]["role"] == "tool":
                self.assertTrue(json.loads(requests[-1]["messages"][-1]["content"])["ok"])
            return io.BytesIO(json.dumps({"choices": [{"message": replies.pop(0)}]}).encode())

        with patch("urllib.request.OpenerDirector.open", side_effect=respond):
            self.assertEqual(main(["Запомни поручение до пятницы", "--project", str(self.project),
                                   "--env-file", str(env)]), 0)
            self.assertEqual(main(["--idle", "--env-file", str(env)]), 0)
        self.assertEqual(replies, [])
        self.assertEqual({tool["function"]["name"] for tool in requests[1]["tools"]},
                         {"create_task", "list_tasks", "complete_task"})
        self.assertEqual(requests[3]["messages"][1]["content"], task)
        self.assertNotIn("manage-tasks", json.dumps(requests[3]["tools"]))
        record, = self.queue.list_tasks("done")["tasks"]
        self.assertEqual(record["due"], "2026-11-27")
        self.assertIn("`README.md`", Path(record["report"]).read_text(encoding="utf-8"))
        self.assertNotIn(task, Path(record["log"]).read_text(encoding="utf-8"))
