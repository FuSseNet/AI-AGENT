from contextlib import ExitStack, redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from assistant_lab.__main__ import main
from assistant_lab.activity import activity
from assistant_lab.idle import IdleDemoModel, ViewTools
from assistant_lab.providers import DemoModel
from study_planner import ALL_TASKS, SORT_FIELDS, STATUSES, apply_view, load_view
from study_planner.__main__ import main as planner
from study_planner.roadmap import BACKLOG, check_view


class IdleTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.stack.enter_context(patch("assistant_lab.__main__.ROOT", self.root))
        self.stack.enter_context(patch.dict("os.environ", {}, clear=True))
        self.output = self.stack.enter_context(redirect_stdout(io.StringIO()))
        self.errors = self.stack.enter_context(redirect_stderr(io.StringIO()))
        self.catalog = self.root / "workspace" / "idle-demo" / "views"

    def test_demo_persists_one_feature_and_planner_uses_it_on_users_data(self):
        self.assertEqual(main(["--demo-idle"]), 0, self.errors.getvalue())
        report = self.catalog / "upcoming" / "report.md"
        first_report = report.read_bytes()
        self.assertIn("Обращений к модели: 3", first_report.decode())
        self.assertEqual(main(["--demo-idle"]), 0)
        self.assertEqual(report.read_bytes(), first_report)
        self.assertTrue((self.catalog / "completed" / "view.json").exists())
        tasks = self.root / "tasks.json"
        tasks.write_text(json.dumps([
            {"title": "Third", "subject": "Art", "due": "2026-11-30", "status": "open"},
            {"title": "Hidden", "subject": "Art", "due": "2026-11-01", "status": "done"},
            {"title": "First", "subject": "Math", "due": "2026-11-09", "status": "open"},
        ]), encoding="utf-8")
        with redirect_stdout(io.StringIO()) as shown:
            self.assertEqual(planner(["--tasks", str(tasks), "--catalog", str(self.catalog),
                                      "--view", "upcoming"]), 0)
        self.assertEqual(shown.getvalue().splitlines(), [
            "2026-11-09  Math  First  [open]", "2026-11-30  Art  Third  [open]"])

    def test_watch_finishes_then_empty_run_never_requests_model(self):
        with patch("assistant_lab.idle.time.sleep") as sleep:
            self.assertEqual(main(["--demo-idle", "--idle-interval", "2"]), 0)
        self.assertEqual(sleep.call_count, 2)
        self.assertEqual({p.name for p in self.catalog.iterdir()}, set(BACKLOG))
        with patch.object(IdleDemoModel, "complete", side_effect=AssertionError) as model:
            self.assertEqual(main(["--demo-idle"]), 0)
            model.assert_not_called()

    def test_each_spec_accepts_only_the_working_rule(self):
        for name in BACKLOG:
            rules = [dict(status=status, sort_by=field, descending=desc)
                     for status in STATUSES for field in SORT_FIELDS for desc in (True, False)]
            self.assertEqual(sum(check_view(name, rule) for rule in rules), 1, name)

    def test_invalid_proposal_can_be_corrected_and_extra_changes_are_rejected(self):
        tools = ViewTools(["upcoming"])
        for invalid in ({"name": "../external", **ALL_TASKS}, {"name": "upcoming", **ALL_TASKS},
                        {"name": "upcoming"}, {"name": "upcoming", **ALL_TASKS, "path": "/tmp"}):
            self.assertFalse(tools.execute("propose_view", invalid)["ok"])
        self.assertIsNone(tools.proposal)
        good = {"name": "upcoming", "status": "open", "sort_by": "due", "descending": False}
        self.assertTrue(tools.execute("propose_view", good)["ok"])
        self.assertFalse(tools.execute("propose_view", good)["ok"])

    def test_owner_session_in_another_process_defers_idle_worker(self):
        code = ("from assistant_lab.activity import activity\n"
                "from pathlib import Path\nimport sys\n"
                "with activity(Path(sys.argv[1]), exclusive=False):\n"
                " print('ready', flush=True)\n sys.stdin.readline()")
        with subprocess.Popen([sys.executable, "-c", code, str(self.root / "runs")],
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True) as owner:
            try:
                self.assertEqual(owner.stdout.readline().strip(), "ready")
                self.assertEqual(main(["--demo-idle"]), 0)
                self.assertFalse(self.catalog.exists())
            finally:
                owner.communicate(input="stop\n", timeout=10)
        self.assertEqual(main(["--demo-idle"]), 0)
        self.assertTrue((self.catalog / "upcoming").is_dir())

    def test_owner_arriving_during_final_model_response_prevents_commit(self):
        original = IdleDemoModel.complete
        with ExitStack() as owner:
            def respond(model, messages, definitions):
                answer = original(model, messages, definitions)
                if messages[-1].get("tool_call_id") == "propose":
                    owner.enter_context(activity(self.root / "runs", exclusive=False))
                return answer

            with patch.object(IdleDemoModel, "complete", respond):
                self.assertEqual(main(["--demo-idle"]), 0)
        self.assertEqual(list(self.catalog.iterdir()), [])
        self.assertEqual(main(["--demo-idle"]), 0)
        self.assertTrue((self.catalog / "upcoming" / "report.md").is_file())

    def test_ordinary_owner_cli_takes_priority_and_recovers_after_exit(self):
        project = self.root / "workspace" / "project"
        project.mkdir(parents=True)
        (project / "README.md").write_text("Учебный проект.", encoding="utf-8")
        original = DemoModel.complete

        def respond(model, messages, definitions):
            if messages[-1]["role"] == "user":
                self.assertEqual(main(["--demo-idle"]), 0)
                self.assertFalse(self.catalog.exists())
            return original(model, messages, definitions)

        with patch.object(DemoModel, "complete", respond):
            self.assertEqual(main(["--demo"]), 0, self.errors.getvalue())
        self.assertIn("Автономная работа отложена", self.output.getvalue())
        self.assertEqual(main(["--demo-idle"]), 0)
        self.assertTrue((self.catalog / "upcoming").is_dir())

    def test_second_idle_worker_defers_until_existing_worker_releases_slot(self):
        with activity(self.root / "runs" / "idle-worker", exclusive=True):
            with patch.object(IdleDemoModel, "complete") as model:
                self.assertEqual(main(["--demo-idle"]), 0)
                model.assert_not_called()
        self.assertFalse(self.catalog.exists())
        self.assertEqual(main(["--demo-idle"]), 0)

    def test_failed_cycle_writes_no_feature_and_existing_invalid_bundle_is_preserved(self):
        self.assertEqual(main(["--demo-idle", "--max-steps", "2"]), 1)
        self.assertEqual(list(self.catalog.iterdir()), [])
        broken = self.catalog / "upcoming"
        broken.mkdir()
        (broken / "view.json").write_text("bad", encoding="utf-8")
        self.assertEqual(main(["--demo-idle"]), 2)
        self.assertEqual((broken / "view.json").read_text(), "bad")
        self.assertFalse((self.catalog / "completed").exists())

    def test_api_path_can_choose_a_nonfirst_task(self):
        env = self.root / ".env"
        env.write_text("ASSISTANT_BASE_URL=http://localhost:8000/v1\nASSISTANT_MODEL=test\n", encoding="utf-8")
        calls = [IdleDemoModel._call("b", "list_work"),
                 IdleDemoModel._call("v", "propose_view", name="by-subject", status="all",
                                     sort_by="subject", descending=False),
                 {"role": "assistant", "content": "Готово."}]

        def respond(request, timeout):
            payload = json.loads(request.data)
            self.assertEqual({t["function"]["name"] for t in payload["tools"]}, {"list_work", "propose_view"})
            if payload["messages"][-1]["role"] == "tool":
                self.assertTrue(json.loads(payload["messages"][-1]["content"])["ok"])
            return io.BytesIO(json.dumps({"choices": [{"message": calls.pop(0)}]}).encode())

        with patch("urllib.request.OpenerDirector.open", side_effect=respond):
            self.assertEqual(main(["--idle", "--env-file", str(env)]), 0, self.errors.getvalue())
        catalog = self.root / "study_planner" / "views"
        self.assertTrue(check_view("by-subject", load_view(catalog, "by-subject")))
        self.assertFalse((catalog / "upcoming").exists())
        self.assertFalse(self.catalog.exists())

    def test_invalid_tasks_and_view_names_report_input_error(self):
        with self.assertRaises(ValueError):
            load_view(self.catalog, "../upcoming")
        with self.assertRaises(ValueError):
            apply_view([{"title": "A", "subject": "Art", "due": "2026-02-30", "status": "open"}], ALL_TASKS)
