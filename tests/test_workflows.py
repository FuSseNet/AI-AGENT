"""Проверки многошаговых workflow без сети и настоящей языковой модели.

Сценарная модель лишь воспроизводит заранее заданные решения. Проверяется
изоляция стадий и передача артефактов, а не качество понимания нейросети.
"""

from copy import deepcopy
import json
from pathlib import Path
import stat
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from assistant_lab.events import EventLog
from assistant_lab.providers import ModelError
from assistant_lab.skills import SkillRegistry
from assistant_lab.workflows import WorkflowError, WorkflowRunner, WorkflowStep


def tool_calls(*calls):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
            }
            for call_id, name, arguments in calls
        ],
    }


def stage_script(report, sources=("README.md",)):
    # Одинаковые идентификаторы намеренно повторяются в независимых стадиях.
    return [
        tool_calls(("list", "list_files", {})),
        tool_calls(*[(f"read-{number}", "read_file", {"path": source})
                     for number, source in enumerate(sources)]),
        tool_calls(("save", "save_report", {"content": report})),
        {"role": "assistant", "content": "Отчёт сохранён."},
    ]


class ScriptedModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.observed = []

    def complete(self, messages, tools):
        self.observed.append({"messages": deepcopy(messages), "tools": deepcopy(tools)})
        if not self.responses:
            raise AssertionError("Неожиданный дополнительный запрос модели.")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return deepcopy(response)


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.library = self.root / "library"
        self.library.mkdir()
        self.make_skill("project-review", "Описать проект.", "REVIEW_SKILL_BODY_MARKER")
        self.make_skill("compare-documents", "Сравнить документы.", "COMPARE_SKILL_BODY_MARKER")
        self.registry = SkillRegistry(self.library)
        self.output = self.root / "output"
        self.output.mkdir()
        self.log = EventLog(self.root / "workflow.jsonl")
        self.project_a = self.root / "project-a"
        self.project_b = self.root / "project-b"
        self.project_a.mkdir()
        self.project_b.mkdir()
        (self.project_a / "README.md").write_text("A_SOURCE_PAYLOAD_MARKER\n", encoding="utf-8")
        (self.project_b / "README.md").write_text("B_SOURCE_PAYLOAD_MARKER\n", encoding="utf-8")

    def make_skill(self, name, description, instructions):
        directory = self.library / name
        directory.mkdir()
        (directory / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: {description}\n---\n{instructions}\n",
            encoding="utf-8",
        )

    def plan(self):
        return [
            WorkflowStep("review-a", "project-review", "TASK_A_MARKER: опиши проект A.", self.project_a),
            WorkflowStep("review-b", "project-review", "TASK_B_MARKER: опиши проект B.", self.project_b),
            WorkflowStep("comparison", "compare-documents", "TASK_COMPARE_MARKER: сравни обзоры.",
                         depends_on=("review-a", "review-b")),
        ]

    def runner(self, model, **kwargs):
        return WorkflowRunner(model, self.registry, self.log, self.output, **kwargs)

    @staticmethod
    def tool_results(snapshot):
        return {
            message["tool_call_id"]: json.loads(message["content"])
            for message in snapshot["messages"] if message["role"] == "tool"
        }

    def test_two_reviews_then_comparison_have_fresh_contexts_and_exact_artifacts(self):
        model = ScriptedModel(
            stage_script("A_REPORT_PAYLOAD_MARKER: обзор A.")
            + stage_script("B_REPORT_PAYLOAD_MARKER: обзор B.")
            + stage_script("Итог сравнения.", ("review-a.md", "review-b.md"))
        )
        result = self.runner(model).run(self.plan())
        self.assertEqual(result.status, "completed")
        self.assertTrue(result.final_text.strip())
        self.assertIsInstance(result.stages, tuple)
        self.assertEqual([stage.step_id for stage in result.stages], ["review-a", "review-b", "comparison"])
        self.assertTrue(all(stage.result.status == "completed" for stage in result.stages))
        self.assertEqual(result.steps, sum(stage.result.steps for stage in result.stages))
        self.assertEqual(result.tool_calls, sum(stage.result.tool_calls for stage in result.stages))
        self.assertEqual(len(model.observed), 12)
        self.assertEqual(result.stages[0].read_paths, ("README.md",))
        self.assertEqual(result.stages[2].read_paths, ("review-a.md", "review-b.md"))
        for stage in result.stages:
            self.assertEqual(stage.report_path, self.output / stage.step_id / "report.md")
            self.assertTrue(stage.report_path.is_file())
        for call_index, marker in ((0, "REVIEW_SKILL_BODY_MARKER"), (4, "REVIEW_SKILL_BODY_MARKER"),
                                   (8, "COMPARE_SKILL_BODY_MARKER")):
            snapshot = model.observed[call_index]
            self.assertEqual(len(snapshot["messages"]), 2)
            self.assertEqual([message["role"] for message in snapshot["messages"]], ["system", "user"])
            self.assertIn(marker, snapshot["messages"][0]["content"])
            self.assertEqual({tool["function"]["name"] for tool in snapshot["tools"]},
                             {"list_files", "read_file", "save_report"})
            first_request = json.dumps(snapshot, ensure_ascii=False)
            self.assertNotIn("A_SOURCE_PAYLOAD_MARKER", first_request)
            self.assertNotIn("B_SOURCE_PAYLOAD_MARKER", first_request)
            self.assertNotIn("A_REPORT_PAYLOAD_MARKER", first_request)
        inputs = self.output / "comparison" / "inputs"
        self.assertEqual(sorted(path.name for path in inputs.iterdir()), ["review-a.md", "review-b.md"])
        for dependency in ("review-a", "review-b"):
            self.assertEqual((inputs / f"{dependency}.md").read_bytes(),
                             (self.output / dependency / "report.md").read_bytes())
        self.assertEqual((self.project_a / "README.md").read_text(encoding="utf-8"), "A_SOURCE_PAYLOAD_MARKER\n")
        self.assertEqual((self.project_b / "README.md").read_text(encoding="utf-8"), "B_SOURCE_PAYLOAD_MARKER\n")

    def test_dependency_stage_cannot_read_originals_or_undeclared_reports(self):
        steps = self.plan()[:2] + [
            WorkflowStep("summary", "project-review", "Сделай итог по одному указанному обзору.",
                         depends_on=("review-a",)),
        ]
        isolated = [
            tool_calls(("list", "list_files", {})),
            tool_calls(("allowed", "read_file", {"path": "review-a.md"}),
                       ("original", "read_file", {"path": "README.md"}),
                       ("undeclared", "read_file", {"path": "review-b.md"}),
                       ("escape", "read_file", {"path": "../review-b/report.md"})),
            tool_calls(("save", "save_report", {"content": "Итог первого обзора."})),
            {"role": "assistant", "content": "Готово."},
        ]
        model = ScriptedModel(stage_script("Обзор A.") + stage_script("Обзор B.") + isolated)
        result = self.runner(model).run(steps)
        self.assertEqual(result.status, "completed")
        self.assertEqual(len(model.observed[8]["messages"]), 2)
        listing = self.tool_results(model.observed[9])["list"]
        self.assertEqual([item["path"] for item in listing["files"]], ["review-a.md"])
        responses = self.tool_results(model.observed[10])
        self.assertIs(responses["allowed"]["ok"], True)
        for call_id in ("original", "undeclared", "escape"):
            self.assertIs(responses[call_id]["ok"], False)
        self.assertEqual(result.stages[-1].read_paths, ("review-a.md",))
        self.assertEqual([path.name for path in (self.output / "summary" / "inputs").iterdir()], ["review-a.md"])

    def test_model_failure_halts_before_any_later_stage(self):
        model = ScriptedModel([ModelError("Тестовая недоступность модели.")] + stage_script("Не должно выполняться."))
        result = self.runner(model).run(self.plan()[:2])
        self.assertEqual(result.status, "failed")
        self.assertEqual(len(result.stages), 1)
        self.assertEqual(result.stages[0].result.status, "failed")
        self.assertEqual(len(model.observed), 1)
        self.assertFalse((self.output / "review-b" / "report.md").exists())

    def test_saved_report_does_not_turn_subsequent_model_failure_into_workflow_success(self):
        first = stage_script("Сохранённый промежуточный отчёт.")[:3]
        model = ScriptedModel(first + [ModelError("Сбой после сохранения.")] + stage_script("Не выполнять."))
        result = self.runner(model).run(self.plan()[:2])
        self.assertEqual(result.status, "failed")
        self.assertEqual(len(result.stages), 1)
        self.assertTrue((self.output / "review-a" / "report.md").is_file())
        self.assertEqual(len(model.observed), 4)
        self.assertFalse((self.output / "review-b" / "report.md").exists())

    def test_claim_without_real_report_is_a_failed_workflow(self):
        model = ScriptedModel([{"role": "assistant", "content": "Я всё сохранил!"}] + stage_script("Не выполнять."))
        result = self.runner(model).run(self.plan()[:2])
        self.assertEqual(result.status, "failed")
        self.assertEqual(len(result.stages), 1)
        self.assertIsNone(result.stages[0].report_path)
        self.assertEqual(len(model.observed), 1)
        self.assertFalse(list(self.output.rglob("report.md")))

    def test_unreadable_saved_report_is_not_passed_to_later_stage(self):
        model = ScriptedModel(stage_script("Invalid report: \x00") + stage_script("Не выполнять."))
        result = self.runner(model).run(self.plan()[:2])
        self.assertEqual(result.status, "failed")
        self.assertEqual(len(result.stages), 1)
        self.assertEqual(len(model.observed), 4)
        self.assertTrue((self.output / "review-a" / "report.md").is_file())
        self.assertFalse((self.output / "review-b").exists())

    def test_step_limit_halts_even_when_the_last_permitted_step_saved_a_report(self):
        model = ScriptedModel(stage_script("Отчёт перед достижением лимита.") + stage_script("Не выполнять."))
        result = self.runner(model, max_steps=3).run(self.plan()[:2])
        self.assertEqual(result.status, "limit_reached")
        self.assertEqual(len(result.stages), 1)
        self.assertEqual(result.stages[0].result.status, "limit_reached")
        self.assertTrue((self.output / "review-a" / "report.md").is_file())
        self.assertEqual(len(model.observed), 3)
        self.assertFalse((self.output / "review-b" / "report.md").exists())

    def test_invalid_plan_is_rejected_before_model_calls_or_stage_artifacts(self):
        source = WorkflowStep("source", "project-review", "Описание.", self.project_a)
        invalid_plans = [
            [],
            [WorkflowStep(f"source-{index}", "project-review", "Описание.", self.project_a) for index in range(9)],
            [source, source],
            [WorkflowStep("../escape", "project-review", "Описание.", self.project_a)],
            [WorkflowStep("con", "project-review", "Описание.", self.project_a)],
            [WorkflowStep("credentials", "project-review", "Описание.", self.project_a)],
            [WorkflowStep("a" * 33, "project-review", "Описание.", self.project_a)],
            [WorkflowStep("source", "missing-skill", "Описание.", self.project_a)],
            [WorkflowStep("source", "project-review", " \t", self.project_a)],
            [WorkflowStep("source", "project-review", "Описание.", self.root / "missing")],
            [WorkflowStep("source", "project-review", "Описание.", self.project_a / "README.md")],
            [WorkflowStep("source", "project-review", "Описание.")],
            [source, WorkflowStep("mixed", "project-review", "Описание.", self.project_b, ("source",))],
            [source, WorkflowStep("future", "project-review", "Описание.", depends_on=("later",))],
            [source, WorkflowStep("duplicate", "project-review", "Описание.", depends_on=("source", "source"))],
            [source, WorkflowStep("comparison", "compare-documents", "Сравни.", depends_on=("source",))],
            [WorkflowStep("source", "project-review", "Описание.", depends_on=("source",))],
        ]
        for plan in invalid_plans:
            model = ScriptedModel([])
            with self.subTest(plan=plan), self.assertRaises(WorkflowError):
                self.runner(model).run(plan)
            self.assertEqual(model.observed, [])
            self.assertEqual(list(self.output.iterdir()), [])

    def test_reparse_source_directory_is_rejected_before_model_calls(self):
        model = ScriptedModel([])
        original_lstat = Path.lstat

        def marked_lstat(path):
            if path == self.project_a:
                return SimpleNamespace(st_mode=stat.S_IFDIR, st_file_attributes=0x400)
            return original_lstat(path)

        with patch.object(Path, "lstat", autospec=True, side_effect=marked_lstat):
            with self.assertRaises(WorkflowError):
                self.runner(model).run(self.plan()[:1])
        self.assertEqual(model.observed, [])

    def test_comparison_budget_covers_all_generated_reports_above_sixteen_kib(self):
        first_report = "A" * 10000 + "\n"
        second_report = "B" * 10000 + "\n"
        model = ScriptedModel(
            stage_script(first_report) + stage_script(second_report)
            + stage_script("Сравнение больших промежуточных отчётов.", ("review-a.md", "review-b.md"))
        )
        result = self.runner(model, read_budget=64).run(self.plan())
        self.assertEqual(result.status, "completed")
        total_report_bytes = sum((self.output / step_id / "report.md").stat().st_size
                                 for step_id in ("review-a", "review-b"))
        self.assertGreater(total_report_bytes, 16384)
        listing = self.tool_results(model.observed[9])["list"]
        self.assertEqual(listing["remaining_read_bytes"], total_report_bytes)
        reads = self.tool_results(model.observed[10])
        self.assertIs(reads["read-0"]["ok"], True)
        self.assertIs(reads["read-1"]["ok"], True)
        self.assertEqual(reads["read-1"]["remaining_read_bytes"], 0)
        for step_id, call_id in (("review-a", "read-0"), ("review-b", "read-1")):
            saved = (self.output / step_id / "report.md").read_text(encoding="utf-8")
            self.assertEqual(reads[call_id]["content"], saved)

    def test_workflow_events_identify_stages_without_task_source_or_report_payloads(self):
        model = ScriptedModel(stage_script("A_REPORT_PAYLOAD_MARKER")
                              + stage_script("B_REPORT_PAYLOAD_MARKER")
                              + stage_script("COMPARE_REPORT_PAYLOAD_MARKER", ("review-a.md", "review-b.md")))
        result = self.runner(model).run(self.plan())
        self.assertEqual(result.status, "completed")
        text = self.log.path.read_text(encoding="utf-8")
        records = [json.loads(line) for line in text.splitlines()]
        workflow_events = [record for record in records if record["event"] in {
            "workflow_started", "stage_started", "stage_finished", "workflow_finished",
        }]
        self.assertEqual([record["event"] for record in workflow_events], [
            "workflow_started", "stage_started", "stage_finished", "stage_started",
            "stage_finished", "stage_started", "stage_finished", "workflow_finished",
        ])
        self.assertEqual([record["stage"] for record in workflow_events if record["event"] == "stage_started"],
                         ["review-a", "review-b", "comparison"])
        self.assertEqual([record["skill"] for record in workflow_events if record["event"] == "stage_started"],
                         ["project-review", "project-review", "compare-documents"])
        self.assertEqual(workflow_events[-1]["status"], "completed")
        for marker in ("TASK_A_MARKER", "TASK_B_MARKER", "TASK_COMPARE_MARKER", "A_SOURCE_PAYLOAD_MARKER",
                       "B_SOURCE_PAYLOAD_MARKER", "A_REPORT_PAYLOAD_MARKER", "B_REPORT_PAYLOAD_MARKER",
                       "COMPARE_REPORT_PAYLOAD_MARKER", "REVIEW_SKILL_BODY_MARKER", "COMPARE_SKILL_BODY_MARKER"):
            with self.subTest(marker=marker):
                self.assertNotIn(marker, text)


if __name__ == "__main__":
    unittest.main()
