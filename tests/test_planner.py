"""Planner contracts with temporary metadata and a scripted model, without execution."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from assistant_lab.events import EventLog
from assistant_lab.planner import ModelPlanner, PlanningError
from assistant_lab.providers import ModelError
from assistant_lab.skills import SkillRegistry


def message(payload):
    return {"role": "assistant", "content": json.dumps(payload, ensure_ascii=False)}


class ScriptedPlannerModel:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def complete(self, messages, tools):
        self.requests.append((json.loads(json.dumps(messages)), json.loads(json.dumps(tools))))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class ModelPlannerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        library = self.root / "controlled-library"
        self.body_markers = ("DO_NOT_SEND_FULL_SKILL_BODY_SOURCE", "DO_NOT_SEND_FULL_SKILL_BODY_COMPARE")
        for name, description, body in (
            ("source-review", "SOURCE_REVIEW_METADATA", self.body_markers[0]),
            ("compare-materials", "COMPARE_MATERIALS_METADATA", self.body_markers[1]),
        ):
            directory = library / name
            directory.mkdir(parents=True)
            directory.joinpath("SKILL.md").write_text(
                f"---\nname: {name}\ndescription: {description}\n---\n{body}\n", encoding="utf-8")
        self.registry = SkillRegistry(library)
        self.projects = []
        self.source_markers = []
        for index in range(2):
            project = self.root / f"PRIVATE_PROJECT_{index + 1}"
            project.mkdir()
            source_marker = f"PRIVATE_SOURCE_CONTENT_{index + 1}"
            project.joinpath("README.md").write_text(source_marker, encoding="utf-8")
            self.projects.append(project)
            self.source_markers.append(source_marker)
        self.log_path = self.root / "planner.jsonl"
        self.log = EventLog(self.log_path)
        self.goal = "OWNER_GOAL_PRIVATE_MARKER: сопоставить возможности и неизвестные свойства."

    def planner(self, model, **kwargs):
        return ModelPlanner(model, self.registry, self.log, **kwargs)

    @staticmethod
    def source_step(project="project-1", task="Изучи выбранный источник."):
        return {"skill": "source-review", "task": task, "project": project, "depends_on": []}

    def three_stage_plan(self):
        return {"steps": [
            self.source_step("project-1", "PROPOSAL_ALPHA: опиши первый проект."),
            self.source_step("project-2", "PROPOSAL_BETA: опиши второй проект."),
            {"skill": "compare-materials", "task": "PROPOSAL_COMPARE: сопоставь обзоры.",
             "project": None, "depends_on": [1, 2]},
        ]}

    @staticmethod
    def prompt_text(model, request_index=None):
        requests = model.requests if request_index is None else [model.requests[request_index]]
        return "\n".join(
            entry["content"] for messages, _ in requests for entry in messages
            if isinstance(entry.get("content"), str))

    def assert_log_private(self, *additional):
        text = self.log_path.read_text(encoding="utf-8") if self.log_path.exists() else ""
        for forbidden in (self.goal, *self.body_markers, *self.source_markers,
                          *(str(project) for project in self.projects), *additional):
            self.assertNotIn(forbidden, text)
        for line in text.splitlines():
            event = json.loads(line)
            self.assertIn("event", event)
        return text

    def test_three_stage_plan_maps_aliases_and_dependencies_without_executing(self):
        payload = self.three_stage_plan()
        model = ScriptedPlannerModel(message(payload))
        with patch("assistant_lab.workflows.WorkflowRunner.run") as execute, \
                patch("assistant_lab.tools.FileTools.execute") as files:
            result = self.planner(model).plan(self.goal, self.projects)
        execute.assert_not_called()
        files.assert_not_called()
        self.assertEqual((result.status, result.requests), ("ready", 1))
        self.assertIsInstance(result.steps, tuple)
        self.assertEqual([step.step_id for step in result.steps], ["step-1", "step-2", "step-3"])
        self.assertEqual([step.skill for step in result.steps], ["source-review", "source-review", "compare-materials"])
        self.assertEqual([step.project_dir for step in result.steps], [*self.projects, None])
        self.assertEqual([step.depends_on for step in result.steps], [(), (), ("step-1", "step-2")])
        for proposed, step in zip(payload["steps"], result.steps):
            self.assertIn(self.goal, step.task)
            self.assertIn(proposed["task"], step.task)
        with self.assertRaises(FrozenInstanceError):
            result.status = "failed"
        self.assert_log_private()

    def test_single_source_plan_is_supported(self):
        model = ScriptedPlannerModel(message({"steps": [self.source_step()]}))
        result = self.planner(model).plan(self.goal, self.projects[:1])
        self.assertEqual((result.status, result.requests, len(result.steps)), ("ready", 1, 1))
        self.assertEqual(result.steps[0].project_dir, self.projects[0])
        self.assertEqual(result.steps[0].depends_on, ())

    def test_empty_optional_tool_calls_are_accepted_for_backend_compatibility(self):
        for empty in (None, []):
            with self.subTest(tool_calls=empty):
                response = message({"steps": [self.source_step()]})
                response["tool_calls"] = empty
                model = ScriptedPlannerModel(response)
                result = self.planner(model).plan(self.goal, self.projects)
                self.assertEqual((result.status, result.requests), ("ready", 1))

    def test_maximum_owner_goal_and_proposal_fit_workflow_task_without_losing_text(self):
        goal_prefix = 'OWNER "quoted" \\path\\\n'
        proposal_prefix = 'PROPOSAL "quoted" \\path\\\n'
        goal = goal_prefix + "g" * (16_384 - len(goal_prefix))
        proposal = proposal_prefix + "p" * (6000 - len(proposal_prefix))
        model = ScriptedPlannerModel(message({"steps": [self.source_step(task=proposal)]}))
        result = self.planner(model).plan(goal, self.projects[:1])
        self.assertEqual((result.status, result.requests), ("ready", 1))
        self.assertLessEqual(len(result.steps[0].task), 32_768)
        self.assertIn(goal, result.steps[0].task)
        self.assertIn(proposal, result.steps[0].task)

    def test_planner_prompts_contain_metadata_but_no_file_payload_paths_or_skill_bodies(self):
        model = ScriptedPlannerModel(message({"steps": [self.source_step()]}))
        self.planner(model).plan(self.goal, self.projects)
        prompt = self.prompt_text(model)
        self.assertIn("SOURCE_REVIEW_METADATA", prompt)
        self.assertIn("COMPARE_MATERIALS_METADATA", prompt)
        self.assertIn("project-1", prompt)
        self.assertIn("project-2", prompt)
        for forbidden in (*self.body_markers, *self.source_markers, *(str(path) for path in self.projects)):
            self.assertNotIn(forbidden, prompt)
            self.assertNotIn(json.dumps(forbidden), prompt)
        self.assertTrue(all(tools == [] for _, tools in model.requests))

    def test_clarification_returns_question_without_steps_or_execution(self):
        question = "Какие два результата требуется сопоставить?"
        model = ScriptedPlannerModel(message({"clarification": question}))
        with patch("assistant_lab.workflows.WorkflowRunner.run") as execute, \
                patch("assistant_lab.tools.FileTools.execute") as files:
            result = self.planner(model).plan(self.goal, self.projects)
        self.assertEqual((result.status, result.steps, result.requests), ("clarification", (), 1))
        self.assertEqual(result.message, question)
        execute.assert_not_called()
        files.assert_not_called()
        self.assert_log_private(question)

    def test_invalid_output_can_be_repaired_once_without_echoing_raw_output(self):
        raw = "PRIVATE_RAW_MODEL_BODY: not a JSON plan"
        model = ScriptedPlannerModel({"role": "assistant", "content": raw},
                                    message(self.three_stage_plan()))
        result = self.planner(model).plan(self.goal, self.projects)
        self.assertEqual((result.status, result.requests), ("ready", 2))
        self.assertEqual(len(model.requests), 2)
        self.assertNotIn(raw, self.prompt_text(model, 1))
        self.assertTrue(all(tools == [] for _, tools in model.requests))
        self.assert_log_private(raw)

    def test_two_invalid_outputs_stop_at_configured_bound(self):
        raw = "PRIVATE_INVALID_MODEL_BODY"
        model = ScriptedPlannerModel({"role": "assistant", "content": raw},
                                    {"role": "assistant", "content": raw})
        result = self.planner(model).plan(self.goal, self.projects)
        self.assertEqual((result.status, result.steps, result.requests), ("failed", (), 2))
        self.assertTrue(result.message)
        self.assertNotIn(raw, result.message)
        self.assert_log_private(raw)

    def test_tool_calls_are_rejected_and_not_replayed_as_orphan_history(self):
        raw_marker = "PRIVATE_TOOL_ARGUMENT_CONTENT"
        invalid = message({"steps": [self.source_step()]})
        invalid["tool_calls"] = [{"id": "bad-call", "type": "function", "function": {
            "name": "read_file", "arguments": json.dumps({"path": raw_marker})}}]
        model = ScriptedPlannerModel(invalid, message({"steps": [self.source_step()]}))
        with patch("assistant_lab.tools.FileTools.execute") as execute:
            result = self.planner(model).plan(self.goal, self.projects)
        self.assertEqual((result.status, result.requests), ("ready", 2))
        execute.assert_not_called()
        for entry in model.requests[1][0]:
            self.assertNotEqual(entry.get("role"), "tool")
            self.assertNotIn("tool_calls", entry)
        self.assertNotIn(raw_marker, self.prompt_text(model, 1))
        self.assert_log_private(raw_marker)

    def test_absolute_paths_alias_escape_and_invalid_dependencies_are_rejected(self):
        first = self.source_step()
        candidates = [
            {"steps": [self.source_step(str(self.projects[0]))]},
            {"steps": [self.source_step("../project-1")]},
            {"steps": [self.source_step("project-1/README.md")]},
            {"steps": [self.source_step("project-3")]},
            {"steps": [{**first, "project": None}]},
            {"steps": [first, {**self.source_step("project-2"), "depends_on": [1]}]},
            {"steps": [{**first, "project": None, "depends_on": [1]}]},
            {"steps": [first, {"skill": "compare-materials", "task": "Compare", "project": None, "depends_on": [2]}]},
            {"steps": [first, {"skill": "compare-materials", "task": "Compare", "project": None, "depends_on": [1]}]},
            {"steps": [first, {"skill": "compare-materials", "task": "Compare", "project": None, "depends_on": [True]}]},
            {"steps": [first, {"skill": "compare-materials", "task": "Compare", "project": None, "depends_on": [1, 1]}]},
            {"steps": [first, {"skill": "compare-materials", "task": "Compare", "project": None, "depends_on": ["step-1"]}]},
        ]
        for index, payload in enumerate(candidates):
            with self.subTest(case=index):
                model = ScriptedPlannerModel(message(payload))
                result = self.planner(model, max_attempts=1).plan(self.goal, self.projects)
                self.assertEqual((result.status, result.steps, result.requests), ("failed", (), 1))
        self.assert_log_private()

    def test_unknown_skills_wrong_keys_and_invalid_task_shapes_are_rejected(self):
        first = self.source_step()
        candidates = [
            {"steps": []},
            {"steps": [first] * 9},
            {"steps": [first], "extra": "ignored?"},
            {"steps": [{**first, "step_id": "chosen-by-model"}]},
            {"steps": [{key: value for key, value in first.items() if key != "depends_on"}]},
            {"steps": [{**first, "skill": "unknown-skill"}]},
            {"steps": [{**first, "task": " "}]},
            {"steps": [{**first, "task": "x" * 6001}]},
            {"steps": [{**first, "task": []}]},
            {"steps": [{**first, "task": "\ud800"}]},
            {"clarification": ""},
            {"clarification": "x" * 2001},
            {"clarification": "\ud800"},
            {"clarification": "question", "steps": [first]},
        ]
        for index, payload in enumerate(candidates):
            with self.subTest(case=index):
                model = ScriptedPlannerModel(message(payload))
                result = self.planner(model, max_attempts=1).plan(self.goal, self.projects)
                self.assertEqual((result.status, result.steps, result.requests), ("failed", (), 1))

    def test_duplicate_json_keys_nan_deep_json_and_non_assistant_messages_fail_safely(self):
        valid = json.dumps({"steps": [self.source_step()]})
        duplicate_stage = ('{"steps":[{"skill":"source-review","skill":"compare-materials",'
                           '"task":"Review","project":"project-1","depends_on":[]}]}')
        responses = [
            {"role": "assistant", "content": '{"steps":[],"steps":[]}'},
            {"role": "assistant", "content": duplicate_stage},
            {"role": "assistant", "content": '{"steps":NaN}'},
            {"role": "assistant", "content": '{"steps":' + "[" * 1200 + "]" * 1200 + "}"},
            {"role": "user", "content": valid},
            {"role": "assistant", "content": {"steps": [self.source_step()]}},
            {"role": "assistant", "content": None},
            {"role": "assistant", "content": "Plan: " + valid},
            {"role": "assistant", "content": valid, "tool_calls": {}},
            {"role": "assistant", "content": valid, "tool_calls": "not-a-list"},
            {"role": "assistant", "content": valid, "tool_calls": True},
        ]
        for response in responses:
            with self.subTest(response_type=type(response.get("content")).__name__):
                model = ScriptedPlannerModel(response)
                result = self.planner(model, max_attempts=1).plan(self.goal, self.projects)
                self.assertEqual((result.status, result.steps, result.requests), ("failed", (), 1))

    def test_provider_errors_are_failed_results_without_raw_exception_payload_in_log(self):
        private = "PRIVATE_HTTP_EXCEPTION_BODY"
        for error in (ModelError("Локальный API вернул HTTP 401. Проверьте настройки."), RuntimeError(private)):
            with self.subTest(error_type=type(error).__name__):
                model = ScriptedPlannerModel(error)
                result = self.planner(model).plan(self.goal, self.projects)
                self.assertEqual((result.status, result.steps, result.requests), ("failed", (), 1))
                self.assertTrue(result.message)
                self.assertNotIn(private, result.message)
                self.assert_log_private(private, str(error))

    def test_invalid_goals_and_attempt_limits_fail_before_model_calls(self):
        self.assertTrue(issubclass(PlanningError, ValueError))
        for task in (None, "", " \n ", "x" * 16_385, "\ud800"):
            with self.subTest(task_type=type(task).__name__):
                model = ScriptedPlannerModel()
                with self.assertRaises(PlanningError):
                    self.planner(model).plan(task, self.projects)
                self.assertEqual(model.requests, [])
        for attempts in (0, 3, True, 1.5, "2"):
            with self.subTest(attempts=attempts):
                model = ScriptedPlannerModel()
                with self.assertRaises(PlanningError):
                    self.planner(model, max_attempts=attempts).plan(self.goal, self.projects)
                self.assertEqual(model.requests, [])

    def test_invalid_project_roots_fail_before_model_calls(self):
        extra = []
        for index in range(3):
            path = self.root / f"extra-{index}"
            path.mkdir()
            extra.append(path)
        candidates = [[], self.projects + extra,
                      [self.projects[0], self.projects[0]],
                      [self.projects[0], self.projects[0] / ".." / self.projects[0].name],
                      [self.root / "missing"], [self.projects[0] / "README.md"]]
        for projects in candidates:
            with self.subTest(projects=projects):
                model = ScriptedPlannerModel()
                with self.assertRaises(PlanningError):
                    self.planner(model).plan(self.goal, projects)
                self.assertEqual(model.requests, [])

    def test_reparse_root_is_rejected_before_model_calls(self):
        original_lstat = Path.lstat
        marked_root = self.projects[0]

        def marked_lstat(path, *args, **kwargs):
            info = original_lstat(path, *args, **kwargs)
            if path == marked_root:
                return SimpleNamespace(st_mode=info.st_mode, st_file_attributes=0x400)
            return info

        model = ScriptedPlannerModel()
        with patch.object(Path, "lstat", marked_lstat), self.assertRaises(PlanningError):
            self.planner(model).plan(self.goal, [marked_root])
        self.assertEqual(model.requests, [])


if __name__ == "__main__":
    unittest.main()
