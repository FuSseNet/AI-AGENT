"""Запуск: python -m assistant_lab --demo."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
import sys
from uuid import uuid4

from .activity import ActivityBusy, activity, ensure_directory
from .config import ConfigError, load_env, positive_int
from .events import EventLog
from .idle import IdleDemoModel, run_idle
from .owner import run_assignment
from .planner import ModelPlanner, PlanningError
from .providers import DemoModel, LocalChatModel, ModelError
from .skills import SkillError, SkillRegistry
from .task_queue import DEMO_TASK, STATES, TaskQueue
from .task_tools import TaskTools
from .workflows import WorkflowError, WorkflowRunner, project_comparison_plan, validate_project_roots


ROOT = Path(__file__).resolve().parent.parent


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Учебный ассистент: поручения со скиллами, обзор проекта и сравнение документов.")
    result.add_argument("task", nargs="?", help="Поручение для настоящей модели")
    modes = result.add_mutually_exclusive_group()
    modes.add_argument("--demo", action="store_true", help="Сценарная имитация без модели и сети")
    modes.add_argument("--idle", action="store_true", help="Выполнить поручение из очереди или развивать Study Planner")
    modes.add_argument("--add-task", metavar="ТЕКСТ", help="Записать поручение для последующего выполнения")
    modes.add_argument("--add-demo-task", action="store_true", help="Записать учебное поручение по README для --demo-idle")
    modes.add_argument("--list-tasks", action="store_true", help="Показать сохранённые поручения и пути к результатам")
    modes.add_argument("--complete-task", type=int, metavar="ID", help="Отметить поручение выполненным вручную")
    modes.add_argument("--retry-task", type=int, metavar="ID", help="Вернуть неудавшееся поручение в очередь")
    result.add_argument("--due", metavar="ГГГГ-ММ-ДД", help="Срок поручения или верхняя граница при просмотре")
    result.add_argument("--status", choices=["active", "all", *STATES], help="Фильтр для --list-tasks")
    result.add_argument("--offset", type=int, help="Смещение страницы --list-tasks")
    modes.add_argument("--demo-idle", action="store_true", help="Автономный цикл на учебном каталоге без модели")
    result.add_argument("--idle-interval", metavar="СЕКУНДЫ",
                        help="Повторять автономные циклы с паузой до выполнения плана")
    modes.add_argument("--demo-search", metavar="ТЕКСТ",
                       help="Сценарное демо: найти фразу через MCP, прочитать строку и сохранить отчёт")
    modes.add_argument("--check", action="store_true", help="Проверить API и показать доступные модели")
    modes.add_argument("--describe-project", action="store_true",
                       help="Явно выбрать скилл project-review (прежняя команда обзора)")
    modes.add_argument("--skill", metavar="NAME", help="Явно выбрать скилл по имени")
    modes.add_argument("--list-skills", action="store_true", help="Показать скиллы без модели и сети")
    modes.add_argument("--compare-projects", action="store_true",
                       help="Обзоры 2–4 проектов, затем сравнение; повторите --project для каждой папки")
    modes.add_argument("--auto-plan", action="store_true",
                       help="Построить план по поручению и выполнить его в разрешённых папках")
    result.add_argument("--project", type=Path, action="append",
                        help="Разрешённая папка; для --auto-plan и --compare-projects можно повторять")
    result.add_argument("--source", default="README.md", help="Исходный файл для стандартного поручения")
    result.add_argument("--env-file", type=Path, default=ROOT / ".env", help="Файл настроек")
    result.add_argument("--base-url", help="Локальный адрес API, включая /v1")
    result.add_argument("--model", help="Точное имя модели из локального API")
    result.add_argument("--max-steps", help="Максимум запросов к модели (по умолчанию 8)")
    result.add_argument("--max-tool-calls", help="Максимум вызовов инструментов (по умолчанию 8)")
    result.add_argument("--read-budget", help="Общий бюджет чтения для скиллов, байты (по умолчанию 16384)")
    result.add_argument("--timeout", help="Таймаут HTTP-запроса в секундах (по умолчанию 60)")
    return result


def run_workflow(args, projects, plan, model, max_steps, max_calls, read_budget, secret) -> int:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    report_dir = ROOT / "workspace" / "reports" / run_id
    logs_dir = ROOT / "runs"
    ensure_directory(report_dir)
    ensure_directory(logs_dir)
    log_path = logs_dir / f"{run_id}.jsonl"
    log = EventLog(log_path, secrets=[secret] if secret else [])
    planning_requests = 0
    registry = None
    if args.auto_plan:
        registry = SkillRegistry()
        print("Планирование той же локальной моделью. Разрешённые папки:")
        for index, project in enumerate(projects, 1):
            print(f"  project-{index}: {project}")
        proposed = ModelPlanner(model, registry, log).plan(args.task, projects)
        planning_requests = proposed.requests
        print(f"Запросов к модели на планирование: {planning_requests}.")
        if proposed.status != "ready":
            print(proposed.message)
            print(f"Всего запросов к модели: {planning_requests}; вызовов инструментов: 0.")
            print(f"Журнал: {log_path}")
            return 1
        plan = list(proposed.steps)
        aliases = {os.path.normcase(str(project)): f"project-{index}"
                   for index, project in enumerate(projects, 1)}
        plan_data = {"projects": [{"alias": f"project-{index}", "name": project.name}
                                   for index, project in enumerate(projects, 1)], "steps": []}
        print("Проверенный план модели:")
        for step in plan:
            source = aliases[os.path.normcase(str(step.project_dir))] if step.project_dir else None
            plan_data["steps"].append({"id": step.step_id, "skill": step.skill, "task": step.task,
                                      "project": source, "depends_on": list(step.depends_on)})
            print(f"  {step.step_id}: {step.skill}; входы: "
                  + (source if source else ", ".join(step.depends_on)))
        plan_path = report_dir / "plan.json"
        with plan_path.open("x", encoding="utf-8") as stream:
            json.dump(plan_data, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        print(f"План сохранён: {plan_path}")
    if not args.auto_plan:
        print("План: отдельный обзор каждого проекта, затем сравнение обзоров.")
    for step in plan:
        if step.project_dir is not None:
            print(f"  {step.step_id}: {step.project_dir}")
    outcome = WorkflowRunner(model, registry or SkillRegistry(), log, report_dir, max_steps=max_steps,
                             max_tool_calls=max_calls, read_budget=read_budget,
                             on_event=print).run(plan)
    print("\n" + outcome.final_text)
    print(f"Всего запросов к модели: {outcome.steps + planning_requests}; "
          f"вызовов инструментов: {outcome.tool_calls}.")
    if args.auto_plan:
        print(f"Из них планирование: {planning_requests}; выполнение этапов: {outcome.steps}.")
    print(f"Журнал: {log_path}")
    for stage in outcome.stages:
        print(f"Этап {stage.step_id}: {stage.result.status}")
        if stage.report_path is not None:
            print(f"  Отчёт: {stage.report_path}")
    if outcome.status == "completed":
        if args.auto_plan:
            dependencies = {dep for step in plan for dep in step.depends_on}
            print("Итоговые отчёты:")
            for stage in outcome.stages:
                if stage.step_id not in dependencies:
                    print("  " + str(stage.report_path))
        else:
            print("Итоговое сравнение: " + str(outcome.stages[-1].report_path))
    return 0 if outcome.status == "completed" else 1


def main(argv: list[str] | None = None) -> int:
    arg_parser = parser()
    args = arg_parser.parse_args(argv)
    search_demo = args.demo_search is not None
    idle_mode = args.idle or args.demo_idle
    adding = args.add_task is not None or args.add_demo_task
    managing = adding or args.list_tasks or args.complete_task is not None or args.retry_task is not None
    if (args.status is not None or args.offset is not None) and not args.list_tasks:
        arg_parser.error("--status и --offset применяются с --list-tasks.")
    if args.due is not None and not (adding or args.list_tasks):
        arg_parser.error("--due применяется при добавлении или просмотре поручений.")
    if managing and args.task is not None:
        arg_parser.error("Текст нового поручения указывается в --add-task.")
    if args.idle_interval is not None and not idle_mode:
        arg_parser.error("--idle-interval применяется с --idle или --demo-idle.")
    if idle_mode and (args.task is not None or args.project is not None):
        arg_parser.error("Автономный режим использует каталоги, сохранённые при добавлении поручений.")
    if args.task and (args.demo or search_demo or args.check or args.list_skills):
        arg_parser.error("Поручение задаётся только в режиме настоящей модели.")
    if args.auto_plan and not args.task:
        arg_parser.error("Для --auto-plan напишите поручение в кавычках.")
    projects = args.project or [ROOT / "workspace" / "project"]
    if len(projects) > 1 and not (args.compare_projects or args.auto_plan):
        arg_parser.error("Несколько --project поддерживаются с --compare-projects или --auto-plan.")
    try:
        queue = TaskQueue(ROOT)
        if adding:
            task_id = queue.add(DEMO_TASK if args.add_demo_task else args.add_task, projects[0],
                                args.due, demo=args.add_demo_task)
            print(f"Сохранено поручение #{task_id}.")
            return 0
        if args.list_tasks:
            page = queue.list_tasks(args.status or "all", args.due, args.offset or 0)
            for item in page["tasks"]:
                mode = " [демо]" if item["demo"] else ""
                due = f" до {item['due']}" if item["due"] else ""
                print(f"#{item['id']} {item['status']}{mode}{due}  {item['task']}")
                if item["report"]:
                    print("  Отчёт: " + item["report"])
                elif item["status"] == "failed" and item["log"]:
                    print("  Журнал: " + item["log"])
            if not page["tasks"]:
                print("Поручений по этим условиям нет.")
            if page["next_offset"] is not None:
                print(f"Следующая страница: --offset {page['next_offset']}")
            return 0
        if args.complete_task is not None or args.retry_task is not None:
            retry = args.retry_task is not None
            task_id = args.retry_task if retry else args.complete_task
            queue.change(task_id, retry=retry)
            print(f"Поручение #{task_id}: {'pending' if retry else 'done'}.")
            return 0
        if args.list_skills:
            for skill in SkillRegistry().list_skills():
                print(f"{skill.name}: {skill.description}")
            return 0
        plan = project_comparison_plan(projects, args.task or "") if args.compare_projects else None
        if args.auto_plan:
            projects = list(validate_project_roots(projects))
        skill_mode = bool(args.task or args.skill or args.describe_project or search_demo)
        config = load_env(args.env_file)
        secret = config.get("ASSISTANT_API_KEY", "")
        if any(char in secret for char in "\r\n"):
            raise ConfigError("ASSISTANT_API_KEY не должен содержать переносы строк.")
        max_steps = positive_int(args.max_steps or config.get("ASSISTANT_MAX_STEPS", "8"),
                                 "max_steps", 50)
        max_calls = positive_int(args.max_tool_calls or config.get("ASSISTANT_MAX_TOOL_CALLS", "8"),
                                 "max_tool_calls", 50)
        read_budget = positive_int(args.read_budget or config.get("ASSISTANT_READ_BUDGET", "16384"),
                                   "read_budget", 1_048_576)
        if args.demo or search_demo or args.demo_idle:
            model = IdleDemoModel() if args.demo_idle else DemoModel(args.source, search_query=args.demo_search)
            print("ДЕМО: сценарная имитация. Нейросеть и сеть не используются.")
        else:
            base_url = args.base_url or config.get("ASSISTANT_BASE_URL", "")
            model_name = args.model or config.get("ASSISTANT_MODEL", "")
            if not base_url:
                raise ConfigError("Не задан ASSISTANT_BASE_URL. "
                                  "Скопируйте .env.example в .env и укажите адрес API из Unsloth. "
                                  "Для запуска без модели используйте --demo.")
            if not model_name and not args.check:
                raise ConfigError("Не задан ASSISTANT_MODEL. "
                                  "Узнайте имя модели командой python -m assistant_lab --check.")
            timeout = positive_int(args.timeout or config.get("ASSISTANT_TIMEOUT", "60"),
                                   "timeout", 600)
            max_tokens = positive_int(config.get("ASSISTANT_MAX_TOKENS", "2048"),
                                      "max_tokens", 16_384)
            model = LocalChatModel(base_url, model_name, secret, timeout, max_tokens)
            if args.check:
                models = model.list_models()
                if not models:
                    print("API доступен, но список моделей пуст. Загрузите модель в Unsloth.")
                    return 1
                print("API доступен. Имена моделей для ASSISTANT_MODEL:")
                for name in models:
                    print("  " + name)
                print("Поддержку инструментов проверяет запуск стандартного поручения.")
                return 0
            print("Режим: настоящая локальная модель через API.")

        if args.auto_plan or plan is not None:
            with activity(ROOT / "runs", exclusive=False):
                return run_workflow(args, projects, plan, model, max_steps, max_calls, read_budget, secret)

        if idle_mode:
            interval = (positive_int(args.idle_interval, "idle_interval", 86400)
                        if args.idle_interval is not None else None)
            return run_idle(ROOT, model, demo=args.demo_idle, max_steps=max_steps,
                            max_calls=max_calls, read_budget=read_budget, secret=secret, interval=interval)

        selected = "project-review" if args.describe_project or search_demo else args.skill
        task = (f"Найди фразу {args.demo_search!r} через MCP и сохрани первое подтверждённое совпадение."
                if search_demo else args.task)
        success = run_assignment(ROOT, model, project=projects[0], task=task, selected=selected,
                                 skill_mode=skill_mode, source=args.source, max_steps=max_steps,
                                 max_calls=max_calls, read_budget=read_budget, secret=secret,
                                 task_tools=TaskTools(queue, projects[0]))
        return 0 if success else 1
    except (ConfigError, ModelError, SkillError, WorkflowError, PlanningError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nВыполнение остановлено пользователем.", file=sys.stderr)
        return 130
    except ActivityBusy:
        print("Сеанс занят. Попробуйте повторить поручение.", file=sys.stderr)
        return 1
    except (OSError, ValueError, sqlite3.Error):
        print("Ошибка: не удалось подготовить или выполнить запуск. "
              "Проверьте настройки, каталог проекта и права на служебные файлы.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
