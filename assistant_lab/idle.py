"""Автономная разработка представлений Study Planner."""

from datetime import datetime, timezone
from itertools import product
import json
from pathlib import Path
import tempfile
import time
from uuid import uuid4

from study_planner import SORT_FIELDS, STATUSES, check_view_name, load_view
from study_planner.roadmap import BACKLOG, CHECK_TASKS, check_view

from .activity import ActivityBusy, activity, check_regular_file, ensure_directory
from .agent import AgentRunner, SYSTEM_PROMPT
from .config import ConfigError
from .events import EventLog
from .owner import run_assignment
from .providers import DemoModel
from .skills import SkillError
from .task_queue import TaskQueue


class ViewTools:
    def __init__(self, pending: list[str]):
        self.pending = pending
        self.proposal = None

    @property
    def definitions(self):
        return [
            {"type": "function", "function": {
                "name": "list_work", "description": "Посмотреть незавершённые задачи своего проекта.",
                "strict": True, "parameters": {"type": "object", "properties": {},
                                               "required": [], "additionalProperties": False}}},
            {"type": "function", "function": {
                "name": "propose_view", "description": "Предложить фильтр и порядок сортировки и проверить их на заданиях.",
                "strict": True, "parameters": {"type": "object", "properties": {
                    "name": {"type": "string", "enum": self.pending},
                    "status": {"type": "string", "enum": list(STATUSES)},
                    "sort_by": {"type": "string", "enum": list(SORT_FIELDS)},
                    "descending": {"type": "boolean"}},
                    "required": ["name", "status", "sort_by", "descending"], "additionalProperties": False}}},
        ]

    def execute(self, name, arguments):
        if name == "list_work" and arguments == {}:
            return {"ok": True, "tasks": [{"name": key, "goal": BACKLOG[key][0]} for key in self.pending]}
        if name != "propose_view" or not isinstance(arguments, dict):
            return {"ok": False, "error": "Ожидается предложение представления."}
        key = arguments.get("name")
        if key not in self.pending or self.proposal is not None:
            return {"ok": False, "error": "Выберите одну незавершённую задачу за цикл."}
        rule = {k: v for k, v in arguments.items() if k != "name"}
        try:
            passed = check_view(key, rule)
        except ValueError as error:
            return {"ok": False, "error": str(error)}
        if not passed:
            return {"ok": False, "error": "Фильтр или порядок на контрольных данных расходится с целью.",
                    "goal": BACKLOG[key][0]}
        self.proposal = key, rule
        return {"ok": True, "status": "validated", "name": key}


class IdleDemoModel(DemoModel):
    def complete(self, messages, tools):
        previous = messages[-1]
        if previous["role"] != "tool":
            return self._call("work", "list_work")
        result = json.loads(previous["content"])
        if previous["tool_call_id"] == "work" and result["ok"]:
            key = result["tasks"][0]["name"]
            for status, field, descending in product(STATUSES, SORT_FIELDS, (False, True)):
                rule = {"status": status, "sort_by": field, "descending": descending}
                if check_view(key, rule):
                    return self._call("propose", "propose_view", name=key, **rule)
            return {"role": "assistant", "content": "Демо не нашло правила для выбранной задачи."}
        return {"role": "assistant", "content": "Проверка предложения завершена."}


def _pending_work(catalog: Path) -> list[str]:
    ensure_directory(catalog)
    pending = []
    for name in BACKLOG:
        check_view_name(name)
        directory = catalog / name
        try:
            directory.lstat()
        except FileNotFoundError:
            pending.append(name)
            continue
        try:
            ensure_directory(directory)
            for filename in ("view.json", "report.md"):
                if not 0 < check_regular_file(directory / filename).st_size <= 16384:
                    raise ValueError
            if not check_view(name, load_view(catalog, name)):
                raise ValueError
        except (OSError, ValueError, RecursionError):
            raise ConfigError(f"Проверьте сохранённые файлы представления {name} в {catalog}.") from None
    return pending


def _publish(catalog, name, rule, run_id, result, demo):
    # Stage both files on the same filesystem. Readers see the whole change
    # only after the directory rename. Interrupted staging isn't a finished task.
    ensure_directory(catalog)
    with tempfile.TemporaryDirectory(dir=catalog, prefix=".staging-") as temporary:
        staging = Path(temporary)
        view = staging / name
        view.mkdir()
        (view / "view.json").write_text(json.dumps(rule, indent=2) + "\n", encoding="utf-8")
        if not check_view(name, load_view(staging, name)):
            raise ConfigError("Сохранённое представление не прошло повторную проверку.")
        report = (f"# Study Planner · {name}\n\n{BACKLOG[name][0]}\n\n"
                  f"Режим: {'сценарное демо' if demo else 'локальная модель'}.\n"
                  f"Запуск: {run_id}.\n\nПроверены прямая и обратная последовательности "
                  f"из {len(CHECK_TASKS)} учебных заданий и пустой список. "
                  f"Ожидаемый порядок на контрольной выборке: {', '.join(BACKLOG[name][1])}.\n\n"
                  f"Обращений к модели: {result.steps}. Вызовов инструментов: {result.tool_calls}.\n")
        (view / "report.md").write_text(report, encoding="utf-8")
        if (catalog / name).exists() or (catalog / name).is_symlink():
            raise ConfigError(f"Путь представления {name} уже занят.")
        view.rename(catalog / name)


def _cycle(queue, runs, catalog, model, *, demo, max_steps, max_calls, read_budget, secret):
    def busy():
        try:
            with activity(runs, exclusive=True):
                return queue.has_pending(demo)
        except ActivityBusy:
            return True

    with activity(runs, exclusive=True):
        task = queue.claim_next(demo)
    if task is not None:
        print(f"Выполняю поручение владельца #{task['id']}.")
        try:
            success = run_assignment(queue.root, DemoModel() if demo else model,
                                     project=Path(task["project"]), task=task["task"], skill_mode=not demo,
                                     max_steps=max_steps, max_calls=max_calls, read_budget=read_budget,
                                     secret=secret, queue_run_id=task["run_id"])
        except (OSError, ValueError, SkillError):
            success = queue.has_report(task["run_id"])
            if not success:
                print("Ошибка доступа к файлам или настройки поручения.")
        queue.finish(task, success)
        print(f"Поручение #{task['id']}: {'done' if success else 'failed'}.")
        return "owner_done" if success else "owner_failed"

    pending = _pending_work(catalog)
    if not pending:
        with activity(runs, exclusive=True):
            if queue.has_pending(demo):
                return "queued"
        print("План Study Planner выполнен.")
        return "finished"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{stamp}-{uuid4().hex[:8]}"
    tools = ViewTools(pending)
    log = EventLog(runs / f"{run_id}.jsonl", secrets=[secret] if secret else [])
    prompt = (SYSTEM_PROMPT + "\nРазвивай свой проект Study Planner. Посмотри list_work, "
              "выбери одну задачу и предложи её реализацию через propose_view. При ошибке "
              "исправь правило. После проверки заверши ответ.")
    runner = AgentRunner(model, tools, log, max_steps=max_steps, max_tool_calls=max_calls,
                         on_event=print, should_yield=busy, system_prompt=prompt)
    result = runner.run("Реализуй одно новое представление списка заданий.")
    if result.status == "yielded":
        if queue.has_pending(demo):
            return "queued"
        raise ActivityBusy
    if result.status != "completed" or not tools.proposal:
        print(result.final_text)
        print("За цикл не подготовлено завершённое изменение.")
        return "failed"
    name, rule = tools.proposal
    try:
        with activity(runs, exclusive=True):
            if queue.has_pending(demo):
                log.emit("idle_deferred", status="queued")
                return "queued"
            _publish(catalog, name, rule, run_id, result, demo)
    except ActivityBusy:
        log.emit("idle_deferred", status="busy")
        raise
    log.emit("idle_published", status="completed")
    print(f"Добавлено представление {name}. Отчёт: {catalog / name / 'report.md'}")
    return "finished" if len(pending) == 1 else "published"


def run_idle(root, model, *, demo, max_steps, max_calls, read_budget, secret, interval=None):
    runs = root / "runs"
    queue = TaskQueue(root)
    catalog = root / "workspace" / "idle-demo" / "views" if demo else root / "study_planner" / "views"
    while True:
        try:
            with activity(runs / "idle-worker", exclusive=True):
                outcome = _cycle(queue, runs, catalog, model, demo=demo, max_steps=max_steps,
                                 max_calls=max_calls, read_budget=read_budget, secret=secret)
        except ActivityBusy:
            print("Автономная работа отложена: занят сеанс ассистента.")
            outcome = "busy"
        if outcome == "queued":
            continue
        if outcome == "failed" or (interval is None and outcome == "owner_failed"):
            return 1
        if interval is None or outcome == "finished":
            return 0
        time.sleep(interval)
