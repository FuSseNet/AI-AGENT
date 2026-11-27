"""Общий путь выполнения немедленного и сохранённого поручения."""

from datetime import datetime, timezone
from uuid import uuid4

from .activity import activity, ensure_directory
from .agent import AgentRunner
from .events import EventLog
from .skill_tools import SkillToolSet
from .skills import SkillRegistry
from .tools import FileTools


def run_assignment(root, model, *, project, task, skill_mode, max_steps, max_calls, read_budget,
                   secret, source="README.md", selected=None, task_tools=None, queue_run_id=None):
    run_id = queue_run_id or (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])
    report_dir = root / "workspace" / "reports" / run_id
    logs_dir = root / "runs"
    ensure_directory(report_dir)
    ensure_directory(logs_dir)
    report = report_dir / "report.md"
    destination = report_dir / ".draft.md" if queue_run_id else report
    files = FileTools(project, destination, required_source=None if skill_mode else source,
                      project_mode=skill_mode, max_total_read_bytes=read_budget)
    log_path = logs_dir / f"{run_id}.jsonl"
    log = EventLog(log_path, secrets=[secret] if secret else [])
    skills = (SkillToolSet(files, SkillRegistry(), log, selected_name=selected, task_tools=task_tools)
              if skill_mode else None)
    if task is None:
        if skill_mode:
            task = ("Подготовь обзор выбранного проекта и сохрани отчёт." if selected == "project-review"
                    else "Выполни выбранный скилл.")
        else:
            task = (f"Прочитай файл {source!r} с помощью read_file. "
                    "Составь краткое описание назначения проекта, его возможностей и ограничений, "
                    "опираясь только на прочитанный документ. Сохрани отчёт через save_report. "
                    "Если файл недоступен, сообщи об ошибке и не выдумывай содержимое.")
    runner = AgentRunner(model, skills or files, log, max_steps=max_steps, max_tool_calls=max_calls,
                         on_event=print, system_prompt=skills.system_prompt if skills else None)
    with activity(logs_dir, exclusive=False):
        result = runner.run(task)
    print("\n" + result.final_text)
    print(f"Запросов к модели: {result.steps}; вызовов инструментов: {result.tool_calls}.")
    print(f"Журнал: {log_path}")
    completed = result.status == "completed"
    if skills:
        print("Выбранный скилл: " + (skills.selected_name or "не выбран"))
        if skills.selected_name == "manage-tasks":
            return completed and task_tools.last_call_ok
        print("Фактически прочитанные файлы:")
        for path in files.read_paths:
            print("  " + path)
        if not files.read_paths:
            print("  Ни одного файла не прочитано.")
    if not files.report_saved or not destination.is_file():
        print("Отчёт не создан: поручение ещё не выполнено; возможно, требуется уточнение входных данных.")
        return False
    if queue_run_id:
        if not completed:
            return False
        # A final filename serves as a receipt if the process exits before
        # the queue transaction is committed. A draft alone never closes a task.
        destination.rename(report)
    print(f"Отчёт: {report}")
    return completed
