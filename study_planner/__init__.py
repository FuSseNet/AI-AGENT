"""Study Planner: именованные представления списка учебных заданий."""

from datetime import date
import json
from pathlib import Path
import re


SAMPLE_TASKS = [
    {"title": "Эссе", "subject": "История", "due": "2026-12-03", "status": "open"},
    {"title": "Зачёт", "subject": "Математика", "due": "2026-12-01", "status": "done"},
    {"title": "Доклад", "subject": "Физика", "due": "2026-12-02", "status": "open"},
    {"title": "Анализ", "subject": "История", "due": "2026-12-04", "status": "done"},
]
SORT_FIELDS = ("title", "subject", "due")
STATUSES = ("all", "open", "done")
ALL_TASKS = {"status": "all", "sort_by": "title", "descending": False}


def apply_view(tasks: list[dict], rule: dict) -> list[dict]:
    if (not isinstance(rule, dict) or set(rule) != set(ALL_TASKS)
            or rule["status"] not in STATUSES or rule["sort_by"] not in SORT_FIELDS
            or type(rule["descending"]) is not bool):
        raise ValueError("Некорректный фильтр или порядок сортировки.")
    if not isinstance(tasks, list):
        raise ValueError("Ожидается JSON-массив заданий.")
    selected = []
    for task in tasks:
        if (not isinstance(task, dict) or task.get("status") not in ("open", "done")
                or not all(isinstance(task.get(key), str) and task[key].strip() for key in SORT_FIELDS)):
            raise ValueError("Заданию нужны title, subject, due и status (open/done).")
        try:
            due = date.fromisoformat(task["due"])
        except ValueError:
            raise ValueError("Для срока сдачи нужна календарная дата в формате ГГГГ-ММ-ДД.") from None
        if due.isoformat() != task["due"]:
            raise ValueError("Для срока сдачи нужен формат ГГГГ-ММ-ДД.")
        if rule["status"] in ("all", task["status"]):
            selected.append(task)
    return sorted(selected, key=lambda task: (task[rule["sort_by"]], task["title"]),
                  reverse=rule["descending"])


def check_view_name(name: str) -> None:
    if (not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9-]{0,49}", name)
            or re.fullmatch(r"all|aux|con|nul|prn|(?:com|lpt)[1-9]", name)):
        raise ValueError("Недопустимое имя представления.")


def load_view(catalog: Path, name: str) -> dict:
    if name == "all":
        return ALL_TASKS.copy()
    check_view_name(name)
    return json.loads((catalog / name / "view.json").read_text(encoding="utf-8"))
