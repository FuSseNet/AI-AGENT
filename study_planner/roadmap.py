"""Задачи Study Planner и примеры для приёмки новых представлений."""

from . import apply_view


# Контрольная выборка отличается от данных, показываемых планировщиком.
CHECK_TASKS = [
    {"title": "Г", "subject": "Б", "due": "2026-03-02", "status": "open"},
    {"title": "Б", "subject": "А", "due": "2026-03-03", "status": "done"},
    {"title": "А", "subject": "В", "due": "2026-03-04", "status": "open"},
    {"title": "Д", "subject": "А", "due": "2026-03-01", "status": "done"},
    {"title": "В", "subject": "В", "due": "2026-03-01", "status": "open"},
    {"title": "Е", "subject": "В", "due": "2026-03-05", "status": "done"},
]
# У каждой задачи — цель и ожидаемые названия заданий в порядке выдачи.
BACKLOG = {
    "upcoming": ("Показать незавершённые задания по возрастанию срока сдачи.", ("В", "Г", "А")),
    "completed": ("Показать выполненные задания по убыванию срока сдачи.", ("Е", "Б", "Д")),
    "by-subject": ("Показать все задания по предмету, затем названию, по возрастанию.",
                   ("Б", "Д", "Г", "А", "В", "Е")),
}


def check_view(name: str, rule: dict) -> bool:
    for tasks in (CHECK_TASKS, list(reversed(CHECK_TASKS))):
        actual = apply_view(tasks, rule)
        if [task["title"] for task in actual] != list(BACKLOG[name][1]):
            return False
    return not apply_view([], rule)
