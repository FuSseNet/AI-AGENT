"""Инструменты скилла управления очередью. Каталог проекта задаёт владелец."""

from datetime import date

from .config import ConfigError
from .task_queue import STATES


class TaskTools:
    def __init__(self, queue, project):
        self.queue = queue
        self.project = project
        self.last_call_ok = False
        self.created = {}

    @property
    def definitions(self):
        date_field = {"type": "string", "description": "Календарная дата ГГГГ-ММ-ДД."}
        specs = [
            ("create_task", "Сохранить новое поручение владельца в текущем проекте.",
             {"task": {"type": "string", "minLength": 1, "maxLength": 2048}, "due": date_field}, ["task"]),
            ("list_tasks", f"Посмотреть поручения. По умолчанию открытые. Сегодня {date.today().isoformat()}. "
             "due — последний день включительно, offset — смещение следующей страницы.",
             {"status": {"type": "string", "enum": ["active", "all", *STATES]},
              "due": date_field, "offset": {"type": "integer", "minimum": 0}}, []),
            ("complete_task", "Отметить указанное владельцем поручение выполненным вручную.",
             {"id": {"type": "integer", "minimum": 1}}, ["id"]),
        ]
        return [{"type": "function", "function": {
            "name": name, "description": description, "parameters": {"type": "object", "properties": fields,
            "required": required, "additionalProperties": False}}} for name, description, fields, required in specs]

    def execute(self, name, args):
        self.last_call_ok = False
        if not isinstance(args, dict):
            return {"ok": False, "error": "Ожидается объект с аргументами."}
        try:
            if name == "create_task" and set(args) in ({"task"}, {"task", "due"}):
                task, due = args["task"], args.get("due")
                if not isinstance(task, str) or (due is not None and not isinstance(due, str)):
                    raise ConfigError("Нужен текст поручения и, при наличии, дата.")
                key = task.strip(), due
                if key not in self.created:
                    self.created[key] = self.queue.add(task, self.project, due)
                result = {"id": self.created[key], "status": "pending"}
            elif name == "list_tasks" and set(args) <= {"status", "due", "offset"}:
                result = self.queue.list_tasks(**({"status": "active"} | args))
            elif name == "complete_task" and set(args) == {"id"}:
                self.queue.change(args["id"])
                result = {"id": args["id"], "status": "done"}
            else:
                raise ConfigError("Неизвестный инструмент очереди или неверные аргументы.")
        except ConfigError as error:
            return {"ok": False, "error": str(error)}
        self.last_call_ok = True
        return {"ok": True, **result}
