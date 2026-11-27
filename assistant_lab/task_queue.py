"""Очередь поручений владельца между запусками ассистента."""

from contextlib import contextmanager
from datetime import date
from pathlib import Path
import re
import sqlite3
from uuid import uuid4

from .activity import activity, check_regular_file, database, ensure_directory
from .config import ConfigError


DEMO_TASK = "Прочитай README.md и сохрани демонстрационную выдержку."
STATES = ("pending", "running", "done", "failed")
PAGE_SIZE = 10
ORDER = "due IS NULL, due, id"


def _date(value):
    if value is None:
        return None
    try:
        if date.fromisoformat(value).isoformat() == value:
            return value
    except (ValueError, TypeError):
        pass
    raise ConfigError("Для срока нужна календарная дата ГГГГ-ММ-ДД.")


class TaskQueue:
    def __init__(self, root: Path):
        self.root = root
        self.path = root / "workspace" / ".assistant" / "tasks.sqlite3"

    @contextmanager
    def connect(self):
        with database(self.path) as db, db:
            db.row_factory = sqlite3.Row
            db.execute("""CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY, task TEXT NOT NULL, project TEXT NOT NULL,
                due TEXT, demo INTEGER NOT NULL, status TEXT NOT NULL DEFAULT 'pending'
                    CHECK(status IN ('pending','running','done','failed')),
                run_id TEXT UNIQUE)""")
            yield db

    def add(self, task: str, project: Path, due=None, *, demo=False) -> int:
        if not isinstance(task, str) or not 1 <= len(task.strip()) <= 2048 or "\0" in task:
            raise ConfigError("Поручение должно содержать от 1 до 2048 символов текста.")
        due = _date(due)
        # Coordinates enqueue with the idle worker's last check and publication.
        with activity(self.root / "runs", exclusive=False), self.connect() as db:
            return db.execute("INSERT INTO tasks(task,project,due,demo) VALUES(?,?,?,?)",
                              (task.strip(), str(project.absolute()), due, demo)).lastrowid

    def report_path(self, run_id: str) -> Path:
        if not isinstance(run_id, str) or not re.fullmatch(r"[a-f0-9]{32}", run_id):
            raise ConfigError("Некорректная ссылка на запуск в очереди.")
        return self.root / "workspace" / "reports" / run_id / "report.md"

    def has_report(self, run_id: str) -> bool:
        path = self.report_path(run_id)
        if not path.parent.exists():
            return False
        ensure_directory(path.parent)
        try:
            return check_regular_file(path).st_size > 0
        except FileNotFoundError:
            return False

    def list_tasks(self, status="all", due=None, offset=0):
        if status not in (*STATES, "active", "all"):
            raise ConfigError("Неизвестный статус поручения.")
        if type(offset) is not int or not 0 <= offset <= 2**63 - 1:
            raise ConfigError("Смещение списка должно быть неотрицательным целым числом.")
        due = _date(due)
        sql = f"""SELECT * FROM tasks WHERE
                  (?='all' OR status=? OR (?='active' AND status!='done'))
                  AND (? IS NULL OR due<=?) ORDER BY {ORDER} LIMIT ? OFFSET ?"""
        with self.connect() as db:
            rows = db.execute(sql, (status, status, status, due, due, PAGE_SIZE + 1, offset)).fetchall()
        items = []
        for row in rows[:PAGE_SIZE]:
            item = dict(row)
            run_id = item.pop("run_id")
            item["report"] = None
            item["log"] = None
            if run_id:
                report = self.report_path(run_id)
                if item["status"] == "done":
                    item["report"] = str(report)
                log = self.root / "runs" / f"{run_id}.jsonl"
                if log.is_file():
                    item["log"] = str(log)
            item["demo"] = bool(item["demo"])
            items.append(item)
        return {"tasks": items, "next_offset": offset + PAGE_SIZE if len(rows) > PAGE_SIZE else None}

    def change(self, task_id, *, retry=False):
        if type(task_id) is not int or not 1 <= task_id <= 2**63 - 1:
            raise ConfigError("Нужен положительный числовой ID поручения.")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT status FROM tasks WHERE id=?", (task_id,)).fetchone()
            if row is None:
                raise ConfigError("Поручение с таким ID не найдено.")
            if row["status"] == "done" and not retry:
                return
            if row["status"] == "running" or (retry and row["status"] != "failed"):
                raise ConfigError("Операция недоступна для текущего статуса поручения.")
            db.execute("UPDATE tasks SET status=?, run_id=NULL WHERE id=?",
                       ("pending" if retry else "done", task_id))

    def has_pending(self, demo: bool) -> bool:
        with self.connect() as db:
            return db.execute("SELECT 1 FROM tasks WHERE demo=? AND status='pending' LIMIT 1",
                              (demo,)).fetchone() is not None

    def claim_next(self, demo: bool):
        """Вызывается только владельцем межпроцессной блокировки idle-worker."""
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            for row in db.execute("SELECT id,run_id FROM tasks WHERE status='running' AND demo=?",
                                  (demo,)).fetchall():
                # Only a successful, verified run promotes its draft to report.md.
                if self.has_report(row["run_id"]):
                    db.execute("UPDATE tasks SET status='done' WHERE id=?", (row["id"],))
                else:
                    db.execute("UPDATE tasks SET status='pending',run_id=NULL WHERE id=?", (row["id"],))
            row = db.execute(f"SELECT * FROM tasks WHERE demo=? AND status='pending' ORDER BY {ORDER} LIMIT 1",
                             (demo,)).fetchone()
            if row is None:
                return None
            run_id = uuid4().hex
            db.execute("UPDATE tasks SET status='running',run_id=? WHERE id=?", (run_id, row["id"]))
            return dict(row) | {"status": "running", "run_id": run_id}

    def finish(self, row, success: bool):
        with self.connect() as db:
            updated = db.execute("UPDATE tasks SET status=? WHERE id=? AND status='running' AND run_id=?",
                                 ("done" if success else "failed", row["id"], row["run_id"]))
            if updated.rowcount != 1:
                raise ConfigError("Поручение уже не принадлежит этому запуску.")
