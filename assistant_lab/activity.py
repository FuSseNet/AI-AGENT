"""Короткая эксклюзивная секция для публикации и совместимые сеансы владельца."""

from contextlib import closing, contextmanager
from pathlib import Path
import sqlite3
import stat

from .config import ConfigError


def ensure_directory(path: Path) -> None:
    for part in reversed([path, *path.parents]):
        part.mkdir(exist_ok=True)
        info = part.lstat()
        if (not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode)
                or getattr(info, "st_file_attributes", 0) & 0x400):
            raise ConfigError("Служебный путь должен проходить через обычные каталоги.")


class ActivityBusy(Exception):
    """Сейчас работает другой сеанс ассистента."""


def check_regular_file(path: Path):
    info = path.lstat()
    if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
            or getattr(info, "st_file_attributes", 0) & 0x400):
        raise ConfigError("Для работы требуется обычный файл без ссылок.")
    return info


@contextmanager
def database(path: Path, timeout: int = 5):
    ensure_directory(path.parent)
    for suffix in ("", "-journal", "-wal", "-shm"):
        try:
            check_regular_file(path.with_name(path.name + suffix))
        except FileNotFoundError:
            continue
    with closing(sqlite3.connect(path, timeout=timeout)) as connection:
        yield connection


@contextmanager
def activity(directory: Path, *, exclusive: bool):
    # The OS releases these locks even if a worker is killed.
    with database(directory / "activity.sqlite3", timeout=0 if exclusive else 5) as connection:
        try:
            connection.execute("BEGIN EXCLUSIVE" if exclusive else "BEGIN")
            connection.execute("SELECT count(*) FROM sqlite_schema").fetchone()
        except sqlite3.OperationalError as error:
            if getattr(error, "sqlite_errorcode", None) in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED):
                raise ActivityBusy from None
            raise
        yield
