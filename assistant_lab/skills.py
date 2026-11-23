"""Загрузка контролируемой библиотеки навыков как неизменяемого снимка.

Это навыки самого учебного ассистента. Реестр не ищет SKILL.md в проекте
пользователя, не выполняет скрипты и не предоставляет дополнительных прав.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import stat
import unicodedata


DEFAULT_SKILLS_DIR = Path(__file__).parent / "skill_library"
_MAX_SKILLS = 20
_MAX_SKILL_BYTES = 32 * 1024
_MAX_DESCRIPTION_CHARS = 1024
_NAME_PATTERN = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*\Z")


class SkillError(ValueError):
    """Безопасная ошибка проверки библиотеки навыков."""


@dataclass(frozen=True)
class SkillMetadata:
    name: str
    description: str


@dataclass(frozen=True)
class Skill:
    metadata: SkillMetadata
    instructions: str


def _valid_name(name: str) -> bool:
    return isinstance(name, str) and len(name) <= 64 and _NAME_PATTERN.fullmatch(name) is not None


def _check_components(path: Path) -> None:
    """Проверить каждый существующий компонент, не разрешая ссылки и reparse points."""
    for component in [*reversed(path.parents), path]:
        info = component.lstat()
        attributes = getattr(info, "st_file_attributes", 0)
        if (
            stat.S_ISLNK(info.st_mode)
            or attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        ):
            raise SkillError("Ссылки, junction и reparse points в библиотеке навыков запрещены.")
        if component != path and not stat.S_ISDIR(info.st_mode):
            raise SkillError("Родитель навыка должен быть обычной папкой.")


def _check_skill_file(info: os.stat_result) -> None:
    if not stat.S_ISREG(info.st_mode):
        raise SkillError("SKILL.md должен быть обычным файлом.")
    if info.st_nlink != 1:
        raise SkillError("Файлы навыков с жёсткими ссылками запрещены.")
    if info.st_size > _MAX_SKILL_BYTES:
        raise SkillError("Файл навыка превышает лимит 32 КиБ.")


def _description(value: str) -> str:
    if value.startswith('"'):
        try:
            decoded = json.loads(value)
        except (ValueError, RecursionError):
            raise SkillError("Описание в двойных кавычках должно быть JSON-строкой.") from None
        if not isinstance(decoded, str):
            raise SkillError("Описание навыка должно быть строкой.")
        value = decoded
    elif value.startswith(("'", "|", ">", "[", "{")) or value.startswith("- "):
        raise SkillError("Допустимо только однострочное обычное описание или JSON-строка.")
    value = value.strip()
    if not value or len(value) > _MAX_DESCRIPTION_CHARS:
        raise SkillError("Описание навыка должно содержать от 1 до 1024 символов.")
    if any(unicodedata.category(char) in {"Cc", "Zl", "Zp"} for char in value):
        raise SkillError("Описание навыка должно занимать одну строку без управляющих символов.")
    return value


def _parse_skill(data: bytes, directory_name: str) -> Skill:
    try:
        text = data.decode("utf-8-sig", errors="strict")
    except UnicodeError:
        raise SkillError("Файл навыка должен иметь кодировку UTF-8.") from None
    if "\x00" in text:
        raise SkillError("Файл навыка содержит недопустимые символы.")
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].rstrip("\r\n") != "---":
        raise SkillError("SKILL.md должен начинаться с заголовка между строками ---.")
    values: dict[str, str] = {}
    closing_index = None
    for index, line in enumerate(lines[1:], start=1):
        header_line = line.rstrip("\r\n")
        if header_line == "---":
            closing_index = index
            break
        match = re.fullmatch(r"([a-z]+):[ \t]*(.*)", header_line)
        if match is None:
            raise SkillError("Заголовок навыка должен содержать только однострочные поля.")
        key, value = match.groups()
        if key not in {"name", "description"}:
            raise SkillError("В заголовке навыка разрешены только name и description.")
        if key in values:
            raise SkillError("Поля заголовка навыка не должны повторяться.")
        value = value.strip()
        if not value:
            raise SkillError("Поля заголовка навыка не должны быть пустыми.")
        values[key] = value
    if closing_index is None or set(values) != {"name", "description"}:
        raise SkillError("В навыке нужны поля name, description и закрывающая строка ---.")
    if not _valid_name(values["name"]) or values["name"] != directory_name:
        raise SkillError("Имя навыка должно совпадать с папкой и иметь формат kebab-case до 64 символов.")
    description = _description(values["description"])
    instructions = "".join(lines[closing_index + 1:]).strip()
    if not instructions:
        raise SkillError("Навык должен содержать непустые инструкции после заголовка.")
    return Skill(SkillMetadata(values["name"], description), instructions)


class SkillRegistry:
    """Снимок доверенной библиотеки; изменения файлов требуют нового экземпляра.

    Полная проверка происходит при создании. Список показывает только metadata,
    а load() берёт уже проверенный Skill из словаря, не строя путь из ввода модели.
    """

    def __init__(self, root: Path | None = None) -> None:
        try:
            self.root = Path(os.path.abspath(DEFAULT_SKILLS_DIR if root is None else root))
        except (TypeError, ValueError, OSError):
            raise SkillError("Неверный каталог библиотеки навыков.") from None
        try:
            self._skills = self._discover()
        except SkillError:
            raise
        except OSError:
            raise SkillError("Не удалось безопасно прочитать библиотеку навыков.") from None

    def list_skills(self) -> list[SkillMetadata]:
        return [self._skills[name].metadata for name in sorted(self._skills)]

    def load(self, name: str) -> Skill:
        if not _valid_name(name) or name not in self._skills:
            raise SkillError("Навык не найден в контролируемой библиотеке.")
        return self._skills[name]

    def _discover(self) -> dict[str, Skill]:
        _check_components(self.root)
        if not self.root.is_dir():
            raise SkillError("Библиотека навыков должна быть обычной папкой.")
        directories = []
        with os.scandir(self.root) as entries:
            for entry in entries:
                path = Path(entry.path)
                info = path.lstat()
                attributes = getattr(info, "st_file_attributes", 0)
                if (
                    stat.S_ISLNK(info.st_mode)
                    or attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
                ):
                    raise SkillError("Ссылки в каталоге библиотеки навыков запрещены.")
                if not stat.S_ISDIR(info.st_mode):
                    continue
                if not _valid_name(entry.name):
                    raise SkillError("Папка навыка должна иметь имя kebab-case до 64 символов.")
                directories.append(path)
                if len(directories) > _MAX_SKILLS:
                    raise SkillError("В библиотеке допускается не более 20 навыков.")
        skills = {}
        for directory in sorted(directories, key=lambda path: path.name):
            _check_components(directory)
            if not directory.is_dir():
                raise SkillError("Папка навыка должна быть обычным каталогом.")
            path = directory / "SKILL.md"
            _check_components(path)
            _check_skill_file(path.lstat())
            flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(path, flags)
            with os.fdopen(descriptor, "rb") as source:
                _check_skill_file(os.fstat(source.fileno()))
                data = source.read(_MAX_SKILL_BYTES + 1)
            if len(data) > _MAX_SKILL_BYTES:
                raise SkillError("Файл навыка превышает лимит 32 КиБ.")
            skill = _parse_skill(data, directory.name)
            skills[skill.metadata.name] = skill
        return skills
