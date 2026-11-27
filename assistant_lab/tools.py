"""Ограниченные файловые инструменты учебного ассистента."""

from __future__ import annotations

from io import StringIO
import os
from pathlib import Path, PureWindowsPath
import stat
from typing import Any

from .file_policy import decode_text, hidden_name_allowed, ignored_directory, is_hidden, text_candidate


class _ToolError(Exception):
    """Ошибка, которую можно безопасно вернуть модели."""


class FileTools:
    """Ограничения на уровне приложения; системную песочницу не заменяют."""

    _MAX_ENTRIES = 200
    _MAX_SCANNED = 2000
    _MAX_DEPTH = 20
    _MAX_SOURCE_BYTES = 2 * 1024 * 1024
    _MAX_SEARCH_FILES = 32
    _MAX_SEARCH_BYTES = 4 * 1024 * 1024
    _MAX_MATCHES = 30

    def __init__(
        self,
        project_dir: Path,
        report_path: Path | None,
        max_file_bytes: int = 65536,
        required_source: str | None = None,
        project_mode: bool = False,
        max_total_read_bytes: int = 16384,
    ) -> None:
        if type(max_file_bytes) is not int or max_file_bytes <= 0:
            raise ValueError("Лимит размера файла должен быть положительным целым числом.")
        if type(max_total_read_bytes) is not int or max_total_read_bytes <= 0:
            raise ValueError("Общий лимит чтения должен быть положительным целым числом.")
        if type(project_mode) is not bool:
            raise ValueError("project_mode должен быть логическим значением.")
        # resolve() здесь скрыл бы символическую ссылку в переданном пути.
        self.project_dir = Path(os.path.abspath(project_dir))
        self.report_path = Path(os.path.abspath(report_path)) if report_path is not None else None
        self.max_file_bytes = max_file_bytes
        self.project_mode = project_mode
        self._read_limit = min(max_file_bytes, 65536) if project_mode else max_file_bytes
        self._source_limit = self._MAX_SOURCE_BYTES if project_mode else max_file_bytes
        self._remaining_read_bytes = max_total_read_bytes
        if required_source is not None and not isinstance(required_source, str):
            raise ValueError("Путь обязательного источника должен быть строкой.")
        try:
            self._required_source = (
                self._input_path(required_source) if required_source is not None else None
            )
        except _ToolError as error:
            raise ValueError(str(error)) from error
        self._successful_reads: set[Path] = set()
        self._read_ranges: list[dict] = []
        self._report_saved = False
        self._project_listed = False

    @property
    def report_saved(self) -> bool:
        """Был ли отчёт полностью записан этим экземпляром инструментов."""
        return self._report_saved

    @property
    def read_paths(self) -> list[str]:
        """Имена успешно прочитанных файлов без их содержимого."""
        paths = [path.relative_to(self.project_dir).as_posix() for path in self._successful_reads]
        return sorted(paths, key=lambda path: (path.casefold(), path))

    @property
    def remaining_read_bytes(self) -> int:
        """Остаток общего бюджета возвращённого текста в режиме анализа проекта."""
        return self._remaining_read_bytes

    @property
    def read_ranges(self) -> list[dict]:
        """Фактически возвращённые диапазоны строк; копии не меняют историю."""
        return [item.copy() for item in self._read_ranges]

    @property
    def definitions(self) -> list[dict[str, Any]]:
        definitions = [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": (
                        "Прочитать UTF-8 файл .md или .txt внутри учебного проекта. "
                        "Путь должен быть относительным; ссылки запрещены."
                    ),
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Относительный путь, например README.md.",
                            }
                        },
                        "required": ["path"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "save_report",
                    "description": (
                        "Сохранить отчёт в заранее заданный файл текущего запуска. "
                        "Существующий файл перезаписывать нельзя."
                    ),
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "Полный текст отчёта на русском языке.",
                            }
                        },
                        "required": ["content"],
                        "additionalProperties": False,
                    },
                },
            },
        ]
        if self.project_mode:
            definitions[0]["function"]["description"] = (
                "Прочитать текстовый файл любого языка или его фрагмент. "
                "start_line начинается с 1, max_lines по умолчанию 200 (максимум 500). "
                "Ответ содержит диапазон строк и next_line для продолжения. "
                "Фрагмент ограничен 64 КиБ и оставшимся общим бюджетом; файл — 2 МиБ. "
                "Бинарные, секретные и сборочные пути исключены."
            )
            # Необязательные параметры проверяет execute; strict-схема API
            # потребовала бы сделать все поля обязательными.
            definitions[0]["function"]["strict"] = False
            definitions[0]["function"]["parameters"]["properties"].update({
                "start_line": {"type": "integer", "minimum": 1,
                               "description": "Первая строка фрагмента, по умолчанию 1."},
                "max_lines": {"type": "integer", "minimum": 1, "maximum": 500,
                              "description": "Максимум строк фрагмента, по умолчанию 200."},
            })
            definitions[1]["function"]["description"] += (
                " Список фактически прочитанных фрагментов программа добавит автоматически."
            )
            definitions.append({
                "type": "function",
                "function": {
                    "name": "list_files",
                    "description": (
                        "Посмотреть имена и размеры разрешённых файлов проекта без чтения "
                        "содержимого. path — относительный подкаталог, по умолчанию '.'. "
                        "offset — начало страницы, limit — размер страницы (по умолчанию 100, "
                        "максимум 200). next_offset позволяет получить следующую страницу. "
                        "Не более 2000 просмотренных записей и 20 компонентов пути. "
                        "Перед отчётом вызовите "
                        "этот инструмент и прочитайте хотя бы один файл через read_file."
                    ),
                    "strict": False,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Подкаталог проекта или '.' для корня."},
                            "offset": {"type": "integer", "minimum": 0},
                            "limit": {"type": "integer", "minimum": 1, "maximum": self._MAX_ENTRIES},
                        },
                        "required": [],
                        "additionalProperties": False,
                    },
                },
            })
            definitions.append({
                "type": "function",
                "function": {
                    "name": "search_text",
                    "description": (
                        "Найти точную подстроку с учётом регистра в файле или подкаталоге. "
                        "Возвращает пути и номера строк. Для цитаты прочитайте найденную "
                        "строку через read_file. При truncated=true охват неполный."
                    ),
                    "strict": False,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "minLength": 1, "maxLength": 256,
                                      "description": "Однострочный текст для буквального поиска."},
                            "path": {"type": "string", "description": "Файл, подкаталог или '.' (по умолчанию)."},
                            "limit": {"type": "integer", "minimum": 1, "maximum": self._MAX_MATCHES,
                                      "default": 10},
                        },
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                },
            })
        if self.report_path is None:
            return [item for item in definitions if item["function"]["name"] != "save_report"]
        return definitions

    def execute(self, name: str, arguments: dict) -> dict:
        allowed = {"read_file"}
        if self.report_path is not None:
            allowed.add("save_report")
        if self.project_mode:
            allowed.update(("list_files", "search_text"))
        if not isinstance(name, str) or name not in allowed:
            return {"ok": False, "error": "Неизвестный инструмент."}
        if not isinstance(arguments, dict):
            return {"ok": False, "error": "Неверный набор аргументов инструмента."}
        required = {"save_report": {"content"}, "read_file": {"path"},
                    "search_text": {"query"}, "list_files": set()}[name]
        optional: set[str] = set()
        if self.project_mode:
            optional = {"read_file": {"start_line", "max_lines"},
                        "list_files": {"path", "offset", "limit"}, "search_text": {"path", "limit"},
                        "save_report": set()}[name]
        if not required <= set(arguments) or set(arguments) - required - optional:
            return {"ok": False, "error": "Неверный набор аргументов инструмента."}
        for key in ("path", "content", "query"):
            if key in arguments and not isinstance(arguments[key], str):
                return {"ok": False, "error": "Аргумент инструмента должен быть строкой."}
        max_limit = self._MAX_MATCHES if name == "search_text" else self._MAX_ENTRIES
        for key, minimum, maximum in (("start_line", 1, None), ("max_lines", 1, 500),
                                       ("offset", 0, None), ("limit", 1, max_limit)):
            if key in arguments and (type(arguments[key]) is not int
                                     or arguments[key] < minimum
                                     or maximum is not None and arguments[key] > maximum):
                return {"ok": False, "error": f"Недопустимый числовой аргумент {key}."}

        try:
            if name == "search_text":
                return self._search_text(**arguments)
            if name == "list_files":
                return self._list_files(**arguments)
            if name == "read_file":
                return self._read_file(**arguments)
            return self._save_report(arguments["content"])
        except _ToolError as error:
            return {"ok": False, "error": str(error)}
        except FileExistsError:
            return {"ok": False, "error": "Файл отчёта уже существует; перезапись запрещена."}
        except FileNotFoundError:
            return {"ok": False, "error": "Файл или родительская папка не найдены."}
        except PermissionError:
            return {"ok": False, "error": "Нет доступа к файлу или папке."}
        except UnicodeError:
            return {"ok": False, "error": "Текст должен иметь корректную кодировку UTF-8."}
        except OSError:
            return {"ok": False, "error": "Не удалось выполнить файловую операцию."}

    @staticmethod
    def _check_components(path: Path, *, allow_missing_leaf: bool = False) -> None:
        """Проверить также родителей, не разрешая ссылки и Windows reparse points."""
        components = [*reversed(path.parents), path]
        for component in components:
            try:
                info = component.lstat()
            except FileNotFoundError:
                if allow_missing_leaf and component == path:
                    return
                raise
            attributes = getattr(info, "st_file_attributes", 0)
            reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
            if stat.S_ISLNK(info.st_mode) or attributes & reparse_flag:
                raise _ToolError("Символические ссылки, junction и reparse points запрещены.")
            if component != path and not stat.S_ISDIR(info.st_mode):
                raise _ToolError("Родитель пути должен быть обычной папкой.")

    def _relative_path(self, value: str, *, allow_root: bool = False) -> Path:
        if value == "." and allow_root:
            return self.project_dir
        windows = PureWindowsPath(value)
        if not value or Path(value).is_absolute() or windows.drive or windows.root:
            raise _ToolError("Нужен непустой относительный путь внутри проекта.")
        parts = value.replace("\\", "/").split("/")
        reserved = {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"}
        reserved.update(f"{prefix}{number}" for prefix in ("COM", "LPT") for number in range(1, 10))
        for part in parts:
            if (
                part in {"", ".", ".."}
                # NTFS short-name aliases могут скрывать .env/credentials.
                # Отклоняем '~' также на других ОС ради одинаковых правил.
                or "~" in part
                or part.endswith((" ", "."))
                or any(ord(char) < 32 or char in '<>:"|?*' for char in part)
                or part.split(".", 1)[0].upper() in reserved
            ):
                raise _ToolError("Путь содержит запрещённый компонент.")
        path = self.project_dir.joinpath(*parts)
        if not path.is_relative_to(self.project_dir):
            raise _ToolError("Чтение за пределами проекта запрещено.")
        return path

    def _input_path(self, value: str) -> Path:
        path = self._relative_path(value)
        parts = path.relative_to(self.project_dir).parts
        if self.project_mode:
            if (
                len(parts) > self._MAX_DEPTH
                or any(self._ignored_directory(part) for part in parts[:-1])
                or not self._project_filename(parts[-1])
            ):
                raise _ToolError("Этот файл или путь исключён из анализа проекта.")
        elif path.suffix.lower() not in {".md", ".txt"}:
            raise _ToolError("Разрешено читать только файлы .md и .txt.")
        return path

    @staticmethod
    def _ignored_directory(name: str) -> bool:
        return ignored_directory(name)

    @staticmethod
    def _project_filename(name: str) -> bool:
        return text_candidate(name)

    def _check_project_visibility(self, path: Path) -> None:
        relative = path.relative_to(self.project_dir)
        current = self.project_dir
        for part in relative.parts:
            current = current / part
            info = current.lstat()
            if is_hidden(info) and not hidden_name_allowed(part, stat.S_ISDIR(info.st_mode)):
                raise _ToolError("Скрытые файлы и папки исключены из анализа проекта.")

    def _scan_files(self, path: str) -> tuple[list[dict], int, bool]:
        self._check_components(self.project_dir)
        if not self.project_dir.is_dir():
            raise _ToolError("Путь проекта должен быть обычной папкой.")
        start = self._relative_path(path, allow_root=True)
        parts = start.relative_to(self.project_dir).parts
        if len(parts) >= self._MAX_DEPTH or any(self._ignored_directory(part) for part in parts):
            raise _ToolError("Этот каталог исключён из анализа проекта.")
        self._check_components(start)
        self._check_project_visibility(start)
        if not start.is_dir():
            raise _ToolError("Для просмотра списка нужен каталог.")
        files = []
        scanned = 0
        truncated = False
        pending = [(start, len(parts))]
        while pending and scanned < self._MAX_SCANNED:
            directory, depth = pending.pop()
            try:
                self._check_components(directory)
                self._check_project_visibility(directory)
                with os.scandir(directory) as iterator:
                    entries = []
                    while scanned < self._MAX_SCANNED:
                        try:
                            entry = next(iterator)
                        except StopIteration:
                            break
                        scanned += 1
                        entries.append(entry)
                    else:
                        truncated = True
            except (OSError, _ToolError):
                if directory == start:
                    raise
                truncated = True
                continue
            children = []
            for entry in sorted(entries, key=lambda item: (item.name.casefold(), item.name)):
                try:
                    # На Windows DirEntry.stat() может возвращать st_nlink=0.
                    # lstat() получает реальное число ссылок, не переходя по symlink.
                    info = Path(entry.path).lstat()
                except OSError:
                    truncated = True
                    continue
                attributes = getattr(info, "st_file_attributes", 0)
                is_directory = stat.S_ISDIR(info.st_mode)
                if (
                    stat.S_ISLNK(info.st_mode)
                    or attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
                    or is_hidden(info) and not hidden_name_allowed(entry.name, is_directory)
                ):
                    continue
                if is_directory:
                    if "~" in entry.name or self._ignored_directory(entry.name):
                        continue
                    if depth + 1 >= self._MAX_DEPTH:
                        truncated = True
                        continue
                    children.append((Path(entry.path), depth + 1))
                elif stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and self._project_filename(entry.name):
                    path = Path(entry.path)
                    try:
                        self._input_path(path.relative_to(self.project_dir).as_posix())
                    except _ToolError:
                        continue
                    files.append({
                        "path": path.relative_to(self.project_dir).as_posix(),
                        "bytes": info.st_size,
                        # Формат/кодировка проверяются только при read_file.
                        "readable": info.st_size <= self._source_limit,
                    })
            pending.extend(reversed(children))
        if pending:
            truncated = True
        files.sort(key=lambda item: (item["path"].casefold(), item["path"]))
        return files, scanned, truncated

    def _list_files(self, path: str = ".", offset: int = 0, limit: int = 100) -> dict:
        files, scanned, truncated = self._scan_files(path)
        page = files[offset:offset + limit]
        next_offset = offset + len(page) if offset + len(page) < len(files) else None
        self._project_listed = True
        return {
            "ok": True,
            "files": page,
            "truncated": truncated or next_offset is not None,
            "next_offset": next_offset,
            "scanned": scanned,
            "remaining_read_bytes": self.remaining_read_bytes,
        }

    def _search_text(self, query: str, path: str = ".", limit: int = 10) -> dict:
        if not query.strip() or len(query) > 256 or "\r" in query or "\n" in query:
            raise _ToolError("Для поиска нужна непустая фраза в одну строку, до 256 символов.")
        start = self._relative_path(path, allow_root=True)
        self._check_components(start)
        directory = start.is_dir()
        if directory:
            candidates, _, truncated = self._scan_files(path)
        else:
            target = self._input_path(path)
            candidates = [{"path": target.relative_to(self.project_dir).as_posix(),
                           "bytes": target.lstat().st_size}]
            truncated = False

        matches = []
        searched = skipped = 0
        remaining = self._MAX_SEARCH_BYTES
        for candidate in candidates:
            size = candidate["bytes"]
            if searched + skipped >= self._MAX_SEARCH_FILES:
                truncated = True
                break
            if size > remaining and directory:
                skipped += 1
                continue
            try:
                target = self._input_path(candidate["path"])
                # Reserve the listed size even when decoding fails. If a file grows
                # between listing and opening, the bounded read rejects it.
                if size > self._source_limit:
                    raise _ToolError("Файл превышает допустимый размер.")
                remaining -= size
                content = self._source_text(target, size)
            except (OSError, _ToolError):
                if not directory:
                    raise
                skipped += 1
                continue
            searched += 1
            for line_number, line in enumerate(StringIO(content, newline=""), start=1):
                if query in line:
                    matches.append({"path": candidate["path"], "line": line_number})
                    if len(matches) > limit:
                        break
            if len(matches) > limit:
                truncated = True
                break
        return {"ok": True, "matches": matches[:limit], "files_searched": searched,
                "files_skipped": skipped, "truncated": truncated or skipped > 0}

    def _source_text(self, path: Path, byte_limit: int) -> str:
        self._check_components(self.project_dir)
        if not self.project_dir.is_dir():
            raise _ToolError("Путь проекта должен быть обычной папкой.")
        self._check_components(path)
        if self.project_mode:
            self._check_project_visibility(path)
        info = path.lstat()
        self._check_readable_file(info)
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as source:
            self._check_readable_file(os.fstat(source.fileno()))
            content = source.read(byte_limit + 1)
        if len(content) > byte_limit:
            raise _ToolError("Файл превышает допустимый размер.")
        if self.project_mode:
            try:
                return decode_text(content)
            except ValueError as error:
                raise _ToolError(str(error)) from None
        return content.decode("utf-8")

    def _read_file(self, path: str, start_line: int = 1, max_lines: int = 200) -> dict:
        path = self._input_path(path)
        decoded = self._source_text(path, self._source_limit)
        if self.project_mode:
            decoded, end_line, total_lines = self._fragment(decoded, start_line, max_lines)
            self._remaining_read_bytes -= len(decoded.encode("utf-8"))
        self._successful_reads.add(path)
        result = {
            "ok": True,
            "path": path.relative_to(self.project_dir).as_posix(),
            "content": decoded,
        }
        if self.project_mode:
            self._read_ranges.append({"path": result["path"], "start_line": start_line,
                                      "end_line": end_line})
            next_line = end_line + 1 if end_line < total_lines else None
            result.update({"start_line": start_line, "end_line": end_line,
                           "total_lines": total_lines, "truncated": next_line is not None,
                           "next_line": next_line, "remaining_read_bytes": self.remaining_read_bytes})
        return result

    def _fragment(self, text: str, start_line: int, max_lines: int) -> tuple[str, int, int]:
        # Нумерация соответствует LF/CRLF/CR, а не разрывам страницы или U+2028.
        lines = list(StringIO(text, newline=""))
        if start_line > len(lines) and not (not lines and start_line == 1):
            raise _ToolError("Начальная строка находится за пределами файла.")
        capacity = min(self._read_limit, self.remaining_read_bytes)
        selected = []
        byte_count = 0
        for line in lines[start_line - 1:start_line - 1 + max_lines]:
            size = len(line.encode("utf-8"))
            if byte_count + size > capacity:
                break
            selected.append(line)
            byte_count += size
        if lines and not selected:
            if len(lines[start_line - 1].encode("utf-8")) > self._read_limit:
                raise _ToolError("Строка превышает максимальный размер фрагмента. "
                                 "Выберите другую строку или файл.")
            raise _ToolError("Запрошенная строка не помещается в лимит фрагмента или остаток "
                             "бюджета чтения. Завершите обзор либо увеличьте бюджет.")
        return "".join(selected), start_line + len(selected) - 1, len(lines)

    def _check_readable_file(self, info: os.stat_result) -> None:
        if not stat.S_ISREG(info.st_mode):
            raise _ToolError("Разрешено читать только обычные файлы.")
        if info.st_nlink != 1:
            raise _ToolError("Чтение файлов с жёсткими ссылками запрещено.")
        if info.st_size > self._source_limit:
            raise _ToolError("Файл превышает допустимый размер.")

    def _save_report(self, content: str) -> dict:
        if not content.strip():
            raise _ToolError("Отчёт не должен быть пустым или состоять только из пробелов.")
        if self.project_mode and (not self._project_listed or not self._successful_reads):
            raise _ToolError("Перед отчётом вызовите list_files и успешно прочитайте файл через read_file.")
        if self._required_source is not None and self._required_source not in self._successful_reads:
            raise _ToolError("Сначала успешно прочитайте заданный исходный файл через read_file.")
        if self.project_mode:
            evidence = []
            for fragment in self._read_ranges:
                path = fragment["path"]
                # Подбираем разделитель, если имя файла само содержит обратную кавычку.
                delimiter = "`"
                while delimiter in path:
                    delimiter += "`"
                position = (f"строки {fragment['start_line']}–{fragment['end_line']}"
                            if fragment["end_line"] else "пустой файл")
                evidence.append(f"- {delimiter}{path}{delimiter}: {position}")
            content += (
                "\n\n---\nФактически прочитанные фрагменты (отметка программы):\n"
                + "\n".join(evidence)
                + "\n"
            )
        encoded = content.encode("utf-8")
        if len(encoded) > self.max_file_bytes:
            raise _ToolError("Отчёт превышает допустимый размер.")
        self._check_components(self.report_path, allow_missing_leaf=True)
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_BINARY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(self.report_path, flags, 0o600)
        with os.fdopen(descriptor, "wb") as report:
            report.write(encoded)
        self._report_saved = True
        return {"ok": True, "path": str(self.report_path), "bytes": len(encoded)}
