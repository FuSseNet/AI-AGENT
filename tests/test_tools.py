"""Проверки границ файловых инструментов без сети и внешних зависимостей."""

import os
from pathlib import Path
import stat
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from assistant_lab.tools import FileTools


class FileToolsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.project = self.root / "project"
        self.project.mkdir()
        self.output = self.root / "run"
        self.output.mkdir()
        self.report = self.output / "report.md"
        self.tools = FileTools(self.project, self.report, max_file_bytes=64)
        (self.project / "README.md").write_text("Учебный проект", encoding="utf-8")

    def assert_rejected(self, name, arguments):
        result = self.tools.execute(name, arguments)
        self.assertIs(result["ok"], False, result)
        self.assertTrue(result.get("error"), result)
        return result

    def test_read_and_save_success(self):
        nested = self.project / "docs"
        nested.mkdir()
        (nested / "note.txt").write_text("Привет!", encoding="utf-8")
        result = self.tools.execute("read_file", {"path": "docs\\note.txt"})
        self.assertEqual(result, {"ok": True, "path": "docs/note.txt", "content": "Привет!"})
        saved = self.tools.execute("save_report", {"content": "Готово"})
        self.assertEqual(saved, {"ok": True, "path": str(self.report), "bytes": 12})
        self.assertEqual(self.report.read_text(encoding="utf-8"), "Готово")

    def test_reject_traversal_absolute_ads_and_devices(self):
        (self.root / "secret.txt").write_text("секрет", encoding="utf-8")
        paths = [
            "../secret.txt", "..\\secret.txt", "docs/../../secret.txt",
            "docs\\..\\README.md", "docs/..\\README.md", "/secret.txt",
            "\\secret.txt", "C:\\secret.txt", "C:secret.txt", "\\\\server\\share\\secret.txt",
            "\\\\?\\C:\\secret.txt", "README.md:secret.txt", "README.md::$DATA",
            "docs//note.txt", "./README.md", "README.md.", "README.md ",
            "NUL.txt", "con.md", "COM1.txt", "a\x00.txt", "",
        ]
        for value in paths:
            with self.subTest(path=value):
                self.assert_rejected("read_file", {"path": value})

    def test_reject_unknown_extra_missing_and_wrong_types(self):
        cases = [
            ("run_shell", {"command": "echo test"}),
            ("read_file", {}),
            ("read_file", {"path": "README.md", "extra": True}),
            ("read_file", {"path": 123}),
            ("read_file", ["README.md"]),
            ("save_report", {"content": "x", "path": "elsewhere.md"}),
            ("save_report", {"content": None}),
        ]
        for name, arguments in cases:
            with self.subTest(name=name, arguments=arguments):
                self.assert_rejected(name, arguments)
        self.assertFalse(self.report.exists())

    def test_reject_binary_missing_directory_and_invalid_utf8(self):
        (self.project / "code.py").write_text("print(1)", encoding="utf-8")
        (self.project / "broken.txt").write_bytes(b"\xff\xfe")
        (self.project / "folder.md").mkdir()
        for name in ("code.py", "missing.txt", "broken.txt", "folder.md"):
            with self.subTest(path=name):
                self.assert_rejected("read_file", {"path": name})

    def test_limit_counts_utf8_bytes_and_accepts_boundary(self):
        (self.project / "boundary.txt").write_bytes(b"a" * 64)
        self.assertTrue(self.tools.execute("read_file", {"path": "boundary.txt"})["ok"])
        (self.project / "large.txt").write_bytes(b"a" * 65)
        self.assert_rejected("read_file", {"path": "large.txt"})
        self.assert_rejected("save_report", {"content": "я" * 33})
        self.assertFalse(self.report.exists())
        self.assertTrue(self.tools.execute("save_report", {"content": "я" * 32})["ok"])
        self.assertEqual(self.report.stat().st_size, 64)

    def test_read_is_bounded_even_if_file_grows_after_size_check(self):
        original_check = self.tools._check_readable_file
        checks = 0

        def grow_after_check(info):
            nonlocal checks
            original_check(info)
            checks += 1
            if checks == 2:
                with (self.project / "README.md").open("ab") as growing:
                    growing.write(b"a" * 100)

        with patch.object(self.tools, "_check_readable_file", side_effect=grow_after_check):
            self.assert_rejected("read_file", {"path": "README.md"})

    def test_report_cannot_overwrite_existing_file(self):
        self.report.write_text("старый отчёт", encoding="utf-8")
        self.assert_rejected("save_report", {"content": "новый отчёт"})
        self.assertEqual(self.report.read_text(encoding="utf-8"), "старый отчёт")
        self.assertFalse(self.tools.report_saved)

    def test_required_source_must_be_successfully_read_before_save(self):
        tools = FileTools(self.project, self.report, required_source="README.md")
        self.assertFalse(tools.execute("save_report", {"content": "Отчёт"})["ok"])
        self.assertFalse(tools.report_saved)
        self.assertFalse(self.report.exists())
        self.assertTrue(tools.execute("read_file", {"path": "README.md"})["ok"])
        self.assertFalse(tools.report_saved)
        self.assertTrue(tools.execute("save_report", {"content": "Отчёт"})["ok"])
        self.assertTrue(tools.report_saved)

    def test_reading_other_source_does_not_allow_save(self):
        (self.project / "other.txt").write_text("Другой источник", encoding="utf-8")
        tools = FileTools(self.project, self.report, required_source="README.md")
        self.assertTrue(tools.execute("read_file", {"path": "other.txt"})["ok"])
        self.assertFalse(tools.execute("save_report", {"content": "Отчёт"})["ok"])
        self.assertFalse(tools.report_saved)
        self.assertFalse(self.report.exists())

    def test_failed_source_read_does_not_allow_save(self):
        tools = FileTools(self.project, self.report, required_source="README.md")
        (self.project / "README.md").write_bytes(b"\xff\xfe")
        self.assertFalse(tools.execute("read_file", {"path": "README.md"})["ok"])
        (self.project / "README.md").write_text("Исправленный источник", encoding="utf-8")
        self.assertFalse(tools.execute("save_report", {"content": "Отчёт"})["ok"])
        self.assertFalse(tools.report_saved)
        self.assertTrue(tools.execute("read_file", {"path": "README.md"})["ok"])
        self.assertTrue(tools.execute("save_report", {"content": "Отчёт"})["ok"])

    def test_required_source_normalizes_windows_separators(self):
        (self.project / "docs").mkdir()
        (self.project / "docs" / "source.txt").write_text("Источник", encoding="utf-8")
        tools = FileTools(self.project, self.report, required_source="docs\\source.txt")
        self.assertTrue(tools.execute("read_file", {"path": "docs/source.txt"})["ok"])
        self.assertTrue(tools.execute("save_report", {"content": "Отчёт"})["ok"])

    def test_required_source_uses_input_path_validation(self):
        for value in ("../secret.txt", "C:\\secret.txt", "README.md:stream.txt", "code.py"):
            with self.subTest(required_source=value), self.assertRaises(ValueError):
                FileTools(self.project, self.report, required_source=value)

    def test_report_saved_changes_only_after_successful_write(self):
        self.assertFalse(self.tools.report_saved)
        with patch("assistant_lab.tools.os.open", side_effect=PermissionError):
            self.assert_rejected("save_report", {"content": "Отчёт"})
        self.assertFalse(self.tools.report_saved)
        self.assertFalse(self.report.exists())
        self.assertTrue(self.tools.execute("save_report", {"content": "Отчёт"})["ok"])
        self.assertTrue(self.tools.report_saved)

    def test_partial_write_does_not_mark_report_saved(self):
        original_fdopen = os.fdopen

        class FailingReport:
            def __init__(self, descriptor, mode):
                self.file = original_fdopen(descriptor, mode)

            def __enter__(self):
                return self

            def write(self, content):
                self.file.write(content[:1])
                raise OSError("Тестовая ошибка записи.")

            def __exit__(self, *exception):
                self.file.close()

        with patch("assistant_lab.tools.os.fdopen", side_effect=FailingReport):
            self.assert_rejected("save_report", {"content": "Отчёт"})
        self.assertTrue(self.report.exists())
        self.assertFalse(self.tools.report_saved)

    def test_empty_and_whitespace_reports_are_rejected(self):
        for content in ("", " ", "\t\r\n", "\u00a0\u2003"):
            with self.subTest(content=repr(content)):
                self.assert_rejected("save_report", {"content": content})
                self.assertFalse(self.tools.report_saved)
                self.assertFalse(self.report.exists())

    def test_hardlinked_input_is_rejected(self):
        source = self.root / "outside.txt"
        source.write_text("секрет", encoding="utf-8")
        try:
            os.link(source, self.project / "linked.txt")
        except (OSError, NotImplementedError) as error:
            self.skipTest(f"Создание жёсткой ссылки недоступно: {error}")
        self.assert_rejected("read_file", {"path": "linked.txt"})

    def create_symlink(self, source, target, *, is_directory=False):
        try:
            target.symlink_to(source, target_is_directory=is_directory)
        except (OSError, NotImplementedError) as error:
            self.skipTest(f"Создание символической ссылки недоступно: {error}")

    def test_symlink_input_and_parent_are_rejected(self):
        source = self.root / "outside.txt"
        source.write_text("секрет", encoding="utf-8")
        self.create_symlink(source, self.project / "linked.txt")
        self.assert_rejected("read_file", {"path": "linked.txt"})
        self.create_symlink(self.root, self.project / "linked_folder", is_directory=True)
        self.assert_rejected("read_file", {"path": "linked_folder/outside.txt"})

    def test_symlink_project_root_is_rejected(self):
        linked_project = self.root / "linked_project"
        self.create_symlink(self.project, linked_project, is_directory=True)
        tools = FileTools(linked_project, self.report)
        self.assertFalse(tools.execute("read_file", {"path": "README.md"})["ok"])

    def test_symlink_report_file_is_rejected(self):
        source = self.root / "outside.txt"
        source.write_text("не менять", encoding="utf-8")
        self.create_symlink(source, self.report)
        self.assert_rejected("save_report", {"content": "новое"})
        self.assertEqual(source.read_text(encoding="utf-8"), "не менять")

    def test_symlink_output_parent_is_rejected(self):
        linked_output = self.root / "linked_output"
        self.create_symlink(self.output, linked_output, is_directory=True)
        tools = FileTools(self.project, linked_output / "report.md")
        self.assertFalse(tools.execute("save_report", {"content": "новое"})["ok"])
        self.assertFalse(self.report.exists())

    def test_reparse_attribute_is_rejected(self):
        class ReparseInfo:
            st_mode = 0o040755
            st_file_attributes = 0x400

        with patch.object(Path, "lstat", return_value=ReparseInfo()):
            self.assert_rejected("read_file", {"path": "README.md"})
            self.assert_rejected("save_report", {"content": "новое"})

    def test_tool_schemas_have_exact_required_arguments(self):
        definitions = {item["function"]["name"]: item["function"] for item in self.tools.definitions}
        self.assertEqual(set(definitions), {"read_file", "save_report"})
        for name, argument in (("read_file", "path"), ("save_report", "content")):
            parameters = definitions[name]["parameters"]
            self.assertEqual(parameters["required"], [argument])
            self.assertEqual(set(parameters["properties"]), {argument})
            self.assertIs(parameters["additionalProperties"], False)


class ProjectModeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.project = self.root / "project"
        self.project.mkdir()
        self.output = self.root / "run"
        self.output.mkdir()
        self.report = self.output / "report.md"
        self.tools = FileTools(self.project, self.report, project_mode=True)

    def write(self, relative, content="// Пример\n"):
        target = self.project.joinpath(*relative.split("/"))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return target

    def test_project_schema_and_optional_argument_validation(self):
        definitions = {item["function"]["name"]: item["function"] for item in self.tools.definitions}
        self.assertEqual(set(definitions), {"read_file", "save_report", "list_files"})
        parameters = definitions["list_files"]["parameters"]
        self.assertEqual(set(parameters["properties"]), {"path", "offset", "limit"})
        self.assertEqual(parameters["required"], [])
        self.assertIs(parameters["additionalProperties"], False)
        self.assertFalse(self.tools.execute("list_files", {"unexpected": "."})["ok"])
        default_tools = FileTools(self.project, self.report)
        self.assertFalse(default_tools.execute("list_files", {})["ok"])

    def test_listing_is_sorted_excludes_paths_and_does_not_read_contents(self):
        allowed = ["Z.cpp", "a.hpp", "docs/README.md", "CMakeLists.txt", "Makefile",
                   "script.py", "secret.cpp", "Tokens.hpp"]
        excluded = [
            ".env", ".env.txt", ".hidden.cpp",
            "passwords.txt", "credentials.md", "private-key.txt", "id_rsa.txt",
            "certs.txt", "certificate.md", "cert.txt", "key.pem",
            ".GIT/a.cpp", ".idea/a.cpp", ".venv/a.cpp", ".hidden/a.cpp",
            "node_modules/a.cpp", "BUILD/a.cpp", "dist/a.cpp", "target/a.cpp",
            "__pycache__/a.cpp", "CMakeFiles/a.cpp", "CMAKE-BUILD-debug/a.cpp",
            "secrets/a.cpp", "certs/a.cpp",
        ]
        for path in allowed + excluded:
            self.write(path)
        with patch("assistant_lab.tools.os.open", side_effect=AssertionError("Содержимое открыто")):
            result = self.tools.execute("list_files", {})
        self.assertTrue(result["ok"])
        self.assertEqual(
            [item["path"] for item in result["files"]],
            sorted(allowed, key=lambda path: (path.casefold(), path)),
        )
        self.assertFalse(result["truncated"])
        self.assertEqual(result["remaining_read_bytes"], 16384)
        for item in result["files"]:
            self.assertIsInstance(item["bytes"], int)
            self.assertIs(item["readable"], True)
        for path in excluded:
            with self.subTest(path=path):
                self.assertFalse(self.tools.execute("read_file", {"path": path})["ok"])

    def test_source_extensions_and_build_files_can_be_read(self):
        names = [
            "main.c", "main.cc", "main.cpp", "main.cxx", "main.h", "main.hh",
            "main.hpp", "main.hxx", "setup.cmake", "CMakeLists.txt", "Makefile",
            "README.md", "notes.txt",
        ]
        for name in names:
            self.write(name, "Пример")
            result = self.tools.execute("read_file", {"path": name})
            self.assertTrue(result["ok"], result)
            self.assertEqual(result["content"], "Пример")
        self.assertEqual(self.tools.read_paths, sorted(names, key=lambda name: (name.casefold(), name)))
        copy = self.tools.read_paths
        copy.append("непрочитанный.cpp")
        self.assertNotIn("непрочитанный.cpp", self.tools.read_paths)

    def test_listing_caps_results_at_two_hundred(self):
        for number in reversed(range(205)):
            self.write(f"file_{number:03}.cpp", "x")
        result = self.tools.execute("list_files", {"limit": 200})
        self.assertTrue(result["ok"])
        self.assertEqual(len(result["files"]), 200)
        self.assertTrue(result["truncated"])
        self.assertEqual(result["scanned"], 205)
        self.assertEqual(result["files"][0]["path"], "file_000.cpp")
        self.assertEqual(result["files"][-1]["path"], "file_199.cpp")
        self.assertEqual(result["next_offset"], 200)

    def test_listing_uses_real_link_count_when_direntry_cache_reports_zero(self):
        self.write("main.cpp", "int main() {}")
        original_scandir = os.scandir

        class CachedEntry:
            def __init__(self, entry):
                self.name = entry.name
                self.path = entry.path

            def stat(self, **kwargs):
                return SimpleNamespace(
                    st_mode=stat.S_IFREG, st_nlink=0, st_size=13, st_file_attributes=32,
                )

        class CachedScandir:
            def __init__(self, path):
                self.iterator = original_scandir(path)

            def __enter__(self):
                return (CachedEntry(entry) for entry in self.iterator)

            def __exit__(self, *exception):
                self.iterator.close()

        with patch("assistant_lab.tools.os.scandir", side_effect=CachedScandir):
            result = self.tools.execute("list_files", {})
        self.assertEqual(result["files"], [{"path": "main.cpp", "bytes": 13, "readable": True}])

    def test_scan_budget_stops_even_with_endless_excluded_entries(self):
        project = self.project

        class EndlessEntries:
            emitted = 0

            def __enter__(self):
                return self

            def __exit__(self, *exception):
                return False

            def __iter__(self):
                return self

            def __next__(self):
                self.emitted += 1
                return SimpleNamespace(
                    name="ignored.bin", path=str(project / "ignored.bin"),
                    stat=lambda **kwargs: SimpleNamespace(
                        st_mode=stat.S_IFREG, st_nlink=1, st_size=1, st_file_attributes=0,
                    ),
                )

        iterator = EndlessEntries()
        with patch("assistant_lab.tools.os.scandir", return_value=iterator):
            result = self.tools.execute("list_files", {})
        self.assertTrue(result["ok"])
        self.assertEqual(iterator.emitted, 2000)
        self.assertEqual(result["scanned"], 2000)
        self.assertEqual(result["files"], [])
        self.assertTrue(result["truncated"])

    def test_depth_limit_and_ignored_directory_are_not_traversed(self):
        allowed = "/".join(["a"] * 19 + ["main.cpp"])
        deep = "/".join(["a"] * 20 + ["main.cpp"])
        self.write(allowed)
        self.write(deep)
        self.write("build/generated.cpp")
        original_scandir = os.scandir
        visited = []

        def record_scandir(path):
            visited.append(Path(path))
            return original_scandir(path)

        with patch("assistant_lab.tools.os.scandir", side_effect=record_scandir):
            result = self.tools.execute("list_files", {})
        self.assertEqual([item["path"] for item in result["files"]], [allowed])
        self.assertTrue(result["truncated"])
        self.assertNotIn(self.project / "build", visited)
        self.assertNotIn(self.project.joinpath(*(["a"] * 20)), visited)
        self.assertFalse(self.tools.execute("read_file", {"path": deep})["ok"])

    def test_windows_hidden_attribute_prevents_listing_and_reading(self):
        self.write("main.cpp")
        with patch("assistant_lab.tools.is_hidden", return_value=True):
            result = self.tools.execute("list_files", {})
            self.assertEqual(result["files"], [])
            self.assertFalse(self.tools.execute("read_file", {"path": "main.cpp"})["ok"])
        self.assertEqual(self.tools.read_paths, [])

    def test_reparse_directory_is_not_traversed_or_read_directly(self):
        self.write("linked/main.cpp")
        original_lstat = Path.lstat

        def reparse_lstat(path):
            if path == self.project / "linked":
                return SimpleNamespace(st_mode=stat.S_IFDIR, st_file_attributes=0x400)
            return original_lstat(path)

        with patch.object(Path, "lstat", autospec=True, side_effect=reparse_lstat):
            result = self.tools.execute("list_files", {})
            self.assertEqual(result["files"], [])
            self.assertFalse(self.tools.execute("read_file", {"path": "linked/main.cpp"})["ok"])

    def test_hardlinked_file_is_not_listed_or_read(self):
        original = self.root / "outside.cpp"
        original.write_text("int main() {}", encoding="utf-8")
        try:
            os.link(original, self.project / "linked.cpp")
        except (OSError, NotImplementedError) as error:
            self.skipTest(f"Создание жёсткой ссылки недоступно: {error}")
        result = self.tools.execute("list_files", {})
        self.assertEqual(result["files"], [])
        self.assertFalse(self.tools.execute("read_file", {"path": "linked.cpp"})["ok"])

    def test_project_report_requires_both_listing_and_successful_reading(self):
        self.write("main.cpp")
        self.assertFalse(self.tools.execute("save_report", {"content": "Обзор"})["ok"])
        self.assertTrue(self.tools.execute("list_files", {})["ok"])
        self.assertFalse(self.tools.execute("save_report", {"content": "Обзор"})["ok"])
        read_only = FileTools(self.project, self.report, project_mode=True)
        self.assertTrue(read_only.execute("read_file", {"path": "main.cpp"})["ok"])
        self.assertFalse(read_only.execute("save_report", {"content": "Обзор"})["ok"])
        self.assertTrue(self.tools.execute("read_file", {"path": "main.cpp"})["ok"])
        self.assertTrue(self.tools.execute("save_report", {"content": "Обзор"})["ok"])
        self.assertTrue(self.tools.report_saved)
        saved = self.report.read_text(encoding="utf-8")
        self.assertIn("Фактически прочитанные фрагменты (отметка программы):", saved)
        self.assertIn("- `main.cpp`", saved)

    def test_failed_listing_or_read_does_not_satisfy_report_guard(self):
        self.write("main.cpp")
        with patch("assistant_lab.tools.os.scandir", side_effect=PermissionError):
            self.assertFalse(self.tools.execute("list_files", {})["ok"])
        self.assertTrue(self.tools.execute("read_file", {"path": "main.cpp"})["ok"])
        self.assertFalse(self.tools.execute("save_report", {"content": "Обзор"})["ok"])
        second = FileTools(self.project, self.report, project_mode=True)
        self.assertTrue(second.execute("list_files", {})["ok"])
        self.assertFalse(second.execute("read_file", {"path": "missing.cpp"})["ok"])
        self.assertFalse(second.execute("save_report", {"content": "Обзор"})["ok"])
        self.assertEqual(second.read_paths, [])

    def test_total_budget_counts_repeated_reads_and_denies_receipt_for_rejected_file(self):
        self.write("a.cpp", "a" * 10)
        self.write("b.cpp", "b" * 6)
        self.write("small.cpp", "x" * 5)
        tools = FileTools(self.project, self.report, project_mode=True, max_total_read_bytes=15)
        self.assertEqual(tools.execute("list_files", {})["remaining_read_bytes"], 15)
        first = tools.execute("read_file", {"path": "a.cpp"})
        self.assertTrue(first["ok"])
        self.assertEqual(first["remaining_read_bytes"], 5)
        self.assertFalse(tools.execute("read_file", {"path": "a.cpp"})["ok"])
        self.assertFalse(tools.execute("read_file", {"path": "b.cpp"})["ok"])
        self.assertEqual(tools.read_paths, ["a.cpp"])
        self.assertEqual(tools.remaining_read_bytes, 5)
        self.assertTrue(tools.execute("read_file", {"path": "small.cpp"})["ok"])
        self.assertEqual(tools.remaining_read_bytes, 0)
        self.assertEqual(tools.execute("list_files", {})["remaining_read_bytes"], 0)
        self.assertFalse(tools.execute("read_file", {"path": "small.cpp"})["ok"])

    def test_successful_repeated_reads_consume_budget_again(self):
        self.write("a.cpp", "abc")
        tools = FileTools(self.project, self.report, project_mode=True, max_total_read_bytes=9)
        for remaining in (6, 3, 0):
            result = tools.execute("read_file", {"path": "a.cpp"})
            self.assertTrue(result["ok"])
            self.assertEqual(result["remaining_read_bytes"], remaining)
        self.assertEqual(tools.read_paths, ["a.cpp"])

    def test_budget_counts_utf8_bytes_and_failed_decoding_does_not_consume_it(self):
        self.write("a.cpp", "яяя")
        (self.project / "broken.cpp").write_bytes(b"\xff")
        tools = FileTools(self.project, self.report, project_mode=True, max_total_read_bytes=6)
        self.assertFalse(tools.execute("read_file", {"path": "broken.cpp"})["ok"])
        self.assertEqual(tools.remaining_read_bytes, 6)
        self.assertEqual(tools.read_paths, [])
        self.assertTrue(tools.execute("read_file", {"path": "a.cpp"})["ok"])
        self.assertEqual(tools.remaining_read_bytes, 0)

    def test_size_limit_includes_automatic_evidence(self):
        self.write("main.cpp", "x")
        tools = FileTools(self.project, self.report, max_file_bytes=64, project_mode=True)
        self.assertTrue(tools.execute("list_files", {})["ok"])
        self.assertTrue(tools.execute("read_file", {"path": "main.cpp"})["ok"])
        self.assertFalse(tools.execute("save_report", {"content": "Обзор"})["ok"])
        self.assertFalse(tools.report_saved)
        self.assertFalse(self.report.exists())

    def test_returned_line_never_exceeds_sixty_four_kib(self):
        (self.project / "large.cpp").write_bytes(b"x" * 65537)
        tools = FileTools(self.project, self.report, max_file_bytes=131072, project_mode=True)
        result = tools.execute("list_files", {})
        self.assertEqual(result["files"], [{"path": "large.cpp", "bytes": 65537, "readable": True}])
        self.assertFalse(tools.execute("read_file", {"path": "large.cpp"})["ok"])
        self.assertEqual(tools.remaining_read_bytes, 16384)
        self.assertEqual(tools.read_paths, [])

    def test_default_mode_keeps_its_two_tools_and_has_no_aggregate_budget(self):
        self.write("notes.txt", "a" * 20)
        tools = FileTools(self.project, self.report, max_total_read_bytes=1)
        result = tools.execute("read_file", {"path": "notes.txt"})
        self.assertTrue(result["ok"])
        self.assertNotIn("remaining_read_bytes", result)
        self.assertTrue(tools.execute("save_report", {"content": "Готово"})["ok"])
        self.assertEqual(self.report.read_text(encoding="utf-8"), "Готово")


if __name__ == "__main__":
    unittest.main()
