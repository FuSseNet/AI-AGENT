"""Integration checks for bounded, language-independent project inspection.

All fixtures are temporary local files. No models, servers or real secrets are
used, and no source from the user's actual project is executed.
"""

import os
from pathlib import Path
import stat
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from assistant_lab.tools import FileTools


class UniversalProjectToolsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.project = self.root / "project"
        self.project.mkdir()
        self.output = self.root / "output"
        self.output.mkdir()
        self.report = self.output / "report.md"
        self.tools = FileTools(
            self.project, self.report, project_mode=True,
            max_total_read_bytes=16384,
        )

    def write(self, relative, text="example\n", *, encoding="utf-8"):
        target = self.project.joinpath(*relative.split("/"))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(text.encode(encoding))
        return target

    def assert_ok(self, name, arguments):
        result = self.tools.execute(name, arguments)
        self.assertIs(result.get("ok"), True, result)
        return result

    def assert_rejected(self, name, arguments):
        result = self.tools.execute(name, arguments)
        self.assertIs(result.get("ok"), False, result)
        self.assertTrue(result.get("error"), result)
        return result

    @staticmethod
    def paths(listing):
        return [entry["path"] for entry in listing["files"]]

    def test_source_languages_and_build_manifests_are_not_extension_whitelisted(self):
        sources = {
            "pom.xml": "<project><artifactId>example</artifactId></project>\n",
            "build.gradle": "plugins { id 'java' }\n",
            "settings.gradle.kts": 'rootProject.name = "example"\n',
            "src/main/java/com/example/school/App.java": "public class App {}\n",
            "src/main/kotlin/App.kt": 'fun main() = println("hello")\n',
            "app.py": 'print("hello")\n',
            "package.json": '{"name":"example","private":true}\n',
            "src/app.ts": 'export const greeting: string = "hello";\n',
            "app.js": 'console.log("hello");\n',
            "go.mod": "module example\n\ngo 1.22\n",
            "main.go": "package main\nfunc main() {}\n",
            "Cargo.toml": '[package]\nname = "example"\nversion = "0.1.0"\n',
            "src/main.rs": "fn main() {}\n",
            "Example.csproj": '<Project Sdk="Microsoft.NET.Sdk" />\n',
            "Program.cs": 'Console.WriteLine("hello");\n',
            "Makefile": "all:\n\t@echo example\n",
            "source.customlang": "a source language unknown to this tool\n",
            "extensionless": "plain text without an extension\n",
        }
        for name, text in sources.items():
            self.write(name, text)
        listing = self.assert_ok("list_files", {})
        self.assertEqual(set(self.paths(listing)), set(sources))
        for name, text in sources.items():
            with self.subTest(path=name):
                result = self.assert_ok("read_file", {"path": name})
                self.assertEqual(result["path"], name)
                self.assertEqual(result["content"], text)
        self.assertEqual(set(self.tools.read_paths), set(sources))

    def test_pagination_is_stable_complete_and_has_no_duplicate_paths(self):
        expected = [f"source_{number:03}.py" for number in range(235)]
        for name in reversed(expected):
            self.write(name, "x\n")
        first = self.assert_ok("list_files", {"offset": 0, "limit": 80})
        repeated = self.assert_ok("list_files", {"offset": 0, "limit": 80})
        self.assertEqual(self.paths(first), self.paths(repeated))
        self.assertEqual(first["next_offset"], 80)
        self.assertIs(first["truncated"], True)
        found = []
        page = first
        while True:
            found.extend(self.paths(page))
            self.assertLessEqual(page["scanned"], 2000)
            next_offset = page["next_offset"]
            if next_offset is None:
                self.assertIs(page["truncated"], False)
                break
            self.assertIs(type(next_offset), int)
            self.assertGreater(next_offset, len(found) - len(page["files"]))
            page = self.assert_ok("list_files", {"offset": next_offset, "limit": 80})
        self.assertEqual(found, expected)
        self.assertEqual(len(found), len(set(found)))
        self.assertEqual(self.tools.remaining_read_bytes, 16384)

    def test_subtree_paths_stay_relative_to_project_root(self):
        for name in ("root.py", "src/main.py", "src/nested/helper.ts", "tests/test_app.py"):
            self.write(name)
        root = self.assert_ok("list_files", {"path": "."})
        subtree = self.assert_ok("list_files", {"path": "src", "offset": 0, "limit": 100})
        self.assertEqual(set(self.paths(subtree)), {"src/main.py", "src/nested/helper.ts"})
        self.assertIn("root.py", self.paths(root))
        self.assertIs(subtree["truncated"], False)
        self.assertIsNone(subtree["next_offset"])

    def test_invalid_listing_options_are_rejected(self):
        self.write("main.py")
        invalid = [
            {"offset": -1}, {"offset": True}, {"offset": "0"},
            {"limit": 0}, {"limit": 201}, {"limit": True}, {"limit": "10"},
            {"path": 1}, {"unknown": 1},
        ]
        for arguments in invalid:
            with self.subTest(arguments=arguments):
                self.assert_rejected("list_files", arguments)

    def test_reading_and_listing_reject_outside_paths(self):
        (self.root / "outside.py").write_text("private fixture\n", encoding="utf-8")
        self.write("src/main.py")
        outside_paths = ["../outside.py", str(self.root), "src/../../", "C:\\outside"]
        for path in outside_paths:
            with self.subTest(path=path):
                self.assert_rejected("list_files", {"path": path})
                self.assert_rejected("read_file", {"path": path})
        self.assertEqual(self.tools.read_paths, [])

    def test_literal_tilde_components_are_excluded_from_reads_and_listing(self):
        self.write("safe.py", "safe source\n")
        denied = ("ENV~1", "copy~backup.py", "src~copy/main.py")
        for name in denied:
            self.write(name, "synthetic excluded fixture\n")
        listing = self.assert_ok("list_files", {})
        self.assertEqual(self.paths(listing), ["safe.py"])
        for name in denied:
            with self.subTest(path=name):
                self.assert_rejected("read_file", {"path": name})
        self.assert_rejected("list_files", {"path": "src~copy"})
        self.assertEqual(self.tools.read_paths, [])
        self.assertEqual(self.tools.remaining_read_bytes, 16384)

    @unittest.skipUnless(os.name == "nt", "Windows short-path aliases are Windows-specific")
    def test_windows_short_aliases_cannot_bypass_secret_path_policy(self):
        import ctypes
        from ctypes import wintypes

        try:
            get_short_path = ctypes.WinDLL("kernel32", use_last_error=True).GetShortPathNameW
        except (AttributeError, OSError) as error:
            self.skipTest(f"GetShortPathNameW is unavailable: {error}")
        get_short_path.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
        get_short_path.restype = wintypes.DWORD

        def short_path(path):
            required = get_short_path(str(path), None, 0)
            if not required:
                return None
            buffer = ctypes.create_unicode_buffer(required)
            length = get_short_path(str(path), buffer, required)
            if not length or length >= required:
                return None
            return Path(buffer.value)

        synthetic_names = (".env", ".env.production", "credentials.json", "credentials/config.json")
        for name in synthetic_names:
            self.write(name, "synthetic secret fixture, never a real credential\n")
        short_root = short_path(self.project)
        if short_root is None:
            self.skipTest("The temporary volume does not expose Windows short paths")
        aliases = []
        for name in synthetic_names:
            alias_path = short_path(self.project.joinpath(*name.split("/")))
            if alias_path is None:
                continue
            try:
                relative = alias_path.relative_to(short_root).as_posix()
            except ValueError:
                continue
            if "~" in relative:
                aliases.append(relative)
        if not aliases:
            self.skipTest("8.3 aliases are not enabled for these temporary fixtures")
        for alias in aliases:
            with self.subTest(alias=alias):
                self.assert_rejected("read_file", {"path": alias})
                directory = Path(alias).parent.as_posix()
                if "~" in directory:
                    self.assert_rejected("list_files", {"path": directory})
        listing = self.assert_ok("list_files", {})
        self.assertEqual(self.paths(listing), [])
        self.assertEqual(self.tools.read_paths, [])
        self.assertEqual(self.tools.remaining_read_bytes, 16384)

    def test_known_binaries_dependencies_and_secrets_are_excluded_consistently(self):
        self.write("src/main.py", "print('safe')\n")
        blocked = [
            "image.png", "archive.zip", "program.exe", "library.dll",
            ".env", ".env.local", "credentials.json", "private-key.txt", "id_rsa",
            ".git/config", ".idea/workspace.xml", ".venv/lib/module.py",
            "node_modules/dependency/index.js", "build/generated.py",
            "dist/app.js", "target/generated.rs", "__pycache__/module.py",
            "cmake-build-debug/generated.cpp",
        ]
        for name in blocked:
            self.write(name, "synthetic blocked fixture\n")
        listing = self.assert_ok("list_files", {})
        self.assertEqual(self.paths(listing), ["src/main.py"])
        for name in blocked:
            with self.subTest(path=name):
                self.assert_rejected("read_file", {"path": name})
        for directory in (".git", "node_modules", "build", "target"):
            with self.subTest(directory=directory):
                self.assert_rejected("list_files", {"path": directory})
        self.assertEqual(self.tools.remaining_read_bytes, 16384)

    def test_binary_disguised_as_source_and_invalid_encoding_have_no_read_receipt(self):
        fixtures = {
            "fake.py": b"\x00\x01\x02binary payload\x00",
            "fake.customlang": b"hello\x00world",
            "invalid.ts": b"\xff\xfe\xff",  # Incomplete UTF-16 despite a BOM.
            "invalid.customlang": b"\x80\x81\xff",
        }
        for name, payload in fixtures.items():
            (self.project / name).write_bytes(payload)
            with self.subTest(path=name):
                self.assert_rejected("read_file", {"path": name})
        self.assertEqual(self.tools.read_paths, [])
        self.assertEqual(self.tools.remaining_read_bytes, 16384)

    def test_default_read_returns_two_hundred_lines_and_can_resume_exactly(self):
        lines = [f"line {number}\n" for number in range(1, 261)]
        self.write("many.py", "".join(lines))
        first = self.assert_ok("read_file", {"path": "many.py"})
        self.assertEqual(first["content"].splitlines(), [line.rstrip("\n") for line in lines[:200]])
        self.assertEqual((first["start_line"], first["end_line"]), (1, 200))
        self.assertIs(first["truncated"], True)
        self.assertEqual(first["next_line"], 201)
        second = self.assert_ok("read_file", {"path": "many.py", "start_line": first["next_line"]})
        self.assertEqual(second["content"].splitlines(), [line.rstrip("\n") for line in lines[200:]])
        self.assertEqual((second["start_line"], second["end_line"]), (201, 260))
        self.assertIs(second["truncated"], False)
        self.assertIsNone(second["next_line"])

    def test_explicit_line_range_uses_inclusive_source_line_numbers(self):
        self.write("src/main.go", "one\ntwo\nthree\nfour\nfive")
        result = self.assert_ok("read_file", {"path": "src/main.go", "start_line": 2, "max_lines": 2})
        self.assertEqual(result["content"].splitlines(), ["two", "three"])
        self.assertEqual((result["start_line"], result["end_line"]), (2, 3))
        self.assertEqual(result["next_line"], 4)
        self.assertIs(result["truncated"], True)
        last = self.assert_ok("read_file", {"path": "src/main.go", "start_line": 5, "max_lines": 500})
        self.assertEqual(last["content"], "five")
        self.assertEqual((last["start_line"], last["end_line"]), (5, 5))
        self.assertIsNone(last["next_line"])

    def test_source_line_numbers_use_only_lf_crlf_and_cr_boundaries(self):
        self.write("source.f", "one\fpart\u2028still-line\r\ntwo\rthree\n")
        first = self.assert_ok("read_file", {"path": "source.f", "max_lines": 1})
        self.assertEqual(first["content"], "one\fpart\u2028still-line\r\n")
        self.assertEqual((first["start_line"], first["end_line"], first["total_lines"]), (1, 1, 3))
        self.assertEqual(first["next_line"], 2)
        second = self.assert_ok("read_file", {"path": "source.f", "start_line": 2, "max_lines": 1})
        self.assertEqual(second["content"], "two\r")
        self.assertEqual((second["start_line"], second["end_line"], second["total_lines"]), (2, 2, 3))
        self.assertEqual(second["next_line"], 3)
        last = self.assert_ok("read_file", {"path": "source.f", "start_line": 3, "max_lines": 1})
        self.assertEqual(last["content"], "three\n")
        self.assertEqual((last["start_line"], last["end_line"], last["total_lines"]), (3, 3, 3))
        self.assertIsNone(last["next_line"])

    def test_invalid_line_range_types_and_limits_do_not_consume_budget(self):
        self.write("main.py", "example\n")
        invalid = [
            {"start_line": 0}, {"start_line": -1}, {"start_line": True},
            {"start_line": "1"}, {"start_line": 1.5},
            {"max_lines": 0}, {"max_lines": -1}, {"max_lines": 501},
            {"max_lines": True}, {"max_lines": "20"}, {"max_lines": 2.5},
            {"unexpected": 1},
        ]
        for options in invalid:
            with self.subTest(options=options):
                self.assert_rejected("read_file", {"path": "main.py", **options})
        self.assertEqual(self.tools.read_paths, [])
        self.assertEqual(self.tools.remaining_read_bytes, 16384)

    def test_text_above_sixty_four_kib_is_readable_in_bounded_chunks(self):
        text = "".join(f"line {number:05}: content\n" for number in range(7000))
        source = self.write("large.customlang", text)
        self.assertGreater(source.stat().st_size, 65536)
        result = self.assert_ok("read_file", {"path": source.name, "start_line": 4001, "max_lines": 10})
        self.assertEqual(result["content"].splitlines(), text.splitlines()[4000:4010])
        self.assertEqual((result["start_line"], result["end_line"]), (4001, 4010))
        self.assertEqual(result["next_line"], 4011)
        self.assertLessEqual(len(result["content"].encode("utf-8")), 65536)

    def test_file_above_two_mib_is_rejected_even_when_requested_chunk_is_tiny(self):
        (self.project / "oversize.py").write_bytes(b"x\n" * (1024 * 1024) + b"x")
        self.assert_rejected("read_file", {"path": "oversize.py", "max_lines": 1})
        self.assertEqual(self.tools.read_paths, [])
        self.assertEqual(self.tools.remaining_read_bytes, 16384)

    def test_utf16_bom_crlf_and_cyrillic_are_decoded_and_charged_as_returned_utf8(self):
        self.write("Example.cs", "первая\r\nвторая\r\nтретья", encoding="utf-16")
        before = self.tools.remaining_read_bytes
        result = self.assert_ok("read_file", {"path": "Example.cs", "start_line": 2, "max_lines": 1})
        self.assertEqual(result["content"].splitlines(), ["вторая"])
        self.assertEqual(result["content"], "вторая\r\n")
        self.assertNotIn("\ufeff", result["content"])
        self.assertEqual((result["start_line"], result["end_line"]), (2, 2))
        self.assertEqual(result["next_line"], 3)
        expected = before - len(result["content"].encode("utf-8"))
        self.assertEqual(result["remaining_read_bytes"], expected)
        self.assertEqual(self.tools.remaining_read_bytes, expected)

    def test_utf8_bom_does_not_become_source_content(self):
        self.write("bom.py", "первая\nвторая", encoding="utf-8-sig")
        result = self.assert_ok("read_file", {"path": "bom.py", "max_lines": 1})
        self.assertEqual(result["content"].splitlines(), ["первая"])
        self.assertNotIn("\ufeff", result["content"])
        self.assertEqual(result["remaining_read_bytes"], 16384 - len(result["content"].encode("utf-8")))

    def test_repeated_multibyte_reads_spend_aggregate_budget_again(self):
        self.write("main.py", "яяя")
        tools = FileTools(self.project, self.report, project_mode=True, max_total_read_bytes=12)
        for expected in (6, 0):
            result = tools.execute("read_file", {"path": "main.py"})
            self.assertIs(result["ok"], True, result)
            self.assertEqual(result["remaining_read_bytes"], expected)
        self.assertIs(tools.execute("read_file", {"path": "main.py"})["ok"], False)
        self.assertEqual(tools.remaining_read_bytes, 0)
        self.assertEqual(tools.read_paths, ["main.py"])

    def test_byte_budget_returns_only_whole_lines_and_cannot_be_spent_by_failed_read(self):
        self.write("main.py", "1234567\nяяяя\nlast\n")
        self.write("other.py", "ok\n")
        tools = FileTools(self.project, self.report, project_mode=True, max_total_read_bytes=12)
        first = tools.execute("read_file", {"path": "main.py", "max_lines": 500})
        self.assertIs(first["ok"], True, first)
        self.assertEqual(first["content"], "1234567\n")
        self.assertEqual((first["start_line"], first["end_line"]), (1, 1))
        self.assertEqual(first["total_lines"], 3)
        self.assertEqual(first["next_line"], 2)
        self.assertEqual(first["remaining_read_bytes"], 4)
        blocked = tools.execute("read_file", {"path": "main.py", "start_line": 2})
        self.assertIs(blocked["ok"], False, blocked)
        self.assertEqual(tools.remaining_read_bytes, 4)
        other = tools.execute("read_file", {"path": "other.py"})
        self.assertIs(other["ok"], True, other)
        self.assertEqual(other["remaining_read_bytes"], 1)
        self.assertEqual(tools.read_ranges, [
            {"path": "main.py", "start_line": 1, "end_line": 1},
            {"path": "other.py", "start_line": 1, "end_line": 1},
        ])

    def test_chunk_limit_applies_even_when_aggregate_budget_is_larger(self):
        self.write("many.py", ("x" * 199 + "\n") * 600)
        tools = FileTools(self.project, self.report, project_mode=True, max_total_read_bytes=131072)
        result = tools.execute("read_file", {"path": "many.py", "max_lines": 500})
        self.assertIs(result["ok"], True, result)
        self.assertEqual(result["content"], ("x" * 199 + "\n") * 327)
        self.assertEqual((result["start_line"], result["end_line"]), (1, 327))
        self.assertEqual(result["total_lines"], 600)
        self.assertEqual(result["next_line"], 328)
        self.assertIs(result["truncated"], True)
        self.assertEqual(result["remaining_read_bytes"], 131072 - 65400)

    def test_empty_file_has_explicit_empty_range_and_no_budget_charge(self):
        self.write("empty.py", "")
        result = self.assert_ok("read_file", {"path": "empty.py"})
        self.assertEqual(result["content"], "")
        self.assertEqual((result["start_line"], result["end_line"], result["total_lines"]), (1, 0, 0))
        self.assertIs(result["truncated"], False)
        self.assertIsNone(result["next_line"])
        self.assertEqual(result["remaining_read_bytes"], 16384)

    def test_read_ranges_are_chronological_copies_and_exclude_failed_requests(self):
        self.write("src/main.py", "one\ntwo\nthree\nfour\nfive\n")
        self.assert_ok("read_file", {"path": "src/main.py", "start_line": 4, "max_lines": 1})
        self.assert_rejected("read_file", {"path": "missing.py"})
        self.assert_ok("read_file", {"path": "src/main.py", "start_line": 1, "max_lines": 2})
        expected = [
            {"path": "src/main.py", "start_line": 4, "end_line": 4},
            {"path": "src/main.py", "start_line": 1, "end_line": 2},
        ]
        self.assertEqual(self.tools.read_ranges, expected)
        ranges = self.tools.read_ranges
        ranges[0]["start_line"] = 999
        ranges.append({"path": "not_read.py", "start_line": 1, "end_line": 20})
        self.assertEqual(self.tools.read_ranges, expected)

    def test_report_evidence_identifies_actual_fragments_and_excludes_unread_files(self):
        self.write("src/main.py", "one\ntwo\nthree\nfour\nfive\n")
        self.write("unread.py", "not inspected\n")
        self.assert_ok("list_files", {})
        self.assert_ok("read_file", {"path": "src/main.py", "start_line": 2, "max_lines": 2})
        self.assert_ok("read_file", {"path": "src/main.py", "start_line": 5, "max_lines": 1})
        self.assert_rejected("read_file", {"path": "missing.py"})
        self.assert_ok("save_report", {"content": "Обзор только прочитанных фрагментов."})
        saved = self.report.read_text(encoding="utf-8")
        self.assertIn("Фактически прочитанные фрагменты (отметка программы):", saved)
        self.assertIn("`src/main.py`", saved)
        self.assertIn("строки 2–3", saved)
        self.assertIn("строки 5–5", saved)
        self.assertNotIn("unread.py", saved)
        self.assertNotIn("missing.py", saved)
        self.assertNotIn("Фактически прочитанные файлы (отметка программы):", saved)
        self.assertIs(self.tools.report_saved, True)

    def test_scan_limit_is_bounded_for_large_directories(self):
        for number in range(2010):
            self.write(f"source_{number:04}.py", "x")
        result = self.assert_ok("list_files", {})
        self.assertLessEqual(result["scanned"], 2000)
        self.assertLessEqual(len(result["files"]), 200)
        self.assertIs(result["truncated"], True)

    def test_hardlinked_source_is_neither_listed_nor_read(self):
        outside = self.root / "outside.py"
        outside.write_text("synthetic private fixture\n", encoding="utf-8")
        try:
            os.link(outside, self.project / "linked.py")
        except (OSError, NotImplementedError) as error:
            self.skipTest(f"Hard links are unavailable: {error}")
        self.assertEqual(self.paths(self.assert_ok("list_files", {})), [])
        self.assert_rejected("read_file", {"path": "linked.py"})

    def test_reparse_subtree_is_neither_traversed_nor_read_directly(self):
        self.write("linked/main.py")
        original_lstat = Path.lstat

        def marked_lstat(path):
            if path == self.project / "linked":
                return SimpleNamespace(st_mode=stat.S_IFDIR, st_file_attributes=0x400)
            return original_lstat(path)

        with patch.object(Path, "lstat", autospec=True, side_effect=marked_lstat):
            self.assertEqual(self.paths(self.assert_ok("list_files", {})), [])
            self.assert_rejected("list_files", {"path": "linked"})
            self.assert_rejected("read_file", {"path": "linked/main.py"})

    def test_default_mode_still_reads_whole_documents_and_rejects_range_arguments(self):
        text = "".join(f"строка {number}\n" for number in range(250))
        self.write("notes.txt", text)
        self.write("main.py", "print('example')\n")
        tools = FileTools(self.project, self.report, max_total_read_bytes=1)
        result = tools.execute("read_file", {"path": "notes.txt"})
        self.assertEqual(result, {"ok": True, "path": "notes.txt", "content": text})
        self.assertIs(tools.execute("read_file", {"path": "notes.txt", "start_line": 1})["ok"], False)
        self.assertIs(tools.execute("read_file", {"path": "main.py"})["ok"], False)
        self.assertIs(tools.execute("list_files", {})["ok"], False)
        self.assertIs(tools.execute("search_text", {"query": "example"})["ok"], False)

    def test_search_locations_can_be_read_with_the_same_line_numbers(self):
        for name, encoding in [("src/a", "utf-16"), ("src/b", "utf-32")]:
            self.write(name, "lower\rX[1] X[1]\r\nabc\fpart\u2028X[1]\n", encoding=encoding)
        self.write("other.md", "X[1]")
        result = self.assert_ok("search_text", {"query": "X[1]", "path": "src"})
        self.assertEqual(result["matches"], [{"path": path, "line": line}
                                             for path in ("src/a", "src/b") for line in (2, 3)])
        self.assertFalse(result["truncated"])
        self.assertEqual(self.tools.remaining_read_bytes, 16384)
        self.assertEqual(self.tools.read_ranges, [])
        self.assert_ok("list_files", {})
        self.assert_rejected("save_report", {"content": "Ещё не читали."})
        for hit in result["matches"]:
            read = self.assert_ok("read_file", {"path": hit["path"], "start_line": hit["line"], "max_lines": 1})
            self.assertIn("X[1]", read["content"])
        self.assertEqual(self.assert_ok("search_text", {"query": "x[1]"})["matches"], [])

    def test_search_reports_limits_and_skipped_files(self):
        self.write("a.txt", "abc\nabc")
        self.write("b.txt", "abc")
        exact = self.assert_ok("search_text", {"query": "abc", "path": "a.txt", "limit": 2})
        self.assertFalse(exact["truncated"])
        limited = self.assert_ok("search_text", {"query": "abc", "limit": 2})
        self.assertTrue(limited["truncated"])
        self.assertEqual(limited["matches"], exact["matches"])
        for field, maximum in [("_MAX_SEARCH_BYTES", 7), ("_MAX_SEARCH_FILES", 1),
                               ("_MAX_SCANNED", 1)]:
            with self.subTest(field=field), patch.object(self.tools, field, maximum):
                limited = self.assert_ok("search_text", {"query": "absent"})
                self.assertTrue(limited["truncated"])
                self.assertEqual(limited["files_searched"], 1)
        self.write("bad.md", "abc\x00")
        partial = self.assert_ok("search_text", {"query": "abc"})
        self.assertEqual(partial["files_skipped"], 1)
        self.assertTrue(partial["truncated"])
        self.assert_rejected("search_text", {"path": "bad.md", "query": "abc"})

    def test_search_uses_the_same_file_policy_and_checks_arguments(self):
        for path in (".env", ".git/config", "build/main.py", "credentials.json", "img.png"):
            self.write(path, "needle")
            self.assert_rejected("search_text", {"query": "needle", "path": path})
        self.assertEqual(self.assert_ok("search_text", {"query": "needle"})["matches"], [])
        for args in ({}, {"query": " "}, {"query": "x\ny"}, {"query": "a" * 257},
                     {"query": False}, {"query": "a", "limit": True}, {"query": "a", "limit": 31},
                     {"query": "a", "path": ".."}, {"query": "a", "regex": True}):
            with self.subTest(args=args):
                self.assert_rejected("search_text", args)


if __name__ == "__main__":
    unittest.main()
