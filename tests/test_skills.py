"""Проверки контролируемой библиотеки навыков; все данные временные."""

import codecs
from dataclasses import FrozenInstanceError
import json
import os
from pathlib import Path
import stat
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from assistant_lab.skills import Skill, SkillError, SkillMetadata, SkillRegistry


class SkillRegistryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "library"
        self.root.mkdir()

    def write_skill(self, name="example-skill", description="Пример навыка", body="Инструкции навыка.", *, raw=None):
        directory = self.root / name
        directory.mkdir(exist_ok=True)
        if raw is None:
            raw = f"---\nname: {name}\ndescription: {description}\n---\n{body}\n".encode("utf-8")
        path = directory / "SKILL.md"
        path.write_bytes(raw)
        return path

    def assert_invalid_header(self, header, body="Инструкции."):
        self.write_skill(raw=f"---\n{header}\n---\n{body}\n".encode("utf-8"))
        with self.assertRaises(SkillError):
            SkillRegistry(self.root)

    def test_stable_metadata_only_catalog_and_immutable_skills(self):
        self.write_skill("zebra", body="Скрытое до загрузки тело zebra.")
        self.write_skill("alpha", description="Описание alpha", body="Тело alpha.")
        registry = SkillRegistry(self.root)
        metadata = registry.list_skills()
        self.assertEqual(metadata, [SkillMetadata("alpha", "Описание alpha"), SkillMetadata("zebra", "Пример навыка")])
        self.assertFalse(hasattr(metadata[0], "instructions"))
        metadata.clear()
        self.assertEqual(len(registry.list_skills()), 2)
        loaded = registry.load("alpha")
        self.assertEqual(loaded, Skill(SkillMetadata("alpha", "Описание alpha"), "Тело alpha."))
        with self.assertRaises(FrozenInstanceError):
            loaded.instructions = "Подмена"
        with self.assertRaises(FrozenInstanceError):
            loaded.metadata.name = "Подмена"

    def test_empty_catalog_is_valid(self):
        self.assertEqual(SkillRegistry(self.root).list_skills(), [])

    def test_json_description_utf8_bom_and_crlf(self):
        description = 'Описание: текст с "кавычками".'
        quoted = json.dumps(description, ensure_ascii=False)
        raw = codecs.BOM_UTF8 + f"---\r\nname: example-skill\r\ndescription: {quoted}\r\n---\r\n# Инструкции\r\n".encode("utf-8")
        self.write_skill(raw=raw)
        skill = SkillRegistry(self.root).load("example-skill")
        self.assertEqual(skill.metadata.description, description)
        self.assertEqual(skill.instructions, "# Инструкции")

    def test_plain_description_is_literal_without_yaml_evaluation(self):
        self.write_skill(description="Описание: это текст # с решёткой")
        self.assertEqual(SkillRegistry(self.root).load("example-skill").metadata.description, "Описание: это текст # с решёткой")

    def test_unknown_duplicate_and_missing_fields_are_rejected(self):
        headers = (
            "name: example-skill\ndescription: Описание\nversion: 1",
            "name: example-skill\nname: example-skill\ndescription: Описание",
            "name: example-skill\ndescription: Описание\ndescription: Ещё",
            "name: example-skill", "description: Описание",
            "name: example-skill\ndescription:",
        )
        for header in headers:
            with self.subTest(header=header):
                self.assert_invalid_header(header)

    def test_multiline_yaml_collections_and_single_quotes_are_rejected(self):
        descriptions = ("|", ">", "|-", "'Описание'", "[one, two]", "{key: value}")
        for description in descriptions:
            with self.subTest(description=description):
                self.assert_invalid_header(f"name: example-skill\ndescription: {description}")
        self.assert_invalid_header("name: example-skill\ndescription: Описание\n  продолжение")

    def test_description_json_must_be_one_complete_single_line_string(self):
        for description in ('"line\\nnext"', '"line\\tvalue"', '"not closed', '"ok" trailing', '"  "'):
            with self.subTest(description=description):
                self.assert_invalid_header(f"name: example-skill\ndescription: {description}")

    def test_invalid_names_and_directory_mismatch_are_rejected(self):
        for name in ("Upper", "has_underscore", "-bad", "bad-", "bad--name", "a~1", "../escape", "a" * 65, '"example-skill"', "different-name"):
            with self.subTest(name=name):
                self.assert_invalid_header(f"name: {name}\ndescription: Описание")

    def test_invalid_directory_name_is_rejected(self):
        (self.root / "Bad_Name").mkdir()
        with self.assertRaises(SkillError):
            SkillRegistry(self.root)

    def test_body_header_and_encoding_must_be_valid(self):
        fixtures = (
            b"Not a skill", b"---\nname: example-skill\ndescription: Example\n",
            b"---\nname: example-skill\ndescription: Example\n---\n \t\n",
            b"---\nname: example-skill\ndescription: Example\n---\n\xff",
            b"---\nname: example-skill\ndescription: Example\n---\nbody\x00",
        )
        for raw in fixtures:
            with self.subTest(raw=raw):
                self.write_skill(raw=raw)
                with self.assertRaises(SkillError):
                    SkillRegistry(self.root)

    def test_description_length_limit(self):
        self.write_skill(description="я" * 1024)
        self.assertEqual(len(SkillRegistry(self.root).load("example-skill").metadata.description), 1024)
        self.write_skill(description="я" * 1025)
        with self.assertRaises(SkillError):
            SkillRegistry(self.root)

    def test_skill_size_limit_includes_header(self):
        prefix = b"---\nname: example-skill\ndescription: Example\n---\n"
        path = self.write_skill(raw=prefix + b"x" * (32768 - len(prefix)))
        self.assertEqual(path.stat().st_size, 32768)
        SkillRegistry(self.root)
        path.write_bytes(path.read_bytes() + b"x")
        with self.assertRaises(SkillError):
            SkillRegistry(self.root)

    def test_read_is_bounded_if_skill_grows_after_metadata_checks(self):
        path = self.write_skill()
        original_fdopen = os.fdopen
        requested_sizes = []

        class GrowingReader:
            def __init__(self, descriptor, mode):
                self.source = original_fdopen(descriptor, mode)

            def __enter__(self):
                return self

            def fileno(self):
                return self.source.fileno()

            def read(self, size):
                requested_sizes.append(size)
                with path.open("ab") as target:
                    target.write(b"x" * 32768)
                return self.source.read(size)

            def __exit__(self, *exception):
                self.source.close()

        with patch("assistant_lab.skills.os.fdopen", side_effect=GrowingReader):
            with self.assertRaises(SkillError):
                SkillRegistry(self.root)
        self.assertEqual(requested_sizes, [32769])

    def test_catalog_limit_is_twenty_skills(self):
        for number in range(20):
            self.write_skill(f"skill-{number}")
        self.assertEqual(len(SkillRegistry(self.root).list_skills()), 20)
        self.write_skill("skill-20")
        with self.assertRaises(SkillError):
            SkillRegistry(self.root)

    def test_unknown_and_path_like_names_never_trigger_disk_reads(self):
        self.write_skill()
        registry = SkillRegistry(self.root)
        with patch("assistant_lab.skills.os.open", side_effect=AssertionError("Неожиданное чтение")):
            for name in ("unknown", "../escape", "/absolute", "C:\\file", "a~1", "", None, ["example-skill"]):
                with self.subTest(name=name), self.assertRaises(SkillError) as caught:
                    registry.load(name)
                self.assertNotIn("escape", str(caught.exception))

    def test_loaded_skills_are_a_predictable_snapshot(self):
        self.write_skill(description="Первое описание", body="Первая инструкция.")
        registry = SkillRegistry(self.root)
        self.write_skill(description="Новое описание", body="Новая инструкция.")
        self.write_skill("another")
        self.assertEqual(registry.list_skills(), [SkillMetadata("example-skill", "Первое описание")])
        self.assertEqual(registry.load("example-skill").instructions, "Первая инструкция.")
        self.assertEqual(len(SkillRegistry(self.root).list_skills()), 2)

    def test_missing_root_and_missing_skill_file_have_safe_errors(self):
        for root in (self.root / "missing", self.root):
            if root == self.root:
                (root / "missing-file").mkdir()
            with self.subTest(root=root), self.assertRaises(SkillError) as caught:
                SkillRegistry(root)
            self.assertNotIn(str(self.root), str(caught.exception))

    def test_hardlinked_skill_file_is_rejected(self):
        path = self.write_skill()
        try:
            os.link(path, Path(self.temporary.name) / "second-link.md")
        except (OSError, NotImplementedError) as error:
            self.skipTest(f"Жёсткие ссылки недоступны: {error}")
        with self.assertRaises(SkillError):
            SkillRegistry(self.root)

    def create_symlink(self, target, source, *, directory=False):
        try:
            source.symlink_to(target, target_is_directory=directory)
        except (OSError, NotImplementedError) as error:
            self.skipTest(f"Символические ссылки недоступны: {error}")

    def test_symlink_root_is_rejected(self):
        self.write_skill()
        alias = Path(self.temporary.name) / "linked-root"
        self.create_symlink(self.root, alias, directory=True)
        with self.assertRaises(SkillError):
            SkillRegistry(alias)

    def test_symlink_skill_directory_is_rejected(self):
        path = self.write_skill()
        self.create_symlink(path.parent, self.root / "linked-skill", directory=True)
        with self.assertRaises(SkillError):
            SkillRegistry(self.root)

    def test_symlink_skill_file_is_rejected(self):
        path = self.write_skill()
        original = Path(self.temporary.name) / "original.md"
        path.rename(original)
        self.create_symlink(original, path)
        with self.assertRaises(SkillError):
            SkillRegistry(self.root)

    def test_reparse_parent_is_rejected_without_symlink_privileges(self):
        self.write_skill()
        original_lstat = Path.lstat

        def marked_lstat(path):
            if path == self.root.parent:
                return SimpleNamespace(st_mode=stat.S_IFDIR, st_file_attributes=0x400)
            return original_lstat(path)

        with patch.object(Path, "lstat", autospec=True, side_effect=marked_lstat):
            with self.assertRaises(SkillError):
                SkillRegistry(self.root)


if __name__ == "__main__":
    unittest.main()
