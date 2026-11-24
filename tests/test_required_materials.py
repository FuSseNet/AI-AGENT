"""Сохранение зависимого результата требует покрытия всех входных материалов."""

import os
from pathlib import Path
import tempfile
import unittest

from assistant_lab.events import EventLog
from assistant_lab.skills import SkillRegistry
from assistant_lab.skill_tools import SkillToolSet
from assistant_lab.tools import FileTools


class RequiredMaterialsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.inputs = self.root / "inputs"
        self.inputs.mkdir()
        self.names = ("one.md", "two.md", "three.md")
        for name in self.names:
            (self.inputs / name).write_bytes(b"first\r\nsecond\r\nthird\r\nfourth\r\nfifth\r\n")
        self.report = self.root / "report.md"
        self.files = FileTools(self.inputs, self.report, project_mode=True)
        self.tools = SkillToolSet(self.files, SkillRegistry(), EventLog(self.root / "log.jsonl"),
                                  selected_name="compare-materials", required_reads=self.names)
        self.assertTrue(self.tools.execute("list_files", {})["ok"])

    def read(self, path, start=1, lines=200):
        result = self.tools.execute("read_file", {"path": path, "start_line": start, "max_lines": lines})
        self.assertTrue(result["ok"], result)
        return result

    def save(self):
        return self.tools.execute("save_report", {"content": "Сравнение всех переданных материалов."})

    def test_two_read_reports_do_not_authorize_omitting_third(self):
        self.read("one.md")
        self.read("two.md")
        result = self.save()
        self.assertFalse(result["ok"])
        self.assertEqual(result["missing"], ["three.md"])
        self.assertFalse(self.report.exists())
        self.read("three.md")
        self.assertTrue(self.save()["ok"])

    def test_repeats_and_last_line_do_not_cover_missing_middle(self):
        self.read("two.md")
        self.read("three.md")
        self.read("one.md", 1, 2)
        self.read("one.md", 1, 2)
        self.read("one.md", 5, 1)
        self.assertFalse(self.save()["ok"])
        self.assertFalse(self.report.exists())
        self.read("one.md", 3, 2)
        self.assertTrue(self.save()["ok"])

    def test_overlapping_out_of_order_ranges_can_cover_entire_material(self):
        self.read("two.md")
        self.read("three.md")
        self.read("one.md", 3, 3)
        self.assertFalse(self.save()["ok"])
        self.read("one.md", 1, 3)
        self.assertTrue(self.save()["ok"])

    def test_empty_material_is_accounted_for_only_after_actual_read(self):
        (self.inputs / "three.md").write_bytes(b"")
        self.read("one.md")
        self.read("two.md")
        self.assertFalse(self.save()["ok"])
        result = self.read("three.md")
        self.assertEqual(result["total_lines"], 0)
        self.assertTrue(self.save()["ok"])

    @unittest.skipUnless(os.name == "nt", "Windows paths are case-insensitive")
    def test_windows_successful_read_with_other_case_counts_as_required_material(self):
        self.read("ONE.MD")
        self.read("TWO.MD")
        self.read("THREE.MD")
        self.assertTrue(self.save()["ok"])


if __name__ == "__main__":
    unittest.main()
