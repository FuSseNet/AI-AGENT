"""Проверки отбора файлов и декодирования без языкового белого списка."""

import codecs
from types import SimpleNamespace
import unittest

from assistant_lab.file_policy import (
    decode_text,
    hidden_name_allowed,
    ignored_directory,
    is_hidden,
    text_candidate,
)


class CandidatePolicyTests(unittest.TestCase):
    def test_languages_unknown_extensions_and_extensionless_files_are_candidates(self):
        names = (
            "Main.java", "pom.xml", "application.properties", "build.gradle", "settings.gradle.kts",
            "gradlew", "mvnw", "app.py", "package.json", "index.ts", "index.jsx", "App.vue",
            "main.go", "Cargo.toml", "lib.rs", "main.cpp", "program.cs", "app.csproj",
            "service.dll.config", "README", "Makefile", "Dockerfile", "data.custom_format",
            "unknown.brandnew", "config.yaml", "config.yml", "Cargo.lock", "notebook.ipynb",
        )
        for name in names:
            with self.subTest(name=name):
                self.assertTrue(text_candidate(name))

    def test_no_source_token_or_secret_substring_false_positives(self):
        for name in (
            "Tokenizer.java", "TokenService.java", "secret_detection.py", "secrets.py",
            "credentials.js", "PasswordValidator.cs", "tokenizer.properties", "secretary.txt",
            "tokenization.json", "credentialization.yaml", "myapikeydocumentation.properties",
            "keymap.properties", "KeyService.java",
        ):
            with self.subTest(name=name):
                self.assertTrue(text_candidate(name))

    def test_safe_hidden_configuration_files(self):
        allowed = (
            ".gitignore", ".gitattributes", ".editorconfig", ".dockerignore", ".prettierrc",
            ".prettierrc.json", ".prettierrc.yaml", ".prettierrc.cjs", ".prettierrc.mjs",
            ".eslintrc", ".eslintrc.json", ".eslintrc.yml", ".eslintrc.js",
        )
        for name in allowed:
            with self.subTest(name=name):
                self.assertTrue(hidden_name_allowed(name, is_directory=False))
                self.assertTrue(text_candidate(name))
        for name in (".private.txt", ".env", ".env.example", ".env.local", ".eslintrc.secret.json", ".npmrc"):
            with self.subTest(name=name):
                self.assertFalse(hidden_name_allowed(name, is_directory=False))
                self.assertFalse(text_candidate(name))
        self.assertFalse(hidden_name_allowed("ordinary.txt", is_directory=False))
        self.assertFalse(hidden_name_allowed(".github", is_directory=False))

    def test_safe_hidden_directories_do_not_allow_git_or_arbitrary_hidden_dirs(self):
        for name in (".github", ".gitlab", ".GITHUB"):
            with self.subTest(name=name):
                self.assertTrue(hidden_name_allowed(name, is_directory=True))
                self.assertFalse(ignored_directory(name))
        for name in (".git", ".mvn", ".gradle", ".other", ".env", ".env.local"):
            with self.subTest(name=name):
                self.assertFalse(hidden_name_allowed(name, is_directory=True))
                self.assertTrue(ignored_directory(name))

    def test_credential_data_keys_and_certificates_are_excluded(self):
        names = (
            ".env", ".ENV.local", ".env.example", "id_rsa", "id_rsa.pub", "id_rsa_backup", "id_ed25519",
            "authorized_keys", "key.pem", "key.pem.bak", "server.key", "store.p12",
            "store.pfx", "store.jks", "server.crt", "server.cer", "client.keystore",
            "credentials.json", "credentials.prod.json", "my-credentials.xml", "secrets.yaml",
            "passwords.txt", "access_token.txt", "api-key.properties", "private_key.json",
            "service-account.json", "serviceAccount.json", "token", "credentials", "secrets.yml.backup",
            "keys.properties", "apiKeys.json",
        )
        for name in names:
            with self.subTest(name=name):
                self.assertFalse(text_candidate(name))

    def test_binary_and_artifact_extensions_are_excluded_case_insensitively(self):
        for extension in (
            "exe", "DLL", "class", "jar", "png", "pdf", "zip", "7z", "pyc", "o", "obj",
            "so", "db", "sqlite", "pkl", "pt", "gguf", "safetensors", "npy", "parquet",
            "woff2", "mp4", "wav", "docx", "h5", "onnx", "wasm", "glb", "gz",
        ):
            with self.subTest(extension=extension):
                self.assertFalse(text_candidate("file." + extension))

    def test_credential_named_documents_remain_excluded(self):
        for name in ("credentials.md", "certificate.md", "secrets.rst", "api-keys.text"):
            with self.subTest(name=name):
                self.assertFalse(text_candidate(name))
        for name in ("secret_detection.py", "secret.cpp", "Tokens.hpp", "tokenizer.md"):
            with self.subTest(name=name):
                self.assertTrue(text_candidate(name))

    def test_environment_suffix_and_backup_are_excluded_without_source_false_positive(self):
        for name in ("production.env", "config.env.bak", "production.ENV"):
            with self.subTest(name=name):
                self.assertFalse(text_candidate(name))
        self.assertTrue(text_candidate("sourceEnvironment.java"))

    def test_dependency_build_cache_and_secret_directories_are_excluded(self):
        for name in (
            "vendor", "packages", "venv", "env", "node_modules", "build", "bin", "obj",
            "out", "target", "dist", "CMakeFiles", "cmake-build-debug", "CMAKE-BUILD-release",
            ".vs", ".idea", ".vscode", ".next", ".nuxt", ".cache", "coverage",
            "site-packages", ".terraform", "__pycache__", "secrets", "credentials", "certs",
        ):
            with self.subTest(name=name):
                self.assertTrue(ignored_directory(name))
        for name in ("src", "java", "python", "scripts", "docs", "secret_detection", "tokenizer"):
            with self.subTest(name=name):
                self.assertFalse(ignored_directory(name))

    def test_functions_take_names_not_paths_or_control_characters(self):
        for name in ("", ".", "..", "a/b.txt", "a\\b.txt", "a.txt:stream", "bad\x00.txt", "bad\n.txt"):
            with self.subTest(name=repr(name)):
                self.assertFalse(text_candidate(name))
                self.assertTrue(ignored_directory(name))

    def test_windows_hidden_attribute(self):
        self.assertTrue(is_hidden(SimpleNamespace(st_file_attributes=0x22)))
        self.assertFalse(is_hidden(SimpleNamespace(st_file_attributes=0x20)))
        self.assertFalse(is_hidden(SimpleNamespace()))


class DecodeTextTests(unittest.TestCase):
    def test_utf8_and_bom_utf8(self):
        text = "Привет, мир! 😀\n"
        self.assertEqual(decode_text(text.encode("utf-8")), text)
        self.assertEqual(decode_text(codecs.BOM_UTF8 + text.encode("utf-8")), text)
        self.assertEqual(decode_text(b""), "")

    def test_utf16_and_utf32_with_either_bom(self):
        text = "Текст с эмодзи 😀\n"
        for encoding, bom in (
            ("utf-16-le", codecs.BOM_UTF16_LE), ("utf-16-be", codecs.BOM_UTF16_BE),
            ("utf-32-le", codecs.BOM_UTF32_LE), ("utf-32-be", codecs.BOM_UTF32_BE),
        ):
            with self.subTest(encoding=encoding):
                self.assertEqual(decode_text(bom + text.encode(encoding)), text)

    def test_no_legacy_codepage_or_bomless_utf16_guessing(self):
        for data in (
            "Привет".encode("cp1251"), "Text".encode("utf-16-le"),
            "Text".encode("utf-16-be"), "Text".encode("utf-32-le"), b"\xff\xfe\x01",
        ):
            with self.subTest(data=data), self.assertRaises(ValueError):
                decode_text(data)

    def test_binary_content_with_text_name_is_rejected_after_decoding(self):
        self.assertTrue(text_candidate("pretends_to_be_text.unknown"))
        for data in (b"\x89PNG\r\n\x1a\n", b"text\x00binary", b"PK\x03\x04payload", b"\xff\xd8\xff"):
            with self.subTest(data=data), self.assertRaises(ValueError):
                decode_text(data)

    def test_controls_are_rejected_in_all_supported_encodings(self):
        for character in ("\x00", "\x01", "\x08", "\x0b", "\x1b", "\x7f", "\x85"):
            text = "начало" + character + "конец"
            for encoding in ("utf-8", "utf-16", "utf-32"):
                with self.subTest(character=repr(character), encoding=encoding), self.assertRaises(ValueError):
                    decode_text(text.encode(encoding))

    def test_tab_line_endings_and_form_feed_are_allowed(self):
        text = "column\tvalue\nnext\r\npage\fnew"
        self.assertEqual(decode_text(text.encode("utf-8")), text)

    def test_error_does_not_include_input_contents(self):
        with self.assertRaises(ValueError) as caught:
            decode_text(b"sensitive_marker\x00")
        self.assertNotIn("sensitive_marker", str(caught.exception))
        self.assertIn("Файл", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
