"""Политика отбора текстовых файлов без привязки к языку проекта.

Имена и расширения помогают исключить очевидные артефакты и секреты.
Они не доказывают отсутствие чувствительных данных внутри обычного исходника.
Проверку путей, ссылок, прав и размеров выполняет вызывающий инструмент.
"""

from __future__ import annotations

import codecs
from pathlib import PurePath
import re
import stat
import unicodedata


__all__ = [
    "ignored_directory",
    "text_candidate",
    "decode_text",
    "is_hidden",
    "hidden_name_allowed",
]


_SAFE_HIDDEN_FILES = frozenset({
    ".gitignore", ".gitattributes", ".editorconfig", ".dockerignore",
    ".prettierrc", ".eslintrc",
})
_SAFE_HIDDEN_DIRECTORIES = frozenset({".github", ".gitlab"})
_DOT_CONFIG_SUFFIXES = frozenset({
    ".json", ".json5", ".yaml", ".yml", ".toml", ".js", ".cjs", ".mjs",
})
_IGNORED_DIRECTORIES = frozenset({
    ".git", ".hg", ".svn", ".idea", ".vscode", ".vs", ".venv", "venv", "env",
    "node_modules", "vendor", "packages", "site-packages", "build", "dist", "target",
    "bin", "obj", "out", "__pycache__", "cmakefiles", ".gradle", ".mvn",
    ".next", ".nuxt", ".cache", "coverage", ".terraform", ".tox", ".nox",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", ".parcel-cache", ".yarn",
    ".pnpm-store", "htmlcov",
})
_CREDENTIAL_DIRECTORIES = frozenset({
    "secrets", "credentials", "certs", "certificates", "private_key", "private_keys",
    "keys", "keystore", "keystores",
})
_BINARY_EXTENSIONS = frozenset({
    # Исполняемые файлы, пакеты и результаты компиляции.
    ".exe", ".dll", ".com", ".msi", ".so", ".dylib", ".a", ".lib", ".o", ".obj",
    ".class", ".jar", ".war", ".ear", ".pyc", ".pyo", ".pyd", ".wasm", ".bin",
    ".dex", ".apk", ".aab", ".ipa", ".deb", ".rpm", ".nupkg",
    # Архивы и образы дисков.
    ".zip", ".7z", ".rar", ".tar", ".gz", ".bz2", ".xz", ".zst", ".tgz",
    ".tbz2", ".txz", ".lz", ".lzma", ".br", ".cab", ".iso", ".dmg",
    # Изображения, звук, видео, шрифты и бинарные 3D-ресурсы.
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".ico", ".tif", ".tiff",
    ".avif", ".heic", ".heif", ".psd", ".xcf", ".mp3", ".wav", ".flac", ".ogg",
    ".m4a", ".aac", ".opus", ".wma", ".mp4", ".mkv", ".mov", ".avi", ".webm",
    ".mpeg", ".mpg", ".m4v", ".3gp", ".woff", ".woff2", ".ttf", ".otf", ".eot",
    ".fbx", ".blend", ".stl", ".glb", ".usdc",
    # Документы-контейнеры, базы данных, модели и числовые массивы.
    ".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx", ".odt", ".ods",
    ".odp", ".db", ".db3", ".sqlite", ".sqlite3", ".mdb", ".accdb", ".pkl",
    ".pickle", ".pt", ".pth", ".onnx", ".h5", ".hdf5", ".safetensors", ".gguf",
    ".ggml", ".ckpt", ".model", ".npy", ".npz", ".parquet", ".arrow", ".feather",
    ".mat",
})
_KEY_EXTENSIONS = frozenset({
    ".pem", ".key", ".p12", ".pfx", ".crt", ".cer", ".der", ".jks",
    ".keystore", ".kdbx", ".gpg", ".pgp", ".pub",
})
_CREDENTIAL_DATA_EXTENSIONS = frozenset({
    "", ".json", ".json5", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf",
    ".properties", ".xml", ".txt", ".md", ".rst", ".text", ".csv", ".tsv", ".config", ".env", ".log",
    ".bak", ".backup", ".old",
})
_CREDENTIAL_WORDS = frozenset({
    "credential", "credentials", "secret", "secrets", "password", "passwords",
    "token", "tokens", "keystore", "keystores", "certificate", "certificates",
    "cert", "certs", "key", "keys", "apikey", "apikeys", "privatekey", "privatekeys",
})
_PRIVATE_KEY_NAMES = frozenset({
    "id_rsa", "id_dsa", "id_ecdsa", "id_ed25519", "authorized_keys", "known_hosts",
})
_ALLOWED_CONTROLS = frozenset({"\t", "\n", "\r", "\f"})


def _valid_name(name: str) -> bool:
    return (
        isinstance(name, str)
        and name not in {"", ".", ".."}
        and not any(char in "/\\:" or unicodedata.category(char) == "Cc" for char in name)
    )


def _environment_name(name: str) -> bool:
    lowered = name.casefold()
    return (
        lowered == ".env"
        or lowered.startswith(".env.")
        or ".env" in PurePath(lowered).suffixes
    )


def hidden_name_allowed(name: str, is_directory: bool) -> bool:
    """True только для явно разрешённого скрытого имени, а не любого видимого.

    Вызывающий код может применить эту функцию и к Windows HIDDEN-атрибуту.
    Разрешение .github не даёт разрешения на скрытые файлы внутри неё.
    """
    if not _valid_name(name) or _environment_name(name):
        return False
    lowered = name.casefold()
    if is_directory:
        return lowered in _SAFE_HIDDEN_DIRECTORIES
    if lowered in _SAFE_HIDDEN_FILES:
        return True
    return any(
        lowered.startswith(prefix) and lowered[len(prefix):] in _DOT_CONFIG_SUFFIXES
        for prefix in (".prettierrc", ".eslintrc")
    )


def ignored_directory(name: str) -> bool:
    """Исключить служебные каталоги, зависимости, скрытые и credential-папки."""
    if not _valid_name(name):
        return True
    lowered = name.casefold()
    normalized = lowered.replace("-", "_")
    return (
        lowered in _IGNORED_DIRECTORIES
        or normalized in _CREDENTIAL_DIRECTORIES
        or lowered.startswith("cmake-build-")
        or (name.startswith(".") and not hidden_name_allowed(name, is_directory=True))
    )


def _credential_file(name: str) -> bool:
    lowered = name.casefold()
    if _environment_name(name):
        return True
    # Копия ключа с дополнительным суффиксом (.pem.bak) тоже не кандидат.
    suffixes = PurePath(lowered).suffixes
    if any(suffix in _KEY_EXTENSIONS for suffix in suffixes):
        return True
    if any(
        lowered == key or any(lowered.startswith(key + separator) for separator in (".", "_", "-"))
        for key in _PRIVATE_KEY_NAMES
    ):
        return True
    suffix = PurePath(lowered).suffix
    if suffix not in _CREDENTIAL_DATA_EXTENSIONS:
        return False
    stem = lowered[:-len(suffix)] if suffix else lowered
    words = set(filter(None, re.split(r"[._\-\s]+", stem)))
    if words & _CREDENTIAL_WORDS:
        return True
    normalized = re.sub(r"[.\-\s]+", "_", stem)
    return bool(re.search(
        r"(?:^|_)(?:private_keys?|privatekeys?|api_keys?|apikeys?|service_account|serviceaccount)(?:_|$)",
        normalized,
    ))


def text_candidate(name: str) -> bool:
    """Кандидат для последующей проверки содержимого; язык и расширение не нужны."""
    if not _valid_name(name):
        return False
    if name.startswith(".") and not hidden_name_allowed(name, is_directory=False):
        return False
    if _credential_file(name):
        return False
    return PurePath(name.casefold()).suffix not in _BINARY_EXTENSIONS


def is_hidden(info) -> bool:
    """Прочитать Windows HIDDEN-атрибут; точечные имена проверяются отдельно."""
    attributes = getattr(info, "st_file_attributes", 0)
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_HIDDEN", 0x2))


def decode_text(data: bytes) -> str:
    """Строго декодировать UTF-8, либо UTF-16/32 с BOM, и отсеять управляющие байты."""
    if not isinstance(data, bytes):
        raise ValueError("Для проверки текста нужен массив байтов.")
    if data.startswith((codecs.BOM_UTF32_LE, codecs.BOM_UTF32_BE)):
        encoding = "utf-32"
    elif data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        encoding = "utf-16"
    else:
        encoding = "utf-8-sig"
    try:
        text = data.decode(encoding, errors="strict")
    except UnicodeError:
        raise ValueError(
            "Файл не является поддерживаемым текстом: нужен UTF-8 или UTF-16/32 с BOM."
        ) from None
    if any(unicodedata.category(char) == "Cc" and char not in _ALLOWED_CONTROLS for char in text):
        raise ValueError("Файл содержит двоичные данные или недопустимые управляющие символы.")
    return text
