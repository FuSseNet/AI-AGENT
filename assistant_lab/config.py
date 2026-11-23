"""Настройки локального подключения. Файл .env не исполняется как код."""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


class ConfigError(ValueError):
    """Ошибка пользовательских настроек."""


def load_env(path: Path) -> dict[str, str]:
    """Прочитать KEY=VALUE; переменные окружения имеют приоритет."""
    values: dict[str, str] = {}
    if path.exists():
        try:
            if path.stat().st_size > 16_384:
                raise ConfigError("Файл .env слишком большой (максимум 16 КиБ).")
            lines = path.read_text(encoding="utf-8-sig").splitlines()
        except (OSError, UnicodeError) as exc:
            raise ConfigError("Не удалось прочитать .env как UTF-8.") from exc
        for number, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                raise ConfigError(f"В .env на строке {number} ожидается KEY=VALUE.")
            key, value = (part.strip() for part in line.split("=", 1))
            if not key.startswith("ASSISTANT_"):
                continue
            if value[:1] in {"'", '"'}:
                if len(value) < 2 or value[-1] != value[0]:
                    raise ConfigError(f"В .env на строке {number} не закрыта кавычка.")
                value = value[1:-1]
            values[key] = value
    values.update({key: value for key, value in os.environ.items()
                   if key.startswith("ASSISTANT_")})
    return values


def local_base_url(value: str) -> str:
    """Этот прототип обращается только к API на самом компьютере."""
    try:
        parts = urlsplit(value.strip())
        port = parts.port  # Проверить диапазон и синтаксис порта.
    except ValueError as exc:
        raise ConfigError("Некорректный адрес API.") from exc
    if parts.scheme not in {"http", "https"} or parts.hostname not in {
        "localhost", "127.0.0.1", "::1"
    }:
        raise ConfigError("Укажите локальный API: http://127.0.0.1:ПОРТ/v1 "
                          "(также поддерживаются localhost и [::1]).")
    if parts.username is not None or parts.password is not None:
        raise ConfigError("Адрес API не должен содержать логин или пароль.")
    if parts.query or parts.fragment or parts.path.rstrip("/") not in {"", "/v1"}:
        raise ConfigError("Нужен базовый адрес с окончанием /v1, без /chat/completions.")
    if port == 0:
        raise ConfigError("Порт API должен быть больше нуля.")
    return urlunsplit((parts.scheme, parts.netloc, "/v1", "", ""))


def positive_int(value: str, name: str, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name}: ожидается целое число.") from exc
    if not 1 <= parsed <= maximum:
        raise ConfigError(f"{name}: допустимы значения от 1 до {maximum}.")
    return parsed
