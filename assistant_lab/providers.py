"""Минимальный клиент локального OpenAI-compatible API без SDK."""

from __future__ import annotations

import json
import socket
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from .config import local_base_url


class ModelError(RuntimeError):
    """Безопасное сообщение об ошибке без тела HTTP-ответа и ключей."""


class _NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ModelError("API вернул перенаправление. Укажите конечный локальный адрес.")


class LocalChatModel:
    def __init__(self, base_url: str, model: str, api_key: str = "",
                 timeout: int = 60, max_tokens: int = 2048):
        self.base_url = local_base_url(base_url)
        self.model = model
        self.api_key = api_key
        if not 1 <= timeout <= 600 or not 1 <= max_tokens <= 16_384:
            raise ValueError("Недопустимый timeout или max_tokens.")
        self.timeout = timeout
        self.max_tokens = max_tokens
        # Локальные запросы не должны проходить через системный прокси.
        self._opener = build_opener(ProxyHandler({}), _NoRedirects())

    def _request(self, endpoint: str, payload: dict | None = None) -> dict:
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        data = None
        if payload is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = Request(self.base_url + endpoint, data=data, headers=headers)
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                raw = response.read(524_289)
        except HTTPError as exc:
            status = exc.code
            exc.close()
            hints = {
                400: "Проверьте имя модели и поддержку tools в выбранной модели/сервере.",
                401: "Проверьте ASSISTANT_API_KEY, если сервер требует ключ.",
                403: "Сервер отказал в доступе.",
                404: "Проверьте базовый адрес /v1 и имя модели.",
                429: "Сервер занят или ограничил количество запросов.",
            }
            raise ModelError(f"Локальный API вернул HTTP {status}. "
                             + hints.get(status, "Проверьте состояние сервера.")) from None
        except (TimeoutError, socket.timeout):
            raise ModelError("Истекло время ожидания локальной модели.") from None
        except (URLError, OSError):
            raise ModelError("Не удалось подключиться к локальной модели. "
                             "Запустите API в Unsloth и проверьте адрес и порт.") from None
        if len(raw) > 524_288:
            raise ModelError("Ответ API превышает лимит 512 КиБ.")
        try:
            result = json.loads(raw.decode("utf-8"))
        except (UnicodeError, ValueError):
            raise ModelError("API вернул некорректный JSON.") from None
        if not isinstance(result, dict):
            raise ModelError("API вернул неожиданный формат ответа.")
        return result

    def list_models(self) -> list[str]:
        data = self._request("/models").get("data")
        if not isinstance(data, list):
            raise ModelError("API не вернул список моделей в поле data.")
        return [item["id"] for item in data
                if isinstance(item, dict) and isinstance(item.get("id"), str)]

    def complete(self, messages: list[dict], tools: list[dict]) -> dict:
        result = self._request("/chat/completions", {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "max_tokens": self.max_tokens,
            "stream": False,
        })
        choices = result.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise ModelError("API не вернул choices[0]. Проверьте загруженную модель.")
        choice = choices[0]
        if choice.get("finish_reason") in {"length", "content_filter"}:
            raise ModelError("Ответ модели прерван. Проверьте лимит токенов и настройки модели.")
        message = choice.get("message")
        if not isinstance(message, dict) or message.get("role") != "assistant":
            raise ModelError("API не вернул сообщение assistant.")
        # Не пересылаем серверные поля, которые не относятся к истории диалога.
        clean = {"role": "assistant", "content": message.get("content")}
        if message.get("tool_calls") is not None:
            clean["tool_calls"] = message["tool_calls"]
        return clean


class DemoModel:
    """Сценарная имитация: реальные инструменты, но без нейросети и сети."""

    def __init__(self, source: str = "README.md", *, search_query: str | None = None):
        self.source = source
        self.search_query = search_query

    @staticmethod
    def _call(call_id: str, tool_name: str, **arguments) -> dict:
        return {"role": "assistant", "content": None, "tool_calls": [{
            "id": call_id, "type": "function", "function": {
                "name": tool_name, "arguments": json.dumps(arguments, ensure_ascii=False)
            }
        }]}

    def complete(self, messages: list[dict], tools: list[dict]) -> dict:
        if messages[-1]["role"] != "tool":
            if self.search_query is not None:
                return self._call("demo_list", "list_files", limit=10)
            return self._call("demo_read", "read_file", path=self.source)
        try:
            result = json.loads(messages[-1]["content"])
        except (ValueError, KeyError, TypeError):
            raise ModelError("Деморежим получил неожиданный результат инструмента.") from None
        if not result.get("ok"):
            return {"role": "assistant", "content": "Поручение не выполнено: "
                    + result.get("error", "инструмент сообщил об ошибке.")}
        previous = messages[-1]["tool_call_id"]
        if previous == "demo_list":
            return self._call("demo_search", "search_text", query=self.search_query)
        if previous == "demo_search":
            if not result["matches"]:
                message = "Совпадений в просмотренной части проекта не найдено."
                if result["truncated"]:
                    message += " Поиск был ограничен."
                return {"role": "assistant", "content": message}
            hit = result["matches"][0]
            return self._call("demo_read", "read_file", path=hit["path"], start_line=hit["line"], max_lines=1)
        if previous == "demo_read":
            if self.search_query is not None:
                if self.search_query not in result["content"]:
                    return {"role": "assistant", "content": "При повторном чтении совпадение не подтвердилось."}
                line = result["content"].rstrip("\r\n")
                report = ("# Первое подтверждённое совпадение\n\n"
                          "Сценарное демо. MCP-поиск указал строку, прочитанную ниже.\n\n"
                          f"> {line}\n")
                return self._call("demo_save", "save_report", content=report)
            excerpt = result["content"][:3000]
            quoted = "\n".join("> " + line for line in excerpt.splitlines())
            report = ("# Учебный отчёт\n\n"
                      "Режим: сценарное демо, нейросеть не подключалась.\n"
                      "Ниже извлечён фрагмент исходного документа; это не ИИ-анализ.\n\n"
                      f"Источник: `{self.source}`\n\n{quoted}\n")
            return self._call("demo_save", "save_report", content=report)
        if previous != "demo_save":
            raise ModelError("Неожиданный шаг демонстрационного сценария.")
        if self.search_query is not None:
            return {"role": "assistant", "content": "Демо завершено. Совпадение проверено, отчёт сохранён."}
        return {"role": "assistant", "content":
                "Демо завершено: файл прочитан, учебный отчёт сохранён. "
                "Оба действия выполнены настоящими файловыми инструментами."}
