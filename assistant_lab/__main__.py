"""Запуск: python -m assistant_lab --demo."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import stat
import sys
from uuid import uuid4

from .agent import AgentRunner
from .config import ConfigError, load_env, positive_int
from .events import EventLog
from .providers import DemoModel, LocalChatModel, ModelError
from .skills import SkillError, SkillRegistry
from .skill_tools import SkillToolSet
from .tools import FileTools


ROOT = Path(__file__).resolve().parent.parent


def _directory(path: Path) -> None:
    """Создать служебный каталог, отклоняя существующие reparse points."""
    for part in reversed([path, *path.parents]):
        if part.exists() or part.is_symlink():
            info = part.lstat()
            if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
                raise ConfigError("Служебные каталоги не должны быть ссылками/junction.")
            if not part.is_dir():
                raise ConfigError("На месте служебного каталога находится файл.")
        else:
            part.mkdir()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Учебный ассистент: поручения со скиллами, обзор проекта и сравнение документов.")
    result.add_argument("task", nargs="?", help="Поручение для настоящей модели")
    modes = result.add_mutually_exclusive_group()
    modes.add_argument("--demo", action="store_true", help="Сценарная имитация без модели и сети")
    modes.add_argument("--check", action="store_true", help="Проверить API и показать доступные модели")
    modes.add_argument("--describe-project", action="store_true",
                       help="Явно выбрать скилл project-review (прежняя команда обзора)")
    modes.add_argument("--skill", metavar="NAME", help="Явно выбрать скилл по имени")
    modes.add_argument("--list-skills", action="store_true", help="Показать скиллы без модели и сети")
    result.add_argument("--project", type=Path, default=ROOT / "workspace" / "project",
                        help="Корень разрешённых документов и исходников для поручения")
    result.add_argument("--source", default="README.md", help="Исходный файл для стандартного поручения")
    result.add_argument("--env-file", type=Path, default=ROOT / ".env", help="Файл настроек")
    result.add_argument("--base-url", help="Локальный адрес API, включая /v1")
    result.add_argument("--model", help="Точное имя модели из локального API")
    result.add_argument("--max-steps", help="Максимум запросов к модели (по умолчанию 8)")
    result.add_argument("--max-tool-calls", help="Максимум вызовов инструментов (по умолчанию 8)")
    result.add_argument("--read-budget", help="Общий бюджет чтения для скиллов, байты (по умолчанию 16384)")
    result.add_argument("--timeout", help="Таймаут HTTP-запроса в секундах (по умолчанию 60)")
    return result


def main(argv: list[str] | None = None) -> int:
    arg_parser = parser()
    args = arg_parser.parse_args(argv)
    if args.task and (args.demo or args.check or args.list_skills):
        arg_parser.error("Поручение задаётся только в режиме настоящей модели.")
    log = None
    try:
        if args.list_skills:
            for skill in SkillRegistry().list_skills():
                print(f"{skill.name}: {skill.description}")
            return 0
        skill_mode = bool(args.task or args.skill or args.describe_project)
        config = load_env(args.env_file)
        secret = config.get("ASSISTANT_API_KEY", "")
        if any(char in secret for char in "\r\n"):
            raise ConfigError("ASSISTANT_API_KEY не должен содержать переносы строк.")
        max_steps = positive_int(args.max_steps or config.get("ASSISTANT_MAX_STEPS", "8"),
                                 "max_steps", 50)
        max_calls = positive_int(args.max_tool_calls or config.get("ASSISTANT_MAX_TOOL_CALLS", "8"),
                                 "max_tool_calls", 50)
        read_budget = positive_int(args.read_budget or config.get("ASSISTANT_READ_BUDGET", "16384"),
                                   "read_budget", 1_048_576)
        if args.demo:
            model = DemoModel(args.source)
            print("ДЕМО: сценарная имитация. Нейросеть и сеть не используются.")
        else:
            base_url = args.base_url or config.get("ASSISTANT_BASE_URL", "")
            model_name = args.model or config.get("ASSISTANT_MODEL", "")
            if not base_url:
                raise ConfigError("Не задан ASSISTANT_BASE_URL. "
                                  "Скопируйте .env.example в .env и укажите адрес API из Unsloth. "
                                  "Для запуска без модели используйте --demo.")
            if not model_name and not args.check:
                raise ConfigError("Не задан ASSISTANT_MODEL. "
                                  "Узнайте имя модели командой python -m assistant_lab --check.")
            timeout = positive_int(args.timeout or config.get("ASSISTANT_TIMEOUT", "60"),
                                   "timeout", 600)
            max_tokens = positive_int(config.get("ASSISTANT_MAX_TOKENS", "2048"),
                                      "max_tokens", 16_384)
            model = LocalChatModel(base_url, model_name, secret, timeout, max_tokens)
            if args.check:
                models = model.list_models()
                if not models:
                    print("API доступен, но список моделей пуст. Загрузите модель в Unsloth.")
                    return 1
                print("API доступен. Имена моделей для ASSISTANT_MODEL:")
                for name in models:
                    print("  " + name)
                print("Поддержку инструментов проверяет запуск стандартного поручения.")
                return 0
            print("Режим: настоящая локальная модель через API.")

        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
        report_dir = ROOT / "workspace" / "reports" / run_id
        logs_dir = ROOT / "runs"
        _directory(report_dir)
        _directory(logs_dir)
        report_path = report_dir / "report.md"
        log_path = logs_dir / f"{run_id}.jsonl"
        files = FileTools(args.project, report_path,
                          required_source=args.source if not skill_mode else None,
                          project_mode=skill_mode,
                          max_total_read_bytes=read_budget)
        log = EventLog(log_path, secrets=[secret] if secret else [])
        skill_tools = None
        if skill_mode:
            selected = "project-review" if args.describe_project else args.skill
            skill_tools = SkillToolSet(files, SkillRegistry(), log, selected_name=selected)
            tools = skill_tools
            task = args.task or ("Подготовь обзор выбранного проекта и сохрани отчёт."
                                 if selected == "project-review" else "Выполни выбранный скилл.")
        else:
            tools = files
            task = (
                f"Прочитай файл {args.source!r} с помощью read_file. "
                "Составь краткое описание назначения проекта, его возможностей и ограничений, "
                "опираясь только на прочитанный документ. Сохрани отчёт через save_report. "
                "Если файл недоступен, сообщи об ошибке и не выдумывай содержимое.")
        runner = AgentRunner(model, tools, log, max_steps=max_steps,
                             max_tool_calls=max_calls, on_event=print,
                             system_prompt=skill_tools.system_prompt if skill_tools else None)
        result = runner.run(task)
        print("\n" + result.final_text)
        print(f"Запросов к модели: {result.steps}; вызовов инструментов: {result.tool_calls}.")
        print(f"Журнал: {log_path}")
        if skill_tools:
            print("Выбранный скилл: " + (skill_tools.selected_name or "не выбран"))
            print("Фактически прочитанные файлы:")
            for path in files.read_paths:
                print("  " + path)
            if not files.read_paths:
                print("  Ни одного файла не прочитано.")
        if files.report_saved and report_path.is_file():
            print(f"Отчёт: {report_path}")
        else:
            print("Отчёт не создан: поручение ещё не выполнено; возможно, требуется уточнение входных данных.")
            return 1
        return 0 if result.status == "completed" else 1
    except (ConfigError, ModelError, SkillError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nВыполнение остановлено пользователем.", file=sys.stderr)
        return 130
    except (OSError, ValueError):
        print("Ошибка: не удалось подготовить или выполнить запуск. "
              "Проверьте настройки, каталог проекта и права на служебные файлы.", file=sys.stderr)
        return 2
    finally:
        if log is not None and callable(getattr(log, "close", None)):
            log.close()


if __name__ == "__main__":
    raise SystemExit(main())
