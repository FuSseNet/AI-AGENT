"""Просмотр учебных заданий с выбранным фильтром и сортировкой."""

import argparse
import json
from pathlib import Path
import sys

from . import SAMPLE_TASKS, apply_view, load_view


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=Path(__file__).parent / "views")
    parser.add_argument("--tasks", type=Path, help="JSON-файл заданий. По умолчанию учебный пример.")
    parser.add_argument("--view", default="all", help="Имя опубликованного представления или all")
    args = parser.parse_args(argv)
    try:
        tasks = json.loads(args.tasks.read_text(encoding="utf-8-sig")) if args.tasks else SAMPLE_TASKS
        selected = apply_view(tasks, load_view(args.catalog, args.view))
    except (OSError, ValueError) as error:
        print(f"Ошибка просмотра: {error}", file=sys.stderr)
        return 1
    for task in selected:
        print(f"{task['due']}  {task['subject']}  {task['title']}  [{task['status']}]")
    if not selected:
        print("Заданий нет.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
