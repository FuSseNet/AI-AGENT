"""Совместимый вход к рецепту обзора; инструкции теперь находятся в SKILL.md."""

from .skills import SkillRegistry


def project_task(focus: str | None = None) -> str:
    instructions = SkillRegistry().load("project-review").instructions
    if focus:
        instructions += "\nДополнительное пожелание владельца:\n" + focus
    return instructions
