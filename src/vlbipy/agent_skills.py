"""The VLBI data-reduction skills that ship with vlbipy, and their installation for coding agents.

A skill is a directory with a ``SKILL.md`` (YAML front matter with ``name`` and ``description``, then Markdown) and
optional ``references/`` (the open Agent Skills layout read by Devin, Claude Code, Codex, Cursor, Copilot, ...). They
live in ``vlbipy/agent/skills/`` so the installed copy always matches the installed vlbipy; :func:`install` copies (or links)
them to where an agent looks for skills, and :func:`status` says whether an installed copy is older than the package.
``vlbipy/agent/`` is also an agent plugin (Devin ``.devin-plugin``, Claude Code ``.claude-plugin``) with the skills in
its standard ``skills/`` directory, installable straight from the repository.
"""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ._version import __version__
from .logging_utils import get_logger

logger = get_logger()

#: Directory of the shipped skills (package data).
SKILLS_DIR = Path(__file__).resolve().parent / "agent" / "skills"

#: File written into an installed copy: which vlbipy version it came from. Its presence marks the copy as ours.
MARKER = ".vlbipy-skill.json"

#: Where each agent reads skills: (global directory, directory inside a project). "agents" is the open standard
#: (~/.agents/skills, .agents/skills) read by Codex, Devin and others.
AGENT_DIRS: dict[str, tuple[str, str]] = {
    "agents": ("~/.agents/skills", ".agents/skills"),
    "devin": ("~/.config/devin/skills", ".devin/skills"),
    "claude": ("~/.claude/skills", ".claude/skills"),
    "cursor": ("~/.cursor/skills", ".cursor/skills"),
    "copilot": ("~/.copilot/skills", ".github/skills"),
    "windsurf": ("~/.codeium/windsurf/skills", ".windsurf/skills"),
}


@dataclass(frozen=True)
class Skill:
    """One shipped skill: its ``name`` and ``description`` (front matter) and its directory."""

    name: str
    description: str
    path: Path


def read_frontmatter(skill_md: Path) -> dict[str, str]:
    """Return the ``key: value`` pairs of the YAML front matter of a ``SKILL.md`` ({} without front matter).

    Only flat one-line scalars are read, which is all the skill format needs (``name``, ``description``).
    """
    lines = skill_md.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    fields = {}
    for line in lines[1:]:
        if line.strip() == "---":
            break
        key, sep, value = line.partition(":")
        if sep and not line.startswith((" ", "\t")):
            fields[key.strip()] = value.strip().strip("'\"")
    return fields


def available() -> list[Skill]:
    """Return the shipped skills, sorted by name."""
    skills = []
    for skill_md in sorted(SKILLS_DIR.glob("*/SKILL.md")):
        meta = read_frontmatter(skill_md)
        skills.append(Skill(name=meta.get("name", skill_md.parent.name), description=meta.get("description", ""),
                            path=skill_md.parent))
    return skills


def get(name: str) -> Skill:
    """Return the shipped skill ``name``; raises KeyError listing the known ones."""
    for skill in available():
        if skill.name == name:
            return skill
    raise KeyError(f"no skill {name!r}; available: {', '.join(s.name for s in available())}")


def target_dir(agent: str = "agents", project: Optional[str] = None) -> Path:
    """Return the skills directory of ``agent``: global (home), or inside the project directory ``project``.

    On Windows Devin's global directory is ``%APPDATA%\\devin\\skills``.
    """
    if agent not in AGENT_DIRS:
        raise ValueError(f"unknown agent {agent!r}; choose one of {', '.join(AGENT_DIRS)}")
    global_dir, project_dir = AGENT_DIRS[agent]
    if project:
        return Path(project).expanduser().resolve() / project_dir
    if agent == "devin" and os.name == "nt" and os.environ.get("APPDATA"):
        return Path(os.environ["APPDATA"]) / "devin" / "skills"
    return Path(global_dir).expanduser()


def _is_ours(dest: Path) -> bool:
    """True when ``dest`` is a copy made by :func:`install` or a link into :data:`SKILLS_DIR`."""
    if dest.is_symlink():
        return SKILLS_DIR in dest.resolve().parents
    return (dest / MARKER).is_file()


def installed_state(dest: Path) -> str:
    """Return the state of one installed skill directory.

    ``missing``, ``linked`` (a link into this vlbipy), ``current`` (a copy of this version), ``outdated <version>``
    (a copy from another vlbipy version) or ``foreign`` (a skill of that name not installed by vlbipy).
    """
    if not dest.exists() and not dest.is_symlink():
        return "missing"
    if dest.is_symlink():
        return "linked" if _is_ours(dest) else "foreign"
    if not (dest / MARKER).is_file():
        return "foreign"
    version = json.loads((dest / MARKER).read_text(encoding="utf-8")).get("vlbipy_version", "?")
    return "current" if version == __version__ else f"outdated {version}"


def status(agent: str = "agents", project: Optional[str] = None) -> list[tuple[Skill, str]]:
    """Return ``(skill, state)`` for every shipped skill in the skills directory of ``agent`` (see :func:`installed_state`)."""
    base = target_dir(agent, project)
    return [(skill, installed_state(base / skill.name)) for skill in available()]


def install(names: Optional[list[str]] = None, *, agent: str = "agents", project: Optional[str] = None,
            link: bool = False, force: bool = False) -> list[Path]:
    """Install the shipped skills (all, or ``names``) where ``agent`` reads them; returns the installed paths.

    A copy (default) carries :data:`MARKER` with the vlbipy version; ``link`` makes symbolic links into the package
    instead, which follow upgrades of an editable install. An earlier vlbipy installation of a skill is replaced; a
    skill of the same name that vlbipy did not install is left alone unless ``force`` (then it is replaced).
    """
    skills = [get(name) for name in names] if names else available()
    base = target_dir(agent, project)
    base.mkdir(parents=True, exist_ok=True)
    installed = []
    for skill in skills:
        dest = base / skill.name
        if dest.exists() or dest.is_symlink():
            if not (_is_ours(dest) or force):
                raise FileExistsError(f"{dest} exists and was not installed by vlbipy; use --force to replace it")
            if dest.is_symlink() or dest.is_file():
                dest.unlink()
            else:
                shutil.rmtree(dest)
        if link:
            dest.symlink_to(skill.path, target_is_directory=True)
        else:
            shutil.copytree(skill.path, dest, ignore=shutil.ignore_patterns("__pycache__", ".DS_Store"))
            (dest / MARKER).write_text(json.dumps({"vlbipy_version": __version__, "source": str(skill.path),
                                                   "installed": datetime.now(timezone.utc).isoformat(timespec="seconds")},
                                                  indent=2), encoding="utf-8")
        logger.info("skills: {} {} -> {}", "linked" if link else "copied", skill.name, dest)
        installed.append(dest)
    return installed
