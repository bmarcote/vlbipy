"""The agent skills shipped in vlbipy/agent/skills: format, consistency with the code, installation, plugins.

The skills describe vlbipy's commands, options, steps and settings in prose; these tests fail when one of those no
longer exists, so the skills cannot silently drift from the code. They also keep project codes, target names and
local paths out of the public skills.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import re
import tomllib
from pathlib import Path

import pytest

from vlbipy import agent_skills
from vlbipy.cli import build_parser, main
from vlbipy.observation import PER_SOURCE_STEP_PREFIXES, PER_SOURCE_STEPS, STEP_ORDER

PACKAGE = Path(agent_skills.__file__).resolve().parent
REPO = PACKAGE.parent.parent
SKILL_FILES = sorted(agent_skills.SKILLS_DIR.glob("*/SKILL.md"))
ALL_FILES = sorted(p for p in agent_skills.SKILLS_DIR.rglob("*.md"))
VLBIPY_SKILL = agent_skills.SKILLS_DIR / "reduce-vlbi-vlbipy" / "SKILL.md"

#: Size budgets (bytes): a SKILL.md is loaded whole into the agent's context; references only when needed.
SKILL_BUDGET = 30_000
REFERENCE_BUDGET = 12_000

#: Experiment codes (EVN/global/VLBA/LBA style) and J2000 source names that must not appear in public skills.
PROJECT_CODE = re.compile(r"\b(E[A-Z]\d{3}[A-Z]?|G[A-Z]\d{3}[A-Z]?|RSM\d{2}[A-Z]?|N\d{2}[A-Z]\d|V\d{3}[A-Z]?|"
                          r"B[A-Z]\d{3}[A-Z]?)\b", re.IGNORECASE)
J2000_NAME = re.compile(r"\bJ\d{4}[+-]\d{2,4}\b")
LOCAL_PATH = re.compile(r"/Users/|/home/\w|Programing|other_pipelines")


def _parser_tables() -> tuple[set[str], set[str]]:
    """Return (subcommand names, every option string of every subcommand) of the vlbipy CLI."""
    parser = build_parser()
    commands, options = set(), {"--help", "--version"}
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for name, subparser in action.choices.items():
                commands.add(name)
                for sub_action in subparser._actions:
                    options.update(sub_action.option_strings)
    return commands, options


def _defaults() -> dict:
    """The built-in configuration defaults."""
    return tomllib.loads((PACKAGE / "templates" / "defaults.toml").read_text(encoding="utf-8"))


def _has_key(config: dict, section: str, key: str) -> bool:
    """True when ``[section]`` (dotted) holds ``key`` in the defaults."""
    node = config
    for part in section.split("."):
        if not isinstance(node, dict) or part not in node:
            return False
        node = node[part]
    return isinstance(node, dict) and key in node


@pytest.mark.parametrize("skill_md", SKILL_FILES, ids=lambda p: p.parent.name)
def test_front_matter(skill_md):
    meta = agent_skills.read_frontmatter(skill_md)
    assert meta.get("name") == skill_md.parent.name
    assert re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", meta["name"])
    assert 50 < len(meta.get("description", "")) <= 1024


def test_the_four_skills_ship():
    assert [s.name for s in agent_skills.available()] == ["reduce-vlbi-aips", "reduce-vlbi-casa", "reduce-vlbi-data",
                                                          "reduce-vlbi-vlbipy"]


@pytest.mark.parametrize("path", ALL_FILES, ids=lambda p: str(p.relative_to(agent_skills.SKILLS_DIR)))
def test_size_budget_and_public_content(path):
    text = path.read_text(encoding="utf-8")
    assert len(text.encode()) <= (SKILL_BUDGET if path.name == "SKILL.md" else REFERENCE_BUDGET)
    assert not PROJECT_CODE.findall(text), "project codes in a public skill"
    assert not J2000_NAME.findall(text), "source names in a public skill (use placeholders)"
    assert not LOCAL_PATH.findall(text), "local paths in a public skill"


def test_references_named_in_a_skill_exist():
    for skill_md in SKILL_FILES:
        for other, name in re.findall(r"(?:(reduce-vlbi-[a-z]+)/)?references/([a-z-]+\.md)",
                                      skill_md.read_text(encoding="utf-8")):
            skill_dir = agent_skills.SKILLS_DIR / other if other else skill_md.parent
            assert (skill_dir / "references" / name).is_file(), f"{skill_md.parent.name}: {other}/{name}"


def test_cross_references_name_existing_skills():
    names = {s.name for s in agent_skills.available()}
    for path in ALL_FILES:
        for name in re.findall(r"/(reduce-vlbi-[a-z]+)", path.read_text(encoding="utf-8")):
            assert name in names, f"{path.name} refers to /{name}"


def test_vlbipy_commands_and_options_exist():
    commands, options = _parser_tables()
    text = VLBIPY_SKILL.read_text(encoding="utf-8") + (REPO / "docs" / "usage" / "agents.md").read_text(encoding="utf-8")
    code = "\n".join(re.findall(r"```[a-z]*\n(.*?)```", text, re.DOTALL) + re.findall(r"`([^`\n]+)`", text))
    for command in re.findall(r"\bvlbipy ([a-z_]+)\b", code):
        assert command in commands, f"vlbipy {command}"
    for option in set(re.findall(r"(?<![\w-])(--[a-z][a-z-]+)", text)):
        assert option in options, option


def test_vlbipy_steps_exist():
    text = VLBIPY_SKILL.read_text(encoding="utf-8")
    section = text.split("## Pipeline steps", 1)[1].split("\n## ", 1)[0]
    rows = [line for line in section.splitlines() if line.startswith("| `")]
    names = [name for row in rows for name in re.findall(r"`([a-z0-9_]+)`", row.split("|")[1])]
    assert len(names) >= 25
    assert set(names) <= set(STEP_ORDER)
    assert set(STEP_ORDER) <= set(names), "every pipeline step is described"


def test_step_contract_names_exist():
    text = (agent_skills.SKILLS_DIR / "reduce-vlbi-data" / "SKILL.md").read_text(encoding="utf-8")
    section = text.split("## Step contract", 1)[1].split("\n## ", 1)[0]
    allowed = set(STEP_ORDER) | set(PER_SOURCE_STEPS) | {"combine", "report"}
    for row in section.splitlines()[2:]:
        if not row.startswith("|"):
            continue
        for name in re.findall(r"`([a-z0-9_<>]+)`", row.split("|")[2]):
            assert name in allowed or name.startswith(PER_SOURCE_STEP_PREFIXES), name


def test_vlbipy_settings_exist():
    config = _defaults()
    checked = 0
    for line in VLBIPY_SKILL.read_text(encoding="utf-8").splitlines():
        for cell in line.split("|") if line.startswith("|") else [line]:
            match = re.search(r"`\[([a-z_.]+)\][ .]?([a-z_]+)`", cell)
            if not match:
                continue
            section = match.group(1)
            keys = [match.group(2)] + re.findall(r"`([a-z_]+)`", cell[match.end():]) if line.startswith("|") \
                else [match.group(2)]
            for key in keys:
                assert _has_key(config, section, key), f"[{section}] {key}"
                checked += 1
    assert checked >= 15


def test_install_copy_status_and_replace(tmp_path):
    project = str(tmp_path)
    paths = agent_skills.install(project=project)
    assert [p.name for p in paths] == [s.name for s in agent_skills.available()]
    assert all(state == "current" for _, state in agent_skills.status(project=project))
    assert (paths[0] / "SKILL.md").read_text() == (agent_skills.get(paths[0].name).path / "SKILL.md").read_text()
    marker = paths[0] / agent_skills.MARKER
    marker.write_text(json.dumps({"vlbipy_version": "0.0.1"}))
    assert agent_skills.installed_state(paths[0]) == "outdated 0.0.1"
    agent_skills.install([paths[0].name], project=project)        # an earlier vlbipy copy is replaced
    assert agent_skills.installed_state(paths[0]) == "current"


def test_install_never_overwrites_a_foreign_skill(tmp_path):
    foreign = agent_skills.target_dir("claude", str(tmp_path)) / "reduce-vlbi-data"
    foreign.mkdir(parents=True)
    (foreign / "SKILL.md").write_text("mine")
    with pytest.raises(FileExistsError):
        agent_skills.install(["reduce-vlbi-data"], agent="claude", project=str(tmp_path))
    assert (foreign / "SKILL.md").read_text() == "mine"
    agent_skills.install(["reduce-vlbi-data"], agent="claude", project=str(tmp_path), force=True)
    assert agent_skills.installed_state(foreign) == "current"


def test_install_link(tmp_path):
    (dest,) = agent_skills.install(["reduce-vlbi-casa"], agent="devin", project=str(tmp_path), link=True)
    assert dest == tmp_path.resolve() / ".devin" / "skills" / "reduce-vlbi-casa"
    assert dest.is_symlink() and agent_skills.installed_state(dest) == "linked"


def test_cli_skills(tmp_path, capsys):
    assert main(["skills", "path", "reduce-vlbi-data"]) == 0
    assert capsys.readouterr().out.strip().endswith("reduce-vlbi-data")
    assert main(["skills", "install", "--project", str(tmp_path), "--agent", "cursor"]) == 0
    assert (tmp_path / ".cursor" / "skills" / "reduce-vlbi-vlbipy" / "SKILL.md").is_file()
    assert main(["skills", "show", "no-such-skill"]) == 1


def test_package_data_covers_the_skills():
    globs = tomllib.loads((REPO / "pyproject.toml").read_text())["tool"]["setuptools"]["package-data"]["vlbipy"]
    for path in agent_skills.SKILLS_DIR.rglob("*"):
        if path.is_file() and path.name != ".DS_Store":
            relative = str(path.relative_to(PACKAGE))
            assert any(fnmatch.fnmatch(relative, pattern) for pattern in globs), relative


def test_plugin_manifests():
    plugin_root = agent_skills.SKILLS_DIR.parent
    for manifest in (".devin-plugin/plugin.json", ".claude-plugin/plugin.json"):
        data = json.loads((plugin_root / manifest).read_text())
        assert data["name"] == "vlbipy" and "skills" not in data     # the standard skills/ directory
    assert not (plugin_root / "AGENTS.md").exists(), "a plugin-root AGENTS.md is injected into every session"
    marketplace = json.loads((REPO / ".claude-plugin" / "marketplace.json").read_text())
    (entry,) = marketplace["plugins"]
    assert (REPO / entry["source"]).resolve() == plugin_root
