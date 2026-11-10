# AI coding agents (skills)

vlbipy ships **agent skills** on VLBI data reduction: Markdown instructions that coding agents (Devin, Claude Code,
Codex, Cursor, GitHub Copilot, Windsurf, ...) load when a task matches. They follow the open Agent Skills layout (a
directory with a `SKILL.md` and optional `references/`) and live in the package, `vlbipy/agent/skills/`, so they are
versioned with the code.

| Skill | Content |
|---|---|
| `reduce-vlbi-data` | the science: what each calibration and imaging step corrects, the decisions, how to judge the result, a diagnostics table (software-agnostic; detail in `references/`) |
| `reduce-vlbi-vlbipy` | running vlbipy (layout, configuration, CLI and Python, pipeline steps, resuming, AI mode) and reviewing a run (dashboard, comments, amplitude corrections) |
| `reduce-vlbi-casa` | the CASA commands for each step |
| `reduce-vlbi-aips` | the AIPS / ParselTongue recipes for each step |

All four use the same step numbers (S0-S21), so an agent can move from the science to the commands of any package.

## Installing them

```bash
vlbipy skills                                   # list them, and whether your agent's copy is current
vlbipy skills install                           # copy to ~/.agents/skills (Codex, Devin and others)
vlbipy skills install --agent claude            # ~/.claude/skills
vlbipy skills install --agent devin             # ~/.config/devin/skills
vlbipy skills install --project .               # this project's .agents/skills (commit it to share)
vlbipy skills install --link                    # links instead of copies (follow an editable install)
vlbipy skills show reduce-vlbi-data             # print a SKILL.md
vlbipy skills path                              # where the shipped skills are
```

Agents: `agents` (`~/.agents/skills`, default), `devin`, `claude`, `cursor`, `copilot`, `windsurf`. A copy records
the vlbipy version it came from; `vlbipy skills` reports `outdated` after an upgrade, and `install` replaces it.
A skill of the same name that vlbipy did not install is never overwritten without `--force`.

### As a plugin, without installing vlbipy

`src/vlbipy/agent/` is also a plugin for Devin and Claude Code:

```bash
devin plugins install bmarcote/vlbipy#src/vlbipy/agent       # Devin
```

```text
/plugin marketplace add bmarcote/vlbipy                      # Claude Code
/plugin install vlbipy@vlbipy
```

A plugin follows the repository rather than your installed version; prefer `vlbipy skills install` when you run
vlbipy itself.

## Keeping them true

`tests/test_agent_skills.py` checks that the skills' front matter is valid, that every `vlbipy` command, option,
pipeline step and configuration key they mention exists, that they hold no absolute local paths or project codes, and
that each `SKILL.md` stays within its size budget. Lessons learned while reducing data are written to `AGENTS.md`
(development notes) and, condensed and anonymised, to the skills (`reduce-vlbi-data/references/`).
