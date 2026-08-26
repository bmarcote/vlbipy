"""Step-state tracking and smart re-run decisions, persisted across runs.

The state lives in ``<work_dir>/.pipeline_state.json`` so a pipeline that was
interrupted — or that a user wants to extend one step at a time — resumes where
it stopped instead of redoing hours of calibration. Without persistence every
invocation would start from nothing, which is both slow and destructive: the
import step resets the data, so a blind re-run throws away the calibration the
previous run produced.

Three ways to control what runs:

* default — skip steps already recorded as done, run the rest;
* ``from_step`` — invalidate that step and everything after it, then run;
* ``scratch`` — forget all state and start over (the caller also resets the data).
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Optional

from .logging_utils import get_logger

logger = get_logger()

#: File name holding the persisted state inside the working directory.
STATE_FILENAME = ".pipeline_state.json"


class StepState:
    """Track completion of pipeline steps for one observation.

    Parameters
    ----------
    project_code : str
        Owning project code (used in log messages).
    work_dir : str or pathlib.Path, optional
        Directory holding the state file. When omitted the state is in-memory
        only, which is what the dummy backend and unit tests want.
    """

    def __init__(self, project_code: str = "", work_dir=None) -> None:
        self.project_code = project_code
        self.work_dir = Path(work_dir) if work_dir else None
        self._steps: dict[str, dict] = {}
        if self.work_dir is not None:
            self.load()

    @property
    def path(self) -> Optional[Path]:
        """Path of the state file, or ``None`` for in-memory state."""
        return (self.work_dir / STATE_FILENAME) if self.work_dir is not None else None

    # -- persistence --
    def load(self) -> None:
        """Read the state file if it exists; a corrupt file is reported, not fatal."""
        path = self.path
        if path is None or not path.is_file():
            return
        try:
            self._steps = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("[{}] could not read {} ({}); starting with empty state",
                           self.project_code, path, exc)
            return
        done = [name for name, entry in self._steps.items() if entry.get("status") == "done"]
        logger.info("[{}] resuming: {} step(s) already complete ({})", self.project_code,
                    len(done), ", ".join(done) or "none")

    def save(self) -> None:
        """Write the state file (no-op for in-memory state)."""
        path = self.path
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(self._steps, indent=2, sort_keys=True))
        except OSError as exc:
            logger.warning("[{}] could not write {} ({}); progress will not be remembered",
                           self.project_code, path, exc)

    # -- decisions --
    def should_run(self, step: str, force: bool = False) -> bool:
        """Decide whether a step should run, logging the decision.

        Parameters
        ----------
        step : str
            Step name.
        force : bool
            If True, always run (bypass smart-skip).

        Returns
        -------
        bool
            True if the step should run; False if it can be skipped.
        """
        done = self._steps.get(step, {}).get("status") == "done"
        if done and not force:
            logger.info("[{}] skipping: {} (already done)", self.project_code, step)
            return False
        logger.info("[{}] running: {}{}", self.project_code, step,
                    " (forced)" if force and done else "")
        return True

    def mark_complete(self, step: str, outputs=None, inputs=None) -> None:
        """Mark a step complete, recording optional inputs/outputs."""
        self._steps[step] = {
            "status": "done",
            "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
            "outputs": [str(o) for o in (outputs or [])],
            "inputs": [str(i) for i in (inputs or [])],
        }
        self.save()

    def mark_failed(self, step: str, error: str) -> None:
        """Mark a step as failed with an error message.

        A failed step is recorded rather than forgotten so the next run re-runs
        it, and so the failure is visible in the state file afterwards.
        """
        self._steps[step] = {
            "status": "failed",
            "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
            "error": str(error),
        }
        self.save()

    def invalidate_downstream(self, step: str, ordered_steps: list[str]) -> None:
        """Invalidate ``step`` and every step after it in ``ordered_steps``.

        Anything derived from a step's outputs is stale once that step re-runs,
        so resuming from the middle has to forget the later steps too.
        """
        if step not in ordered_steps:
            if self._steps.pop(step, None) is not None:
                logger.info("[{}] invalidated: {}", self.project_code, step)
            self.save()
            return
        for name in ordered_steps[ordered_steps.index(step):]:
            if self._steps.pop(name, None) is not None:
                logger.info("[{}] invalidated: {}", self.project_code, name)
        self.save()

    def status(self, step: str) -> Optional[str]:
        """Return the recorded status of a step, or ``None`` if unknown."""
        return self._steps.get(step, {}).get("status")

    def completed(self) -> list[str]:
        """Return the names of the steps recorded as done."""
        return [name for name, entry in self._steps.items() if entry.get("status") == "done"]

    def reset(self) -> None:
        """Clear all recorded state, on disk as well as in memory."""
        self._steps.clear()
        path = self.path
        if path is not None and path.is_file():
            path.unlink()
            logger.info("[{}] removed {}", self.project_code, path.name)

    def as_dict(self) -> dict:
        """Return a deep-ish copy of the raw state dict."""
        return {k: dict(v) for k, v in self._steps.items()}
