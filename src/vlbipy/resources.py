"""What this process may use of the machine it runs on: CPUs and memory.

vlbipy runs on laptops (macOS, Linux), workstations and cluster nodes, often inside a limit set by
something else: a Docker container, a Slurm job, a ``systemd-run -p MemoryMax=`` unit. Those limits are
cgroups (Linux), and a process that only looks at the machine sees all of its cores and memory, starts
too many workers and is killed by the out-of-memory killer of its cgroup. So both figures here are the
smallest of what the machine has and what every cgroup above this process allows.

- :func:`usable_cpus`: the CPUs of the affinity mask, capped by a cgroup CPU quota.
- :func:`available_memory`: memory new work can take without swapping: the system's (Linux
  ``MemAvailable``; macOS ``psutil`` when installed, else ``vm_stat``) and, under a cgroup, the room left
  below its limit (page cache it could drop counts as room).

Nothing here raises: an unreadable file or an unknown platform gives the machine-wide figure, or None
for memory when even that is unknown (callers then plan by CPUs alone).
"""
from __future__ import annotations

import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

from .logging_utils import get_logger

logger = get_logger()

#: Roots of the files read here (tests point them at a fake tree).
_PROC_ROOT = Path("/proc")
_CGROUP_ROOT = Path("/sys/fs/cgroup")

#: A cgroup v1 memory limit at or above this means "no limit" (the kernel reports ~2**63 rounded to pages).
_V1_UNLIMITED = 1 << 60

_SIZE_SUFFIXES = {"": 1, "B": 1, "K": 1 << 10, "M": 1 << 20, "G": 1 << 30, "T": 1 << 40}


def parse_size(text: str) -> int:
    """Bytes in ``text``: a number with an optional K/M/G/T suffix (binary units), e.g. ``"1.5G"``."""
    match = re.fullmatch(r"\s*([0-9]*\.?[0-9]+)\s*([KMGT]?)i?B?\s*", str(text), flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"not a memory size: {text!r} (expected e.g. 512M, 1.5G)")
    return int(float(match.group(1)) * _SIZE_SUFFIXES[match.group(2).upper()])


def _read(path: Path) -> Optional[str]:
    """Contents of ``path`` stripped, or None when it cannot be read."""
    try:
        return path.read_text().strip()
    except (OSError, UnicodeDecodeError):
        return None


def _int(text: Optional[str]) -> Optional[int]:
    """``text`` as an int; None for missing text, "max" or anything else that is not a number."""
    try:
        return int(text) if text is not None else None
    except ValueError:
        return None


def _stat_value(path: Path, key: str) -> int:
    """Value of ``key`` in a cgroup ``memory.stat`` file (0 when absent)."""
    for line in (_read(path) or "").splitlines():
        name, _, value = line.partition(" ")
        if name == key:
            return _int(value) or 0
    return 0


# -- cgroups (Linux) --

def _cgroup_dirs(controller: str) -> list[Path]:
    """The cgroup directories of this process for ``controller``, its own first, then each parent up to the
    root of what is mounted (where a container's limit is). Empty outside Linux or without cgroups."""
    text = _read(_PROC_ROOT / "self" / "cgroup")
    if not text:
        return []
    for line in text.splitlines():
        hierarchy, _, rest = line.partition(":")
        controllers, _, path = rest.partition(":")
        if hierarchy == "0" and controllers == "":                       # cgroup v2 (unified)
            base = _CGROUP_ROOT
        elif controller in controllers.split(","):                       # cgroup v1
            base = next((_CGROUP_ROOT / name for name in (controllers, controller)
                         if (_CGROUP_ROOT / name).is_dir()), None)
            if base is None:
                continue
        else:
            continue
        dirs = []
        relative = Path(path.lstrip("/"))
        # In a container the path names the host's hierarchy, which is not mounted: start from what is.
        while relative != Path(".") and not (base / relative).is_dir():
            relative = relative.parent
        while True:
            dirs.append(base / relative)
            if relative == Path("."):
                return dirs
            relative = relative.parent
    return []


def cgroup_memory_room() -> Optional[int]:
    """Bytes this process's cgroups still let it take before their memory limit, or None when no cgroup
    above it has one. Page cache they could drop (``inactive_file``) counts as room."""
    room = None
    for directory in _cgroup_dirs("memory"):
        if (directory / "memory.max").exists():                         # v2
            limits = [_int(_read(directory / name)) for name in ("memory.max", "memory.high")]
            used = _int(_read(directory / "memory.current"))
            reclaimable = _stat_value(directory / "memory.stat", "inactive_file")
        else:                                                            # v1
            limits = [_int(_read(directory / "memory.limit_in_bytes"))]
            limits = [value for value in limits if value is not None and value < _V1_UNLIMITED]
            used = _int(_read(directory / "memory.usage_in_bytes"))
            reclaimable = _stat_value(directory / "memory.stat", "total_inactive_file")
        limits = [value for value in limits if value is not None]
        if not limits or used is None:
            continue
        here = max(0, min(limits) - max(0, used - reclaimable))
        room = here if room is None else min(room, here)
    return room


def cgroup_cpu_limit() -> Optional[float]:
    """CPUs' worth of time this process's cgroups allow (quota / period, the tightest of them), or None."""
    limit = None
    for directory in _cgroup_dirs("cpu"):
        quota = period = None
        text = _read(directory / "cpu.max")
        if text is not None:                                             # v2: "<quota|max> <period>"
            parts = text.split()
            if len(parts) == 2:
                quota, period = _int(parts[0]), _int(parts[1])
        else:                                                            # v1
            quota = _int(_read(directory / "cpu.cfs_quota_us"))
            period = _int(_read(directory / "cpu.cfs_period_us"))
            if quota is not None and quota < 0:
                quota = None
        if quota and period:
            limit = quota / period if limit is None else min(limit, quota / period)
    return limit


# -- the machine --

def _system_available_memory() -> Optional[int]:
    """Bytes the operating system can give new work without swapping, or None when unknown."""
    text = _read(_PROC_ROOT / "meminfo")
    if text:                                                             # Linux
        for line in text.splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    try:                                                                 # macOS, and Linux without /proc
        import psutil  # noqa: WPS433 - optional, the most accurate where it is installed
        return int(psutil.virtual_memory().available)
    except Exception:  # noqa: BLE001 - not installed or not supported here
        pass
    if sys.platform == "darwin":
        return _vm_stat_available()
    return None


def _vm_stat_available(output: Optional[str] = None) -> Optional[int]:
    """macOS: free plus inactive pages of ``vm_stat`` (what psutil reports as available), in bytes."""
    if output is None:
        try:
            output = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=10, check=True).stdout
        except (OSError, subprocess.SubprocessError) as exc:
            logger.debug("vm_stat failed: {}", exc)
            return None
    page = re.search(r"page size of (\d+) bytes", output)
    pages = dict(re.findall(r"^Pages ([a-z ]+):\s+(\d+)\.", output, flags=re.MULTILINE))
    if not page or "free" not in pages:
        return None
    return (int(pages["free"]) + int(pages.get("inactive", 0))) * int(page.group(1))


def total_memory() -> Optional[int]:
    """Physical memory of the machine in bytes, or None when unknown."""
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        pass
    if sys.platform == "darwin":
        try:
            return int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True,
                                      timeout=10, check=True).stdout)
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    return None


def available_memory() -> Optional[int]:
    """Bytes of memory new work can take without swapping or reaching a cgroup limit, or None when unknown.

    The smaller of the system's available memory and the room left under this process's cgroups.
    """
    figures = [value for value in (_system_available_memory(), cgroup_memory_room()) if value is not None]
    return min(figures) if figures else None


def usable_cpus() -> int:
    """CPUs this process may keep busy: its affinity mask (all CPUs where the platform has none), capped by a
    cgroup CPU quota (a container or job given 4 CPUs of a 64-core node gets 4)."""
    if hasattr(os, "process_cpu_count"):                                 # Python 3.13+
        cpus = os.process_cpu_count() or 1
    else:
        try:
            cpus = len(os.sched_getaffinity(0))
        except AttributeError:                                           # macOS: no affinity masks
            cpus = os.cpu_count() or 1
    quota = cgroup_cpu_limit()
    if quota is not None:
        cpus = min(cpus, max(1, math.floor(quota)))
    return max(1, cpus)


def describe() -> str:
    """One line on what this process may use, for the logs."""
    available, total = available_memory(), total_memory()
    room = cgroup_memory_room()
    gb = (lambda size: "?" if size is None else f"{size / 2**30:.1f} GB")
    parts = [f"{usable_cpus()} usable CPU(s) of {os.cpu_count()}",
             f"{gb(available)} memory available of {gb(total)}"]
    if room is not None:
        parts.append(f"cgroup room {gb(room)}")
    return ", ".join(parts)
