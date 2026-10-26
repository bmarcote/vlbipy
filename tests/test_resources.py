"""What vlbipy may use of the machine: CPUs and memory, under cgroup limits (containers, batch jobs)."""
from pathlib import Path

import pytest

from vlbipy import resources
from vlbipy.plot_jobs import plot_workers
from vlbipy.solvers import workers

GB = 1 << 30


def _write(root: Path, files: dict) -> None:
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


@pytest.fixture
def fake_host(tmp_path, monkeypatch):
    """A /proc and /sys/fs/cgroup of our own; returns ``(proc, cgroup)`` roots to fill."""
    proc, cgroup = tmp_path / "proc", tmp_path / "cgroup"
    proc.mkdir()
    cgroup.mkdir()
    monkeypatch.setattr(resources, "_PROC_ROOT", proc)
    monkeypatch.setattr(resources, "_CGROUP_ROOT", cgroup)
    monkeypatch.setattr(workers, "_CPUS", None)
    monkeypatch.delenv(workers.WORKERS_ENV, raising=False)
    monkeypatch.delenv(workers.WORKER_MEMORY_ENV, raising=False)
    return proc, cgroup


def test_parse_size():
    assert resources.parse_size("512M") == 512 << 20
    assert resources.parse_size("1.5G") == int(1.5 * GB)
    assert resources.parse_size("2GiB") == 2 * GB and resources.parse_size("4096") == 4096
    with pytest.raises(ValueError):
        resources.parse_size("lots")


def test_cgroup_v2_memory_limit_of_a_parent_applies(fake_host):
    proc, cgroup = fake_host
    _write(proc, {"self/cgroup": "0::/user.slice/job.service\n",
                  "meminfo": "MemTotal: 33554432 kB\nMemAvailable: 25165824 kB\n"})       # 24 GB free
    _write(cgroup, {
        # the unit itself has no limit; its parent slice allows 8 GB, 5 GB used of which 1 GB droppable cache
        "user.slice/job.service/memory.max": "max\n", "user.slice/job.service/memory.high": "max\n",
        "user.slice/job.service/memory.current": "1073741824\n", "user.slice/job.service/memory.stat": "",
        "user.slice/memory.max": f"{8 * GB}\n", "user.slice/memory.high": "max\n",
        "user.slice/memory.current": f"{5 * GB}\n", "user.slice/memory.stat": f"anon 1\ninactive_file {GB}\n"})
    assert resources.cgroup_memory_room() == 4 * GB                      # 8 - (5 - 1)
    assert resources.available_memory() == 4 * GB                        # tighter than the 24 GB of the system


def test_cgroup_v2_memory_high_counts_as_the_limit(fake_host):
    proc, cgroup = fake_host
    _write(proc, {"self/cgroup": "0::/a.service\n", "meminfo": "MemAvailable: 25165824 kB\n"})
    _write(cgroup, {"a.service/memory.max": f"{10 * GB}\n", "a.service/memory.high": f"{6 * GB}\n",
                    "a.service/memory.current": f"{2 * GB}\n", "a.service/memory.stat": "inactive_file 0\n"})
    assert resources.available_memory() == 4 * GB


def test_container_sees_only_its_own_cgroup_root(fake_host):
    # Inside Docker the path in /proc/self/cgroup names the host hierarchy, which is not mounted: the limit
    # is on the root of what is.
    proc, cgroup = fake_host
    _write(proc, {"self/cgroup": "0::/docker/abc123\n", "meminfo": "MemAvailable: 62914560 kB\n"})
    _write(cgroup, {"memory.max": f"{4 * GB}\n", "memory.current": f"{GB}\n", "memory.stat": "",
                    "cpu.max": "250000 100000\n"})
    assert resources.available_memory() == 3 * GB
    assert resources.cgroup_cpu_limit() == 2.5
    assert resources.usable_cpus() <= 2


def test_cgroup_v1_limits(fake_host):
    proc, cgroup = fake_host
    _write(proc, {"self/cgroup": "5:memory:/slurm/job1\n4:cpu,cpuacct:/slurm/job1\n",
                  "meminfo": "MemAvailable: 104857600 kB\n"})
    _write(cgroup, {"memory/slurm/job1/memory.limit_in_bytes": f"{16 * GB}\n",
                    "memory/slurm/job1/memory.usage_in_bytes": f"{6 * GB}\n",
                    "memory/slurm/job1/memory.stat": f"total_inactive_file {2 * GB}\n",
                    "memory/slurm/memory.limit_in_bytes": "9223372036854771712\n",            # unlimited
                    "memory/slurm/memory.usage_in_bytes": "1\n", "memory/memory.limit_in_bytes": "9223372036854771712\n",
                    "memory/memory.usage_in_bytes": "1\n",
                    "cpu,cpuacct/slurm/job1/cpu.cfs_quota_us": "400000\n",
                    "cpu,cpuacct/slurm/job1/cpu.cfs_period_us": "100000\n",
                    "cpu,cpuacct/slurm/cpu.cfs_quota_us": "-1\n", "cpu,cpuacct/slurm/cpu.cfs_period_us": "100000\n"})
    assert resources.available_memory() == 12 * GB                      # 16 - (6 - 2)
    assert resources.cgroup_cpu_limit() == 4.0


def test_no_cgroups_and_no_proc_falls_back_quietly(fake_host, monkeypatch):
    monkeypatch.setattr(resources, "_system_available_memory", lambda: None)
    assert resources.cgroup_memory_room() is None and resources.cgroup_cpu_limit() is None
    assert resources.available_memory() is None
    assert resources.usable_cpus() >= 1


def test_macos_vm_stat_available_memory():
    output = ("Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
              "Pages free:                               10000.\n"
              "Pages active:                            200000.\n"
              "Pages inactive:                          190000.\n"
              "Pages speculative:                         5000.\n"
              "Pages wired down:                         80000.\n")
    assert resources._vm_stat_available(output) == (10000 + 190000) * 16384
    assert resources._vm_stat_available("nothing useful") is None


def test_pool_size_follows_cpus_and_memory(fake_host, monkeypatch):
    monkeypatch.setattr(workers, "_CPUS", 20)
    monkeypatch.setattr(workers, "available_memory", lambda: 64 * GB)
    assert workers.default_workers() == 20                              # a workstation: CPU-bound, capped
    monkeypatch.setattr(workers, "available_memory", lambda: 23 * GB)
    assert workers.default_workers() == 17
    monkeypatch.setattr(workers, "available_memory", lambda: 6 * GB)
    assert workers.default_workers() == 4                                # 6 GB x 0.75 at 1 GB each
    monkeypatch.setattr(workers, "available_memory", lambda: GB // 2)
    assert workers.default_workers() == 1                                # never fewer than one
    monkeypatch.setattr(workers, "available_memory", lambda: None)
    assert workers.default_workers() == 20                              # memory unknown: by CPUs
    monkeypatch.setattr(workers, "_CPUS", 8)
    monkeypatch.setattr(workers, "available_memory", lambda: 64 * GB)
    assert workers.default_workers() == 8                                # a laptop with plenty of memory


def test_pool_size_overrides(fake_host, monkeypatch):
    monkeypatch.setattr(workers, "_CPUS", 20)
    monkeypatch.setattr(workers, "available_memory", lambda: 12 * GB)
    monkeypatch.setenv(workers.WORKER_MEMORY_ENV, "3G")
    assert workers.default_workers() == 3                                # 12 x 0.75 / 3
    monkeypatch.setenv(workers.WORKERS_ENV, "5")
    assert workers.default_workers() == 5                                # an explicit count wins


def test_plot_workers_setting(monkeypatch):
    import vlbipy.plot_jobs as plot_jobs
    monkeypatch.setattr(plot_jobs, "usable_cpus", lambda: 12)
    assert plot_workers("auto") == 12 and plot_workers(0) == 12 and plot_workers(None) == 12
    assert plot_workers(3) == 3 and plot_workers("4") == 4
