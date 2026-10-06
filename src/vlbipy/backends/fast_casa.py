"""Fast CASA backend: CASA for import, flagging and apply; numpy/dask-ms solvers for the slow solves.

This backend is :class:`~vlbipy.backends.casa.CasaBackend` with the fringe-fit engine
replaced. Every fringe fit the pipeline runs (``initial_calibration`` single-band delay,
including its staged variant, the ``fringefit`` multi-band delay and the ``scan_snr``
survey) goes through :meth:`CasaCalibrationOps._run_fringefit_task`; here that method
calls :func:`vlbipy.solvers.fringefit_task.run_fringefit`, which reads the measurement
set with dask-ms and solves with :mod:`vlbipy.solvers.fringe`. The tables it writes are
CASA "Fringe Jones" tables, so ``applycal``, the plots and the dask-ms store conversion
see no difference.

Select it with ``backend = "casa-fast"`` in the configuration (aliases ``fast``).
"""
from __future__ import annotations

from .casa import CasaBackend, CasaCalibrationOps


class FastCalibrationOps(CasaCalibrationOps):
    """CASA calibration operations with the fringe fits solved by :mod:`vlbipy.solvers`."""

    def _run_fringefit_task(self, params: dict) -> None:
        """Solve a ``casatasks.fringefit`` request with the numpy engine (same keyword set, same table format)."""
        from ..solvers.fringefit_task import run_fringefit
        run_fringefit(**params)


class FastCasaBackend(CasaBackend):
    """CASA backend whose fringe fitting runs on dask-ms + numpy (10-100x faster than the CASA task)."""

    kind = "casa-fast"
    calibration_ops = FastCalibrationOps

    def __init__(self, work_dir: str = ".") -> None:
        try:
            import daskms  # noqa: F401
        except ImportError as exc:
            from ..errors import BackendError
            raise BackendError("the casa-fast backend requires dask-ms: pip install vlbipy[daskms]") from exc
        super().__init__(work_dir)
