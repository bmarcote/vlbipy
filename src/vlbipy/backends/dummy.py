"""In-memory dummy backend for vlbipy.

The dummy backend implements the entire component interface without any
external software, data, or file writes. It:

* logs every operation with its resolved parameters and data selection,
* returns realistic, *deterministic* synthetic results (seeded by project code),
* lets the full public API be exercised and unit-tested with no CASA.

Because it implements every operation, it doubles as the reference for what a
complete backend looks like (``backends/template.py`` is the empty skeleton).
"""
from __future__ import annotations

import datetime as dt
import hashlib
import math
from pathlib import Path
from typing import Optional

from ..logging_utils import get_logger
from ..models import (Antenna, CalTable, FreqSetup, ObsMetadata, QualityMetrics, Scan,
                      ScanSNRSurvey, Stokes)
from ..results import Image, SelfcalResult
from .base import (Backend, CalibrationOps, DataOps, ExportOps, FlagOps, ImagingOps, PlotOps)

logger = get_logger()

# Small, representative antenna catalogs per network (code, full name, diameter m, ITRF xyz).
_ANTENNAS: dict[str, list[tuple[str, str, float, tuple[float, float, float]]]] = {
    "EVN": [("EF", "Effelsberg", 100.0, (4033947.0, 486990.0, 4900431.0)),
            ("YS", "Yebes", 40.0, (4848761.0, -261484.0, 4123085.0)),
            ("O8", "Onsala", 25.0, (3370965.0, 711466.0, 5349991.0)),
            ("MC", "Medicina", 32.0, (4461369.0, 919597.0, 4449559.0)),
            ("TR", "Torun", 32.0, (3638558.0, 1221970.0, 5077036.0)),
            ("WB", "Westerbork", 25.0, (3828445.0, 445223.0, 5064921.0)),
            ("JB", "Jodrell Bank", 76.0, (3822626.0, -154105.0, 5086486.0)),
            ("NT", "Noto", 32.0, (4934563.0, 1321201.0, 3806484.0)),
            ("HH", "Hartebeesthoek", 26.0, (5085442.0, 2668263.0, -2768697.0)),
            ("IR", "Irbene", 32.0, (3183649.0, 1276902.0, 5359264.0)),
            ("SV", "Svetloe", 32.0, (2730174.0, 1562443.0, 5529969.0)),
            ("T6", "Tianma", 65.0, (-2826709.0, 4679237.0, 3274668.0))],
    "VLBA": [("PT", "Pie Town", 25.0, (-1640954.0, -5014816.0, 3575412.0)),
             ("LA", "Los Alamos", 25.0, (-1449752.0, -4975299.0, 3709124.0)),
             ("FD", "Fort Davis", 25.0, (-1324009.0, -5332182.0, 3231962.0)),
             ("KP", "Kitt Peak", 25.0, (-1995679.0, -5037318.0, 3357328.0)),
             ("OV", "Owens Valley", 25.0, (-2409150.0, -4478573.0, 3838617.0)),
             ("BR", "Brewster", 25.0, (-2112065.0, -3705357.0, 4726814.0)),
             ("NL", "North Liberty", 25.0, (-130872.0, -4762317.0, 4226851.0)),
             ("MK", "Mauna Kea", 25.0, (-5464075.0, -2495248.0, 2148297.0)),
             ("SC", "St Croix", 25.0, (2607848.0, -5488070.0, 1932739.0)),
             ("HN", "Hancock", 25.0, (1446375.0, -4447939.0, 4322306.0))],
    "LBA": [("AT", "ATCA", 22.0, (-4751639.0, 2791700.0, -3200491.0)),
            ("PA", "Parkes", 64.0, (-4554232.0, 2816759.0, -3454036.0)),
            ("MP", "Mopra", 22.0, (-4682769.0, 2802619.0, -3291759.0)),
            ("HO", "Hobart", 26.0, (-3950237.0, 2522347.0, -4311562.0)),
            ("CD", "Ceduna", 30.0, (-3753443.0, 3912709.0, -3348067.0)),
            ("TI", "Tidbinbilla", 70.0, (-4460895.0, 2682361.0, -3674748.0))],
}


def _seed(*parts: str) -> int:
    """Return a stable 32-bit integer seed derived from the given strings."""
    return int(hashlib.md5(":".join(parts).encode()).hexdigest()[:8], 16)


def _log(operation: str, **params) -> None:
    """Log an operation with its resolved parameters."""
    detail = " ".join(f"{k}={v!r}" for k, v in params.items() if v is not None and v != "")
    logger.info("[dummy] {} {}", operation, detail)


def _synthetic_path(work_dir, *parts: str) -> str:
    """Build a synthetic (never-created) path under the work directory."""
    return "/".join([str(work_dir).rstrip("/"), *parts])


class DummyDataOps(DataOps):
    """Fabricate metadata and log imports; writes nothing."""

    def is_imported(self, project_code: str) -> bool:
        """The dummy backend has no on-disk product, so nothing is ever imported."""
        return False

    def import_data(self, project_code: str, source_names: list[str], *, scan_gap: int = 15,
                    files: Optional[list[str]] = None, delete: bool = False, **kwargs) -> None:
        """Log what a real import would do."""
        _log("import_data", project=project_code, scan_gap=scan_gap,
             split_per_source=",".join(source_names), n_files=len(files or []) or None)

    def get_metadata(self, project_code: str, source_names: list[str], observatory: str) -> ObsMetadata:
        """Fabricate deterministic metadata for a project (seeded by code + network)."""
        rng = _seed(project_code, observatory)
        catalog = _ANTENNAS.get(observatory, _ANTENNAS["EVN"])
        n_ant = 6 + (rng % max(1, len(catalog) - 6))
        chosen = catalog[:n_ant]
        n_sub = 8
        antennas = {
            code: Antenna(name=code, fullname=full, diameter=diam, position=xyz, observed=True,
                          subbands=tuple(range(n_sub)), mount="ALT-AZ")
            for code, full, diam, xyz in chosen
        }
        freq = FreqSetup(ref_freq=1.6e9, total_bandwidth=256e6, n_subbands=n_sub,
                         n_channels=32, channel_width=1e6, polarizations=[Stokes.RR, Stokes.LL])
        names = list(source_names) or ["UNKNOWN"]
        scans: list[Scan] = []
        t0 = 0.0
        for i in range(len(names) * 3):
            src = names[i % len(names)]
            scans.append(Scan(scan_number=i + 1, source=src, time_start=t0, time_end=t0 + 300.0,
                              antennas=list(antennas.keys()), integration_time=2.0,
                              subbands=tuple(range(n_sub))))
            t0 += 360.0
        for antenna in antennas.values():
            antenna.n_scans = sum(1 for s in scans if antenna.name in s.antennas)
        meta = ObsMetadata(project_code=project_code, obs_date=dt.date(2018, 10, 30),
                           time_range=(0.0, t0), antennas=antennas, scans=scans,
                           freq_setup=freq, source_names=names)
        _log("get_metadata", project=project_code, observatory=observatory,
             n_antennas=len(antennas), n_scans=len(scans), sources=",".join(names),
             max_baseline_km=round(meta.max_baseline / 1e3, 1),
             resolution_mas=round(meta.resolution_mas, 2))
        return meta

    def reset_calibration(self, project_code: str, *, unflag: bool = True,
                          backup_flags: bool = True) -> dict:
        """Log the clearcal/unflag a real backend would do."""
        _log("reset_calibration", project=project_code, unflag=unflag, backup_flags=backup_flags)
        return {"flagged_before": 0.0, "flagged_after": 0.0,
                "backup": f"{project_code}_before_reset" if backup_flags and unflag else ""}

    def get_subband_participation(self, project_code: str,
                                  antenna_names: list[str]) -> dict[str, tuple[int, ...]]:
        """Pretend every antenna recorded every subband."""
        _log("get_subband_participation", project=project_code, n_antennas=len(antenna_names))
        return {name: tuple(range(8)) for name in antenna_names}

    def listobs(self, project_code: str, listfile: Optional[str] = None) -> dict:
        """Return a synthetic scan listing matching :meth:`get_metadata` (no file written)."""
        meta = self.get_metadata(project_code, [], "EVN")
        _log("listobs", project=project_code, listfile=listfile)
        return {f"scan_{s.scan_number}": {"source": s.source, "time_start": s.time_start,
                                          "time_end": s.time_end, "antennas": s.antennas}
                for s in meta.scans}


class DummyCalibrationOps(CalibrationOps):
    """Log calibration solves and return synthetic calibration tables."""

    def a_priori(self, project_code: str, field: str, *, needs_eop: bool = False,
                 eop_file: Optional[str] = None, **kwargs) -> list[CalTable]:
        """Synthesize Tsys / gain-curve (+ EOP) tables."""
        cal_types = ["tsys", "gc"] + (["eop"] if needs_eop else [])
        tables = []
        for cal_type in cal_types:
            _log("gencal", project=project_code, caltype=cal_type, field=field)
            tables.append(CalTable(cal_type=cal_type, field=field, interp="nearest",
                                   path=_synthetic_path(self.work_dir, "cal",
                                                        f"{project_code}.{cal_type}")))
        return tables

    def initial_calibration(self, project_code: str, field: str, refant: str, *,
                            scans=None, suffix: str = "sbd", **kwargs) -> CalTable:
        """Synthesize a single-band delay table."""
        _log("fringefit(sbd)", project=project_code, field=field, refant=refant, zerorates=True,
             scans=scans, suffix=suffix)
        return CalTable(cal_type=suffix, field=field, interp="nearest", snr=180.0,
                        path=_synthetic_path(self.work_dir, "cal", f"{project_code}.{suffix}"))

    def bandpass(self, project_code: str, field: str, refant: str, *, scans=None,
                 solint: str = "inf", combine: str = "scan", **kwargs) -> CalTable:
        """Synthesize a bandpass table."""
        _log("bandpass", project=project_code, field=field, refant=refant, scans=scans,
             solint=solint, combine=combine)
        return CalTable(cal_type="bpass", field=field, interp="nearest,nearest", snr=40.0,
                        path=_synthetic_path(self.work_dir, "cal", f"{project_code}.bpass"))

    def fringefit(self, project_code: str, field: str, refant: str, *, solint: str = "inf",
                  combine: str = "spw", minsnr: float = 5.0, metadata=None,
                  **kwargs) -> CalTable:
        """Synthesize a multi-band delay table, with the spw map apply would need."""
        _log("fringefit(mbd)", project=project_code, field=field, refant=refant, combine=combine,
             solint=solint, minsnr=minsnr)
        n_spw = metadata.freq_setup.n_subbands if metadata is not None else 0
        spwmap = [0] * n_spw if "spw" in combine and n_spw else []
        return CalTable(cal_type="mbd", field=field, interp="linear", snr=25.0, spwmap=spwmap,
                        path=_synthetic_path(self.work_dir, "cal", f"{project_code}.mbd"))

    def scalar_bandpass(self, project_code: str, field: str, refant: str, **kwargs) -> CalTable:
        """Synthesize a per-antenna, per-subband amplitude table."""
        _log("gaincal(scalar_bp)", project=project_code, field=field, refant=refant, calmode="a")
        return CalTable(cal_type="scalar_bp", field=field, interp="nearest", snr=30.0,
                        path=_synthetic_path(self.work_dir, "cal", f"{project_code}.scalar_bp"))

    def scan_snr(self, project_code: str, field: str, *, refant: str = "",
                 channel_fraction: float = 0.8, scans: Optional[list] = None,
                 **kwargs) -> ScanSNRSurvey:
        """Fabricate a deterministic per-scan/antenna/polarization SNR survey.

        SNRs decay with antenna index (bigger dishes first in the catalogs) and
        wobble per scan, so the plotted matrix looks like a real one: a few weak
        antennas, occasional failed solutions (``nan``), and a masked refant.
        """
        meta = self.backend.data.get_metadata(project_code, [f for f in field.split(",") if f],
                                              kwargs.get("observatory", "EVN"))
        wanted = set(scans) if scans else None
        selected = [s for s in meta.scans
                    if (wanted is None or s.scan_number in wanted)
                    and (not field or s.source in field.split(","))]
        antennas = list(meta.antennas)
        refant = refant or (antennas[0] if antennas else "")
        pols = ["RR", "LL"]
        matrices: dict[str, list[list[float]]] = {}
        for pol_index, pol in enumerate(pols):
            rows = []
            for scan in selected:
                row = []
                for ant_index, antenna in enumerate(antennas):
                    rng = _seed(project_code, pol, antenna, str(scan.scan_number))
                    if antenna == refant or antenna not in scan.antennas or rng % 37 == 0:
                        row.append(float("nan"))  # refant sentinel / absent / failed solve
                        continue
                    base = 400.0 * math.exp(-0.25 * ant_index) * (0.7 + 0.6 * (pol_index * 0.5))
                    row.append(round(base * (0.6 + (rng % 100) / 125.0), 1))
                rows.append(row)
            matrices[pol] = rows
        survey = ScanSNRSurvey(project_code=project_code,
                               scan_numbers=[s.scan_number for s in selected],
                               scan_sources=[s.source for s in selected], antennas=antennas,
                               snr=matrices, channel_fraction=channel_fraction, refant=refant)
        _log("scan_snr", project=project_code, field=field, refant=refant,
             channel_fraction=channel_fraction, n_scans=len(selected), n_antennas=len(antennas))
        return survey

    def measure_edge_channels(self, project_code: str, table: CalTable, *,
                              threshold: float = 6.0, **kwargs) -> dict:
        """Synthesize a band profile that rolls off over the outer 10% of each subband."""
        n_chan = 32
        n_edge = max(1, n_chan // 10)
        ramp = [min(1.0, (i + 1) / n_edge) for i in range(n_chan // 2)]
        profile = ramp + ramp[::-1]
        _log("measure_edge_channels", project=project_code, caltable=table.cal_type,
             threshold=threshold, n_edge=n_edge)
        return {"n_edge": n_edge, "first": n_edge, "last": n_chan - 1 - n_edge,
                "n_channels": n_chan, "amplitude_profile": profile,
                "phase_profile": [1.0 - v for v in profile],
                "flagged_fraction": [0.0] * n_chan}

    def smooth(self, project_code: str, table: CalTable, **kwargs) -> CalTable:
        """Return the table unchanged, logging the smoothing that would happen."""
        _log("smooth", project=project_code, caltable=table.cal_type, **kwargs)
        return table

    def apply(self, project_code: str, field: str, tables: list[CalTable], **kwargs) -> None:
        """Log the applycal that would run."""
        _log("applycal", project=project_code, field=field,
             gaintables=",".join(t.cal_type for t in tables))


class DummyFlagOps(FlagOps):
    """Return a deterministic flagged fraction per (project, mode, field)."""

    def measure_quack(self, project_code: str, *, field: str = "", threshold: float = 0.9,
                      **kwargs) -> dict:
        """Synthesize a short settling ramp on two antennas."""
        _log("measure_quack", project=project_code, field=field, threshold=threshold)
        return {"per_antenna": {"EF": 4.0, "WB": 6.0}, "profiles": {}, "offsets": [],
                "threshold": threshold, "integration_time": 2.0}

    def outliers(self, project_code: str, *, field: str = "", threshold: float = 5.0,
                 dry_run: bool = False, **kwargs) -> dict:
        """Return a deterministic synthetic outlier report."""
        rng = _seed(project_code, "outliers", field)
        fraction = round((rng % 300) / 10000.0, 4)
        _log("outliers", project=project_code, field=field, threshold=threshold,
             dry_run=dry_run, fraction=fraction)
        return {"n_outliers": rng % 500, "n_points": 100000, "fraction": fraction,
                "per_baseline": [], "threshold": threshold, "commands": rng % 50,
                "flagged_fraction_of_data": 0.0 if dry_run else fraction}

    def run(self, project_code: str, kind: str, *, field: str = "", **kwargs) -> float:
        """Return a stable pseudo-random flagged fraction between 0 and 0.15."""
        rng = _seed(project_code, "flag", kind, field)
        fraction = round((rng % 1500) / 10000.0, 4)
        _log(f"flag[{kind}]", project=project_code, field=field, flagged_fraction=fraction, **kwargs)
        return fraction


class DummyImagingOps(ImagingOps):
    """Fabricate images and self-calibration runs with plausible statistics."""

    def clean(self, project_code: str, source: str, *, robust: float = 0.0, imager: str = "wsclean",
              imsize: Optional[list[int]] = None, weighting: str = "briggs", niter: int = 0,
              **kwargs) -> Image:
        """Synthesize an image whose rms/beam respond correctly to the robust parameter."""
        rng = _seed(project_code, source)
        peak = round(0.1 + (rng % 5000) / 1000.0, 4)                # 0.1 - 5.1 Jy/beam
        rms_base = 3.0e-5 + (rng % 100) * 1.0e-7                    # ~3e-5 Jy/beam
        rms = round(rms_base * (1.0 + 0.30 * (2.0 - robust)), 9)    # lower robust -> higher rms
        integrated = round(peak * (1.2 + (rng % 80) / 100.0), 4)
        bmaj = round(1.2 * (1.0 + 0.25 * (robust + 2.0)), 3)        # lower robust -> smaller beam
        stats = QualityMetrics(peak=peak, rms=rms, dynamic_range=round(peak / rms, 1),
                               integrated_flux=integrated,
                               beam=(bmaj, round(bmaj * 0.6, 3), round((rng % 180) - 90.0, 1)))
        base = _synthetic_path(self.work_dir, "images", f"{source}.robust{robust:g}")
        paths = {k: f"{base}.{ext}" for k, ext in
                 (("image", "image"), ("residual", "residual"), ("psf", "psf"),
                  ("model", "model"), ("fits", "image.fits"), ("png", "image.png"))}
        _log("clean", project=project_code, source=source, imager=imager, robust=robust,
             weighting=weighting, imsize=imsize, niter=niter, peak=peak, rms=rms,
             dr=stats.dynamic_range)
        return Image(source=source, robust=float(robust), weighting=weighting, paths=paths,
                     stats=stats)

    def uvmodel(self, project_code: str, source: str, *, components: int = 1, **kwargs) -> Image:
        """Synthesize a uv-plane Gaussian model fit (a compact version of :meth:`clean`)."""
        image = self.clean(project_code, source, robust=0.0, imager="uvmodel", niter=0)
        _log("uvmodel", project=project_code, source=source, components=components)
        return image

    def selfcal(self, project_code: str, source: str, *, image: Optional[Image] = None,
                phase_rounds: int = 4, ampphase_rounds: int = 5, threshold: float = 0.05,
                **kwargs) -> SelfcalResult:
        """Run a synthetic self-cal loop with diminishing per-round improvements."""
        dynamic_range = image.stats.dynamic_range if image is not None else 1000.0
        rounds: list[dict] = []
        converged = False
        plan = [("p", "60s")] * phase_rounds + [("ap", "300s")] * ampphase_rounds
        for mode, solint in plan:
            gain = 0.20 if mode == "p" else 0.06          # diminishing improvements
            improvement = gain * (1.0 - len(rounds) / (len(plan) + 1.0))
            new_dr = dynamic_range * (1.0 + improvement)
            accepted = improvement >= threshold
            rounds.append({"mode": mode, "solint": solint, "dr_before": round(dynamic_range, 1),
                           "dr_after": round(new_dr, 1), "accepted": accepted})
            if accepted:
                dynamic_range = new_dr
                converged = True
            _log("selfcal_round", project=project_code, source=source, mode=mode, solint=solint,
                 dr_before=rounds[-1]["dr_before"], dr_after=rounds[-1]["dr_after"],
                 accepted=accepted)
        final_image = None
        if image is not None:
            final_image = self.clean(project_code, source, robust=image.robust, imager="wsclean",
                                     weighting=image.weighting)
            final_image.stats.dynamic_range = round(dynamic_range, 1)
        return SelfcalResult(source=source, rounds=rounds, converged=converged,
                             final_image=final_image)


class DummyPlotOps(PlotOps):
    """Return synthetic plot paths without writing any file."""

    def plot_dir(self, category: str = "") -> Path:
        """Return the category path *without* creating it (this backend writes nothing)."""
        if category and category not in self.CATEGORIES:
            raise ValueError(f"unknown plot category {category!r} "
                             f"(expected one of {', '.join(self.CATEGORIES)})")
        return self.work_dir / "plots" / category if category else self.work_dir / "plots"

    def diagnostic(self, project_code: str, kind: str, **kwargs) -> str:
        """Return the path a real diagnostic plot would be written to."""
        path = _synthetic_path(self.plot_dir("raw"), f"{project_code}.{kind}.png")
        _log(f"plot[{kind}]", project=project_code, output=path, **kwargs)
        return path

    def caltable(self, project_code: str, caltable: str, cal_type: str = "") -> list[str]:
        """Return the path a real caltable plot would be written to."""
        name = Path(caltable).name if caltable else cal_type
        path = _synthetic_path(self.plot_dir("caltables"), f"{name}.png")
        _log("plot_caltable", project=project_code, caltable=caltable, output=path)
        return [path]

    def spectrum(self, project_code: str, *, field: str = "", scans: Optional[list] = None,
                 refant: str = "", column: str = "corrected", label: str = "", **kwargs) -> list[str]:
        """Return the paths a real amplitude/phase spectrum plot would be written to."""
        tag = f".{label}" if label else ""
        directory = self.plot_dir(self.category_for_column(column))
        paths = [_synthetic_path(directory, f"{project_code}{tag}.spectrum.{k}.png")
                 for k in ("amp", "phase")]
        _log("plot[spectrum]", project=project_code, field=field, column=column,
             outputs=",".join(paths))
        return paths

    def radplot(self, project_code: str, *, field: str = "", column: str = "corrected",
                time_bin: float = 10.0, label: str = "", **kwargs) -> str:
        """Return the path a real radplot would be written to."""
        tag = f".{label}" if label else ""
        path = _synthetic_path(self.plot_dir(self.category_for_column(column)),
                               f"{project_code}{tag}.radplot.{field or 'source'}.png")
        _log("plot[radplot]", project=project_code, field=field, time_bin=time_bin, output=path)
        return path

    def timeseries(self, project_code: str, *, field: str = "", refant: str = "",
                   column: str = "corrected", label: str = "", **kwargs) -> list[str]:
        """Return the path a real amplitude/phase vs time plot would be written to."""
        tag = f".{label}" if label else ""
        path = _synthetic_path(self.plot_dir(self.category_for_column(column)),
                               f"{project_code}{tag}.timeseries.png")
        _log("plot[timeseries]", project=project_code, field=field, column=column, output=path)
        return [path]

    def baseline_corner(self, project_code: str, *, field: str = "", column: str = "corrected",
                        quantity: str = "phase", label: str = "", **kwargs) -> str:
        """Return the path a real baseline corner plot would be written to."""
        tag = f".{label}" if label else ""
        path = _synthetic_path(self.plot_dir(self.category_for_column(column)),
                               f"{project_code}{tag}.corner.{quantity}.png")
        _log("plot[corner]", project=project_code, field=field, quantity=quantity, output=path)
        return path

    def bandpass_profile(self, project_code: str, measurement: dict, **kwargs) -> str:
        """Return the path a real band-profile plot would be written to."""
        path = _synthetic_path(self.plot_dir("caltables"), f"{project_code}.bandpass_profile.png")
        _log("plot[bandpass_profile]", project=project_code, output=path,
             n_edge=measurement.get("n_edge"))
        return path

    def scan_snr(self, project_code: str, survey: Optional[ScanSNRSurvey] = None,
                 **kwargs) -> list[str]:
        """Return one synthetic path per polarization of the SNR matrix plot."""
        pols = survey.polarizations if survey is not None else ["RR", "LL"]
        paths = [_synthetic_path(self.plot_dir("raw"), f"{project_code}.snr_matrix.{p}.png")
                 for p in pols]
        _log("plot[scan_snr]", project=project_code, polarizations=",".join(pols),
             outputs=",".join(paths))
        return paths


class DummyExportOps(ExportOps):
    """Return synthetic export paths."""

    def uvfits(self, project_code: str, source: str, **kwargs) -> str:
        """Return the path a real UVFITS export would produce."""
        path = _synthetic_path(self.work_dir, "export", f"{project_code}_{source}.uvfits")
        _log("export_uvfits", project=project_code, source=source, output=path, **kwargs)
        return path

    def ms(self, project_code: str, source: str, **kwargs) -> str:
        """Return the path a real per-source split would produce."""
        path = _synthetic_path(self.work_dir, "export", f"{project_code}_{source}.ms")
        _log("export_ms", project=project_code, source=source, output=path, **kwargs)
        return path

    def merge(self, project_codes: list[str], source_names: list[str], **kwargs) -> str:
        """Return the handle a real multi-epoch concatenation would produce."""
        handle = _synthetic_path(self.work_dir, "merged", f"{'+'.join(project_codes)}.ms")
        _log("merge", projects=",".join(project_codes), sources=",".join(source_names),
             output=handle)
        return handle


class DummyBackend(Backend):
    """A logging, in-memory backend that fabricates deterministic results.

    Writes no files and requires no external software. Every operation logs what
    a real backend *would* do and returns synthetic objects.

    Parameters
    ----------
    work_dir : str
        Directory used only to build synthetic paths (never created).
    """

    kind = "dummy"
    requires_data_files = False

    data_ops = DummyDataOps
    calibration_ops = DummyCalibrationOps
    flag_ops = DummyFlagOps
    imaging_ops = DummyImagingOps
    plot_ops = DummyPlotOps
    export_ops = DummyExportOps
