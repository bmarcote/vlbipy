"""Returned product objects for vlbipy: Image, ImageSet, SelfcalResult.

These are the rich objects a user receives from imaging and self-calibration.
Under the dummy backend their paths are synthetic and no files are written, but
the interface matches what the real backends will return.
"""
from __future__ import annotations

from typing import Iterator, Optional

from .logging_utils import get_logger
from .models import QualityMetrics

logger = get_logger()


class Image:
    """A single image product.

    Parameters
    ----------
    source : str
        Imaged source name.
    robust : float
        Briggs robust value used.
    weighting : str
        Weighting scheme.
    paths : dict
        Mapping of product kind (image/residual/psf/model/fits/png) to path.
    stats : QualityMetrics
        Image quality metrics.
    """

    def __init__(self, source: str, robust: float, weighting: str,
                 paths: dict[str, str], stats: QualityMetrics) -> None:
        self.source = source
        self.robust = robust
        self.weighting = weighting
        self.paths = paths
        self.stats = stats

    def export_fits(self, outfile: str) -> str:
        """Export the image to a FITS file.

        Under the dummy backend this only logs the action and returns the path;
        no file is written.

        Parameters
        ----------
        outfile : str
            Destination path.

        Returns
        -------
        str
            The output path.
        """
        logger.info("export_fits: {} (robust={:g}) -> {}", self.source, self.robust, outfile)
        return outfile

    def preview(self) -> str:
        """Return the path to the PNG preview (empty string if none)."""
        return self.paths.get("png", "")

    def __repr__(self) -> str:
        return f"Image(source={self.source!r}, robust={self.robust:g}, DR={self.stats.dynamic_range:.4g})"


class ImageSet:
    """An ordered collection of :class:`Image` objects (e.g. a robust sweep).

    Parameters
    ----------
    images : list of Image
        The images.
    """

    def __init__(self, images: list[Image]) -> None:
        self._images: list[Image] = list(images)

    def __iter__(self) -> Iterator[Image]:
        return iter(self._images)

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, key) -> Image:
        """Index by integer position, or select by (float) robust value."""
        if isinstance(key, bool):
            raise TypeError("index must be int position or robust value, not bool")
        if isinstance(key, int):
            return self._images[key]
        for img in self._images:
            if img.robust == key:
                return img
        raise KeyError(f"no image with robust={key}")

    @property
    def by_robust(self) -> dict[float, Image]:
        """Mapping of robust value -> Image."""
        return {img.robust: img for img in self._images}

    def best(self) -> Image:
        """Return the image with the highest dynamic range.

        Returns
        -------
        Image

        Raises
        ------
        ValueError
            If the set is empty.
        """
        if not self._images:
            raise ValueError("empty ImageSet")
        return max(self._images, key=lambda im: im.stats.dynamic_range)

    def export_all(self, directory: str) -> list[str]:
        """Export every image to FITS under ``directory``; return the paths."""
        return [img.export_fits(f"{directory}/{img.source}.robust{img.robust:g}.fits")
                for img in self._images]

    def __repr__(self) -> str:
        return f"ImageSet({len(self._images)} images, robust={[im.robust for im in self._images]})"


class SelfcalResult:
    """The outcome of a self-calibration loop for one source.

    Parameters
    ----------
    source : str
        Source that was self-calibrated.
    rounds : list of dict
        One entry per round: ``{mode, solint, dr_before, dr_after, accepted}``.
    converged : bool
        Whether the loop converged (i.e. at least one accepted improvement).
    final_image : Image, optional
        The image produced by the final accepted round.
    """

    def __init__(self, source: str, rounds: list[dict], converged: bool,
                 final_image: Optional[Image] = None) -> None:
        self.source = source
        self.rounds = rounds
        self.converged = converged
        self.final_image = final_image

    def summary(self) -> str:
        """Return a short multi-line summary of the self-cal loop."""
        lines = [f"Self-cal {self.source}: {len(self.rounds)} rounds, converged={self.converged}"]
        for i, r in enumerate(self.rounds, 1):
            lines.append(f"  round {i}: {r.get('mode', '?')} solint={r.get('solint', '?')} "
                         f"DR {r.get('dr_before', 0):.4g} -> {r.get('dr_after', 0):.4g} "
                         f"({'accepted' if r.get('accepted') else 'discarded'})")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"SelfcalResult(source={self.source!r}, rounds={len(self.rounds)}, converged={self.converged})"
