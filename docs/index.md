# vlbipy

**Backend-agnostic VLBI data calibration and imaging pipeline.**

vlbipy is a Python package for reducing [Very Long Baseline Interferometry (VLBI)](https://en.wikipedia.org/wiki/Very-long-baseline_interferometry) data. It provides a unified interface to calibrate and image radio interferometric observations from multiple observatories — the [European VLBI Network (EVN)](observatories/evn.md), the [Very Long Baseline Array (VLBA)](observatories/vlba.md), and the [Long Baseline Array (LBA)](observatories/lba.md) — using either [CASA](https://casa.nrao.edu/) or [AIPS](http://www.aips.nrao.edu/) as the data reduction backend.

---

## Why vlbipy?

VLBI data reduction has traditionally required deep familiarity with observatory-specific data formats, calibration procedures, and software ecosystems. Each VLBI network delivers data in slightly different formats and requires different preparation steps before the core calibration can begin. vlbipy abstracts these differences behind a common interface:

- **Observatory-agnostic**: A single pipeline handles EVN, VLBA, and LBA data. Observatory-specific steps (data download, ANTAB handling, flag file parsing) are handled transparently.
- **Backend-agnostic**: Choose between CASA (`casatools`/`casatasks`), AIPS (`ParselTongue`), or a lazy dask-ms reader at runtime, through abstract interfaces. Only CASA is implemented today; AIPS is a stub.
- **Scriptable and reproducible**: Run the full pipeline from a single TOML configuration file and CLI command, or drive each step interactively from Python — both operate on the same `VLBIObs` object.
- **Modular**: Each stage is a callable namespace (`import_data`, `calibrate`, `flag`, `plot`, `export`, ...). Run the default chain, call one step, or resume from where a previous run stopped.

!!! warning "Work in progress"

    The calibration chain runs end to end and produces calibrated, per-source
    data on the CASA backend. Imaging and self-calibration are not implemented
    yet. See [Status](usage/status.md) for the full inventory.

## Quick Start

```bash
# Install with the CASA backend
pip install "vlbipy[casa]"

# Run the full pipeline
vlbipy pipeline -p EG078B -n EVN -t J1234+5678 --phasecal J1230+5600 --fringe-finder 3C345

# Or from a TOML config file
vlbipy pipeline -p EG078B --config my_project.toml
```

```python
from vlbipy import VLBIObs

obs = VLBIObs("EG078B", network="EVN",
              target="J1234+5678", phasecal="J1230+5600", fringe_finder="3C345")
obs.run()
```

## Supported Observatories

| Observatory | Description | Data Archive |
|---|---|---|
| [**EVN**](observatories/evn.md) | European VLBI Network — a distributed array of radio telescopes across Europe, Asia, and Africa | [JIVE archive](http://archive.jive.nl/) |
| [**VLBA**](observatories/vlba.md) | Very Long Baseline Array — a dedicated 10-antenna array across the United States | [NRAO archive](https://data.nrao.edu/portal/) |
| [**LBA**](observatories/lba.md) | Long Baseline Array — the Australian VLBI network | [ATOA](https://atoa.atnf.csiro.au/) |

## Supported Backends

| Backend | Package | Description |
|---|---|---|
| [**CASA**](backends/casa.md) | `casatools`, `casatasks` | The Common Astronomy Software Applications package, the modern standard for radio interferometric calibration |
| [**AIPS**](backends/aips.md) | `parseltongue` | The Astronomical Image Processing System, the classic VLBI calibration package accessed via ParselTongue Python bindings |

## License

vlbipy is released under the [MIT License](https://opensource.org/licenses/MIT).

## Citation

If you use vlbipy in your research, please cite:

> Marcote, B. (2025). *vlbipy: Backend-agnostic VLBI data calibration and imaging pipeline*. [https://github.com/bmarcote/vlbipy](https://github.com/bmarcote/vlbipy)
