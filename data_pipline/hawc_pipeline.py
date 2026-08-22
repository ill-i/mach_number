#!/usr/bin/env python3
"""Automated HAWC+ -> GETSOURCES -> polarization-SF analysis pipeline.

The original Stokes Q and U maps are never source-subtracted.  GETSOURCES is
run on Stokes I and its final footprints are used only as spatial masks for
the polarization structure-function comparisons.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "hawc-pipeline-mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "hawc-pipeline-cache"))

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.visualization import AsinhStretch, ImageNormalize, ZScaleInterval
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

from SF_analysis import (
    SFAnalysisConfig,
    analyze_sf_table,
    draw_sf_diagnostic,
    estimate_mach_from_break_and_length,
    plot_sf_diagnostic,
)

plt.rcParams["text.usetex"] = False


VARIANTS = ("with_sources", "without_sources", "sources_only")
VARIANT_LABELS = {
    "with_sources": "with sources",
    "without_sources": "without sources",
    "sources_only": "sources only",
}
VARIANT_COLORS = {
    "with_sources": "black",
    "without_sources": "royalblue",
    "sources_only": "crimson",
}


@dataclass
class PipelineParameters:
    snr_threshold: float = 3.0
    min_beams: float = 10.0
    edge_erode_beams: float = 1.0
    max_source_size_arcsec: float = 120.0
    sreliable: float = 5.0
    stentative: float = 3.0
    getsources_beam_fallback_arcsec: float = 18.2
    pix_scale_fallback_arcsec: float = 4.55
    source_catalog_mode: str = "reliable"
    distance_pc: float = 0.0
    vector_step: int = 4


@dataclass
class HawcData:
    path: Path
    object_name: str
    I: np.ndarray
    err_I: np.ndarray
    Q: np.ndarray
    err_Q: np.ndarray
    U: np.ndarray
    err_U: np.ndarray
    qual_I: Optional[np.ndarray]
    header: fits.Header
    wcs: WCS
    beam_arcsec: float
    pixel_scale_arcsec: float
    wavelength_um: int
    original_bunit: str
    warnings: list[str]


@dataclass
class PreparedObject:
    data: HawcData
    root: Path
    components_dir: Path
    getsources_inputs_dir: Path
    run_dir: Path
    report_dir: Path
    prefix: str
    image_base: str
    observation_fits: Path
    detection_fits: Path
    omask_fits: Path


@dataclass
class GetsourcesProducts:
    final_combo_dir: Path
    footprints: Path
    catalog: Path
    catalog_mode_used: str
    reliable_catalog: Path
    relitent_catalog: Path


def _safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")
    return value or "HAWC"


def _short_object_name(path: Path) -> str:
    name = path.stem
    for suffix in ("_RA_Original", "_Original", "_RA"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return _safe_name(name)


def _nominal_beam_arcsec(wavelength_um: float) -> float:
    nominal = {53: 4.85, 63: 5.0, 89: 7.8, 154: 13.6, 214: 18.2, 216: 18.2}
    key = min(nominal, key=lambda item: abs(item - wavelength_um))
    return nominal[key]


def _wcs_pixel_scale_arcsec(wcs: WCS) -> float:
    scales = np.asarray(proj_plane_pixel_scales(wcs.celestial), dtype=float) * 3600.0
    return float(np.mean(np.abs(scales)))


def load_hawc_fits(path: str | Path, parameters: PipelineParameters) -> HawcData:
    path = Path(path).resolve()
    with fits.open(path, memmap=False) as hdul:
        if len(hdul) < 6:
            raise ValueError(f"{path.name}: expected at least 6 HDUs, found {len(hdul)}")
        arrays = [np.asarray(np.squeeze(hdul[index].data), dtype=float) for index in range(6)]
        I, err_I, Q, err_Q, U, err_U = arrays
        qual_I = (
            np.asarray(np.squeeze(hdul[6].data), dtype=float)
            if len(hdul) > 6 and hdul[6].data is not None
            else None
        )
        header = hdul[0].header.copy()

    shapes = {array.shape for array in arrays}
    if qual_I is not None:
        shapes.add(qual_I.shape)
    if len(shapes) != 1 or I.ndim != 2:
        raise ValueError(f"{path.name}: incompatible or non-2D HDU shapes: {sorted(shapes)}")

    wcs = WCS(header).celestial
    wcs_scale = _wcs_pixel_scale_arcsec(wcs)
    hdr_scale = header.get("PIXSCAL")
    warnings: list[str] = []
    if hdr_scale is not None and np.isfinite(float(hdr_scale)):
        relative = abs(float(hdr_scale) - wcs_scale) / max(wcs_scale, 1e-9)
        if relative > 0.05:
            warnings.append(
                f"PIXSCAL={float(hdr_scale):.3f} differs from WCS={wcs_scale:.3f} arcsec/pix; WCS used"
            )
    pixel_scale = wcs_scale if np.isfinite(wcs_scale) and wcs_scale > 0 else parameters.pix_scale_fallback_arcsec

    wavelength = float(header.get("WAVECENT", 216.0))
    beam = header.get("BEAM")
    if beam is None or not np.isfinite(float(beam)) or float(beam) <= 0:
        beam = _nominal_beam_arcsec(wavelength)
        warnings.append(f"BEAM missing; nominal {float(beam):.2f} arcsec used for {wavelength:.1f} um")

    return HawcData(
        path=path,
        object_name=path.stem,
        I=I,
        err_I=err_I,
        Q=Q,
        err_Q=err_Q,
        U=U,
        err_U=err_U,
        qual_I=qual_I,
        header=header,
        wcs=wcs,
        beam_arcsec=float(beam),
        pixel_scale_arcsec=float(pixel_scale),
        wavelength_um=int(round(wavelength)),
        original_bunit=str(header.get("BUNIT", "unknown")),
        warnings=warnings,
    )


def _jy_pixel_to_mjy_sr_factor(pixel_scale_arcsec: float) -> float:
    solid_angle_sr = (pixel_scale_arcsec / 206264.80624709636) ** 2
    return 1.0e-6 / solid_angle_sr


def _intensity_to_mjy_sr(data: HawcData) -> tuple[np.ndarray, float]:
    unit = data.original_bunit.strip().lower().replace(" ", "")
    if unit in {"mjy/sr", "mjy*sr^-1", "mjysr-1"}:
        return data.I.copy(), 1.0
    if unit in {"jy/pixel", "jy/pix", "jypixel-1", "jypix-1"}:
        factor = _jy_pixel_to_mjy_sr_factor(data.pixel_scale_arcsec)
        return data.I * factor, factor
    raise ValueError(
        f"{data.path.name}: unsupported BUNIT={data.original_bunit!r}; expected Jy/pixel or MJy/sr"
    )


def _legacy_getsources_header(header: fits.Header) -> fits.Header:
    out = header.copy()
    out["BUNIT"] = "MJy/sr"
    out["BZERO"] = 0.0
    out["BSCALE"] = 1.0
    # READHEAD from v1.140127 uses prefix matching: RAWFNAME and RADESYS can
    # otherwise be mistaken for RA.  Put the exact RA/DEC cards before those
    # longer keywords, immediately after CRVAL2.
    out.remove("RA", ignore_missing=True, remove_all=True)
    out.remove("DEC", ignore_missing=True, remove_all=True)
    out.remove("RAWFNAME", ignore_missing=True, remove_all=True)
    out.remove("RADESYS", ignore_missing=True, remove_all=True)
    if "CRVAL1" in out and "CRVAL2" in out:
        out.insert("CRVAL2", ("RA", float(out["CRVAL1"])), after=True)
        out.insert("RA", ("DEC", float(out["CRVAL2"])), after=True)
    return out


def _add_legacy_scaling_cards(path: Path) -> None:
    """Write BZERO/BSCALE exactly as expected by GETSOURCES v1.140127."""
    block_size, card_size = 2880, 80
    with path.open("r+b") as stream:
        header = bytearray()
        end_index = None
        while end_index is None:
            block = stream.read(block_size)
            if not block:
                raise RuntimeError(f"FITS END card not found in {path}")
            header.extend(block)
            cards = [header[i : i + card_size] for i in range(0, len(header), card_size)]
            end_index = next((i for i, card in enumerate(cards) if card[:8] == b"END     "), None)
        replacements = (
            "BZERO  = 0.000000000000E+00",
            "BSCALE = 1.000000000000E+00",
            "END",
        )
        for offset, text in enumerate(replacements):
            start = (end_index + offset) * card_size
            header[start : start + card_size] = text.ljust(card_size).encode("ascii")
        stream.seek(0)
        stream.write(header)


def _write_fits(path: Path, array: np.ndarray, header: fits.Header, *, legacy: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.writeto(
        path,
        np.asarray(array, dtype=np.float32),
        header=header,
        overwrite=True,
        output_verify="silentfix",
    )
    if legacy:
        _add_legacy_scaling_cards(path)


def _valid_observation_mask(data: HawcData, edge_erode_beams: float) -> tuple[np.ndarray, int]:
    mask = (
        np.isfinite(data.I)
        & np.isfinite(data.err_I)
        & np.isfinite(data.Q)
        & np.isfinite(data.err_Q)
        & np.isfinite(data.U)
        & np.isfinite(data.err_U)
        & (data.err_I > 0)
        & (data.err_Q > 0)
        & (data.err_U > 0)
    )
    if data.qual_I is not None:
        mask &= np.isfinite(data.qual_I) & (data.qual_I > 0)

    radius = int(math.ceil(edge_erode_beams * data.beam_arcsec / data.pixel_scale_arcsec))
    if radius > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    if not np.any(mask):
        raise RuntimeError(f"{data.path.name}: valid observational mask is empty")
    return mask, radius


def prepare_object(
    fits_path: str | Path,
    output_root: str | Path,
    parameters: PipelineParameters,
) -> PreparedObject:
    data = load_hawc_fits(fits_path, parameters)
    output_root = Path(output_root).resolve()
    short = _short_object_name(data.path)
    root = output_root / "work" / data.object_name
    components_dir = root / "prepared" / "components"
    getsources_inputs_dir = root / "prepared" / "getsources"
    run_dir = root / "getsources_runs"
    report_dir = output_root / "report" / "objects" / data.object_name
    for directory in (components_dir, getsources_inputs_dir, run_dir, report_dir):
        directory.mkdir(parents=True, exist_ok=True)

    components = {
        "I": data.I,
        "err_I": data.err_I,
        "Q": data.Q,
        "err_Q": data.err_Q,
        "U": data.U,
        "err_U": data.err_U,
    }
    if data.qual_I is not None:
        components["qual_I"] = data.qual_I
    for label, array in components.items():
        _write_fits(components_dir / f"{data.object_name}_{label}.fits", array, data.header)

    intensity_mjy, conversion_factor = _intensity_to_mjy_sr(data)
    observation_mask, erosion_radius = _valid_observation_mask(data, parameters.edge_erode_beams)
    prepared_i = np.where(observation_mask, intensity_mjy, 0.0)
    legacy_header = _legacy_getsources_header(data.header)
    image_base = f"{short}_HAWC_I"
    observation_fits = getsources_inputs_dir / f"{image_base}.image.fits"
    detection_fits = getsources_inputs_dir / f"{image_base}.image.det.fits"
    omask_fits = getsources_inputs_dir / f"{image_base}.image.det.omask.fits"
    _write_fits(observation_fits, prepared_i, legacy_header, legacy=True)
    _write_fits(detection_fits, prepared_i, legacy_header, legacy=True)
    _write_fits(omask_fits, observation_mask.astype(float), legacy_header, legacy=True)

    finite_i = intensity_mjy[observation_mask]
    fig, ax = plt.subplots(figsize=(7, 7))
    vmin, vmax = np.nanpercentile(finite_i, [2, 99.5])
    im = ax.imshow(intensity_mjy, origin="lower", cmap="inferno", vmin=vmin, vmax=vmax)
    ax.contour(observation_mask.astype(float), levels=[0.5], colors="cyan", linewidths=0.8)
    ax.set_title(f"{data.object_name}: Stokes I input and safe mask")
    ax.set_xlabel("x [pixel]")
    ax.set_ylabel("y [pixel]")
    fig.colorbar(im, ax=ax, label="Stokes I [MJy/sr]")
    fig.tight_layout()
    fig.savefig(report_dir / f"{data.object_name}_preparation.png", dpi=180)
    plt.close(fig)

    prefix = _safe_name(short + "I")
    metadata = {
        "input_file": str(data.path),
        "shape": list(data.I.shape),
        "wavelength_um": data.wavelength_um,
        "beam_arcsec": data.beam_arcsec,
        "pixel_scale_arcsec": data.pixel_scale_arcsec,
        "original_bunit": data.original_bunit,
        "jy_pixel_to_mjy_sr_factor": conversion_factor,
        "valid_mask_pixels": int(observation_mask.sum()),
        "mask_erosion_radius_px": erosion_radius,
        "warnings": data.warnings,
    }
    (root / "preparation_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return PreparedObject(
        data=data,
        root=root,
        components_dir=components_dir,
        getsources_inputs_dir=getsources_inputs_dir,
        run_dir=run_dir,
        report_dir=report_dir,
        prefix=prefix,
        image_base=image_base,
        observation_fits=observation_fits,
        detection_fits=detection_fits,
        omask_fits=omask_fits,
    )


def _getsources_config_text(
    prepared: PreparedObject,
    parameters: PipelineParameters,
    mono_dir: Path,
    initial_combo: Path,
    stage: str,
) -> str:
    stage_flags = {
        "initial_mono": ("n", "y", "y 1 1", "n", "n y 1", "n y 1 30", "n 1.0"),
        "initial_combo": ("n", "n", "n 1 1", "y", "y y 1", "y y 1 30", "y 1.0"),
        "final_mono": ("y", "y", "y 1 1", "n", "n y 1", "n y 1 30", "n 1.0"),
        "final_combo": ("n", "n", "n 1 1", "y", "y y 1", "y y 1 30", "y 1.0"),
    }
    flattening, decomposing, cleaning, combining, detecting, measuring, visualizing = stage_flags[stage]
    d = prepared.data
    lines = [
        "#___________________________________________________________________________________________________________________________________",
        "#",
        "# GETSOURCES = Multi-Scale Multi-Wavelength Source Extraction = Version 1.140127 = automated Stokes I run",
        "#___________________________________________________________________________________________________________________________________",
        "#                    |                                        !",
        "# VALUES             | VARIABLES' NAMES    {CHOICES/DEFAULTS} ! COMMENTS",
        "#____________________|________________________________________!_____________________________________________________________________",
        "#",
        f" {prepared.prefix}               | prefix",
        " 1                   | nwmax",
        " 1 1                 | nwaves nw[i]",
        f" {d.wavelength_um} {d.beam_arcsec:.4g} {parameters.max_source_size_arcsec:.4g} 0 1 n 0 | wave beam maxsize skmax weight apcorr aprad",
        f" {prepared.getsources_inputs_dir} {mono_dir} | prepared images; monochromatic extraction",
        f" {prepared.image_base}.image.det | detection image",
        f" {prepared.image_base}.image | observed image",
        f" {initial_combo} | initial combined extraction",
        f" {flattening}         <--user.. | flattening",
        f" {decomposing}         <--user.. | decomposing",
        f" {cleaning}     <--user.. | cleaning",
        f" {combining}         <--user.. | combining",
        f" {detecting}     <--user.. | detecting",
        f" {measuring}  <--user.. | measuring",
        f" {visualizing}     <--user.. | visualizing",
        f" {parameters.sreliable:.4g}          ..user.. | sreliable",
        f" {parameters.stentative:.4g}          ..user.. | stentative",
        " 1.0        ..user.. | snratio1min",
        " 1.0        ..user.. | snratio2min",
        " 1          ..user.. | ndetwaves",
        " 0.001      ..user.. | reldiffmax",
        " 0          ..user.. | savespace",
        " 0          ..user.. | verbosity",
        " 0          ..expert | scale1",
        " 99         ..expert | nscales",
        " 0          ..expert | sfactor",
        " 0.333      ..expert | fsbelowbeam",
        " 1.0        ..expert | fscalemax",
        " 0.067      ..expert | ffaintestscale",
        " 6          ..expert | nsigmacutss .............{ optimal 6 }",
        " 2.5        ..expert | nsigmacutfloor",
        " 0.2        ..expert | skewmaxfloor",
        " 0.2        ..expert | kurtmaxfloor",
        " 4          ..expert | minpixfloor",
        " 1.0        ..expert | cleantuning",
        " 7          ..expert | comboexpomax",
        " 7          ..expert | comboexpomin",
        " 1.2        ..expert | contranoise",
        " 2.3        ..expert | factortotal",
        " y 1 30 1   ..expert | filaments",
        " n n 10     ..expert | sproima fproima npxstep",
        "#___________________________________________________________________________________________________________________________________",
    ]
    if len(lines) != 50:
        raise AssertionError(f"GETSOURCES config must have 50 lines, got {len(lines)}")
    return "\n".join(lines) + "\n"


def create_getsources_run_tree(prepared: PreparedObject, parameters: PipelineParameters) -> dict[str, Path]:
    wave = str(prepared.data.wavelength_um)
    stages = {
        "initial_mono": prepared.run_dir / "initial" / wave,
        "initial_combo": prepared.run_dir / "initial" / "combo",
        "final_mono": prepared.run_dir / "final" / wave,
        "final_combo": prepared.run_dir / "final" / "combo",
    }
    for directory in stages.values():
        directory.mkdir(parents=True, exist_ok=True)
    initial_combo = stages["initial_combo"]
    for stage, directory in stages.items():
        mono_dir = stages["initial_mono"] if stage.startswith("initial") else stages["final_mono"]
        (directory / "+getsources.cfg").write_text(
            _getsources_config_text(prepared, parameters, mono_dir, initial_combo, stage),
            encoding="utf-8",
        )
    return stages


def _run_stage(stage: str, directory: Path, getsources_executable: Path, force: bool = False) -> None:
    marker = directory / ".hawc_pipeline_complete.json"
    if marker.exists() and not force:
        print(f"[GETSOURCES] {stage}: already complete")
        return
    log_path = directory / "pipeline_getsources.log"
    env = os.environ.copy()
    env.update(
        {
            "GETSOURCES_BIN": str(getsources_executable.parent),
            "GETSOURCES_HOME": env.get("GETSOURCES_HOME", str(Path.home() / "getsources_home")),
            "PATH": f"{getsources_executable.parent}:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:{env.get('PATH', '')}",
            "LC_ALL": "C",
        }
    )
    print(f"[GETSOURCES] {stage}: starting in {directory}")
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(
            [str(getsources_executable)],
            cwd=directory,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    text = log_path.read_text(encoding="utf-8", errors="replace")
    if process.returncode != 0 or "ERROR in GETSOURCES" in text:
        tail = "\n".join(text.splitlines()[-60:])
        raise RuntimeError(f"GETSOURCES stage {stage} failed (rc={process.returncode}).\n{tail}")
    marker.write_text(json.dumps({"stage": stage, "returncode": process.returncode}, indent=2), encoding="utf-8")
    print(f"[GETSOURCES] {stage}: complete")


def _sync_initial_interface(prepared: PreparedObject, stages: dict[str, Path]) -> None:
    source = stages["initial_mono"] / "+interface"
    target = stages["initial_combo"] / "+interface"
    target.mkdir(parents=True, exist_ok=True)
    pattern = f"{prepared.prefix}.{prepared.data.wavelength_um}.*filaments*.fits"
    copied = 0
    for path in source.glob(pattern):
        shutil.copy2(path, target / path.name)
        copied += 1
    if copied == 0:
        raise FileNotFoundError(f"No initial filament interface files matching {source / pattern}")


def run_getsources(
    prepared: PreparedObject,
    parameters: PipelineParameters,
    getsources_executable: str | Path,
    force: bool = False,
) -> GetsourcesProducts:
    executable = Path(getsources_executable).expanduser().resolve()
    if not executable.exists():
        raise FileNotFoundError(f"GETSOURCES executable not found: {executable}")
    stages = create_getsources_run_tree(prepared, parameters)
    _run_stage("initial_mono", stages["initial_mono"], executable, force=force)
    _run_stage("initial_combo", stages["initial_combo"], executable, force=force)
    _sync_initial_interface(prepared, stages)
    _run_stage("final_mono", stages["final_mono"], executable, force=force)
    _run_stage("final_combo", stages["final_combo"], executable, force=force)

    combo = stages["final_combo"]
    footprints = combo / f"sm.{prepared.prefix}.sw.det.footprints.fits"
    reliable = combo / "+catalogs" / f"{prepared.prefix}.sw.final.reliable.cat"
    relitent = combo / "+catalogs" / f"{prepared.prefix}.sw.final.relitent.cat"
    for path in (footprints, reliable, relitent):
        if not path.exists():
            raise FileNotFoundError(f"Expected final GETSOURCES product missing: {path}")

    requested = parameters.source_catalog_mode.lower()
    catalog = reliable if requested == "reliable" else relitent
    used = requested
    if not read_getsources_catalog_ids(catalog, allow_empty=True):
        fallback = relitent if catalog == reliable else reliable
        if read_getsources_catalog_ids(fallback, allow_empty=True):
            catalog, used = fallback, "relitent" if fallback == relitent else "reliable"
    return GetsourcesProducts(combo, footprints, catalog, used, reliable, relitent)


def locate_getsources_products(
    prepared: PreparedObject,
    parameters: PipelineParameters,
) -> GetsourcesProducts:
    """Locate already completed products without launching GETSOURCES."""
    combo = prepared.run_dir / "final" / "combo"
    footprints = combo / f"sm.{prepared.prefix}.sw.det.footprints.fits"
    reliable = combo / "+catalogs" / f"{prepared.prefix}.sw.final.reliable.cat"
    relitent = combo / "+catalogs" / f"{prepared.prefix}.sw.final.relitent.cat"
    for path in (footprints, reliable, relitent):
        if not path.exists():
            raise FileNotFoundError(f"Expected cached GETSOURCES product missing: {path}")
    requested = parameters.source_catalog_mode.lower()
    catalog = reliable if requested == "reliable" else relitent
    used = requested
    if not read_getsources_catalog_ids(catalog, allow_empty=True):
        fallback = relitent if catalog == reliable else reliable
        if read_getsources_catalog_ids(fallback, allow_empty=True):
            catalog, used = fallback, "relitent" if fallback == relitent else "reliable"
    return GetsourcesProducts(combo, footprints, catalog, used, reliable, relitent)


def read_getsources_catalog_ids(path: str | Path, allow_empty: bool = False) -> list[int]:
    ids: list[int] = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith(("!", "#")):
            continue
        try:
            ids.append(int(float(line.split()[0])))
        except (ValueError, IndexError):
            continue
    ids = sorted(set(ids))
    if not ids and not allow_empty:
        raise ValueError(f"No sources in catalog: {path}")
    return ids


def polarization_snr(data: HawcData) -> np.ndarray:
    err_q = np.where(np.isfinite(data.err_Q) & (data.err_Q > 0), data.err_Q, np.inf)
    err_u = np.where(np.isfinite(data.err_U) & (data.err_U > 0), data.err_U, np.inf)
    if data.qual_I is not None:
        invalid = ~np.isfinite(data.qual_I) | (data.qual_I <= 0)
        err_q[invalid] = np.inf
        err_u[invalid] = np.inf
    p_raw = np.hypot(data.Q, data.U)
    safe = np.where(p_raw > 0, p_raw, np.inf)
    noise = np.sqrt(data.Q**2 * err_q**2 + data.U**2 * err_u**2) / safe
    debiased = np.sqrt(np.maximum(p_raw**2 - noise**2, 0))
    return np.where(np.isfinite(noise) & (noise > 0), debiased / noise, 0.0)


def segment_polarization_regions(
    data: HawcData,
    parameters: PipelineParameters,
) -> tuple[dict[str, np.ndarray], np.ndarray, float, float]:
    snr = polarization_snr(data)
    pixels_per_beam = 1.133 * (data.beam_arcsec / data.pixel_scale_arcsec) ** 2
    smin = parameters.min_beams * pixels_per_beam
    binary = (snr >= parameters.snr_threshold).astype(np.uint8) * 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted((c for c in contours if cv2.contourArea(c) >= smin), key=cv2.contourArea, reverse=True)
    regions: dict[str, np.ndarray] = {}
    for index, contour in enumerate(contours, start=1):
        mask = np.zeros(data.I.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 1, thickness=cv2.FILLED)
        regions[f"Segment_{index}"] = mask.astype(bool)
    return regions, snr, float(pixels_per_beam), float(smin)


def compute_structure_function(
    Q: np.ndarray,
    U: np.ndarray,
    R_list: Iterable[int],
    pixels_per_beam: float,
    bin_width: float = 0.5,
) -> Optional[dict[str, np.ndarray]]:
    Q, U = np.asarray(Q, dtype=float), np.asarray(U, dtype=float)
    ny, nx = Q.shape
    p = np.hypot(Q, U)
    valid = np.isfinite(Q) & np.isfinite(U) & (p > 0)
    q = np.where(valid, Q / p, np.nan)
    u = np.where(valid, U / p, np.nan)
    radii: list[float] = []
    sf: list[float] = []
    pairs: list[int] = []
    errors: list[float] = []

    def contribution(dy: int, dx: int) -> Optional[np.ndarray]:
        y1 = slice(max(0, dy), min(ny, ny + dy))
        y2 = slice(max(0, -dy), min(ny, ny - dy))
        x1 = slice(max(0, dx), min(nx, nx + dx))
        x2 = slice(max(0, -dx), min(nx, nx - dx))
        if q[y1, x1].size == 0 or q[y1, x1].shape != q[y2, x2].shape:
            return None
        mask = np.isfinite(q[y1, x1]) & np.isfinite(q[y2, x2])
        if not np.any(mask):
            return None
        dq = q[y1, x1][mask] - q[y2, x2][mask]
        du = u[y1, x1][mask] - u[y2, x2][mask]
        return 0.25 * (dq**2 + du**2)

    for radius in R_list:
        rmax = int(math.ceil(radius + bin_width))
        shifts = [
            (dy, dx)
            for dy in range(-rmax, rmax + 1)
            for dx in range(-rmax, rmax + 1)
            if (dx or dy) and abs(math.hypot(dx, dy) - radius) <= bin_width
        ]
        values = [contribution(dy, dx) for dy, dx in shifts]
        values = [value for value in values if value is not None and value.size]
        if not values:
            break
        vector = np.concatenate(values)
        n_eff = vector.size / pixels_per_beam
        if n_eff < 5:
            break
        mean = float(np.nanmean(vector))
        sem = float(np.nanstd(vector, ddof=1) / math.sqrt(n_eff)) if vector.size > 1 else np.nan
        radii.append(float(radius))
        sf.append(mean)
        pairs.append(int(vector.size))
        errors.append(sem)
    if not radii:
        return None
    return {
        "R_px": np.asarray(radii),
        "SF": np.asarray(sf),
        "SF_err": np.asarray(errors),
        "Npairs": np.asarray(pairs),
    }


def _mask_geometry(mask: np.ndarray) -> tuple[int, int, int]:
    y, x = np.where(mask)
    if y.size == 0:
        return 0, 0, 0
    length = int(max(x.max() - x.min(), y.max() - y.min()))
    components = int(cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)[0] - 1)
    return int(y.size), length, components


def _write_mask(path: Path, mask: np.ndarray, header: fits.Header) -> None:
    _write_fits(path, mask.astype(float), header)


def _plot_snr_colored_map(
    data: HawcData,
    regions: dict[str, np.ndarray],
    snr: np.ndarray,
    output_path: Path,
    step: int,
) -> None:
    psi = 0.5 * np.arctan2(data.U, data.Q) + np.pi / 2.0
    vx, vy = np.cos(psi), np.sin(psi)
    yy, xx = np.mgrid[: data.I.shape[0], : data.I.shape[1]]
    grid = np.zeros_like(snr, dtype=bool)
    grid[::step, ::step] = True
    finite = data.I[np.isfinite(data.I)]
    norm = ImageNormalize(data.I, interval=ZScaleInterval(), stretch=AsinhStretch(a=0.1))
    fig = plt.figure(figsize=(8.3, 7.2))
    ax = fig.add_subplot(111, projection=data.wcs)
    image = ax.imshow(data.I, origin="lower", cmap="gray", norm=norm, alpha=0.82)
    for index, mask in enumerate(regions.values(), start=1):
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            poly = contour.reshape(-1, 2)
            poly = np.vstack([poly, poly[0]])
            ax.plot(poly[:, 0], poly[:, 1], color="magenta", lw=0.7, alpha=0.8)
        y, x = np.where(mask)
        if x.size:
            ax.text(float(np.median(x)), float(np.median(y)), str(index), color="white", fontsize=8,
                    ha="center", va="center", bbox={"facecolor": "black", "alpha": 0.65, "pad": 1})
    bins = [
        ((snr >= 3) & (snr < 5), "cyan", r"$3 \leq S/N < 5$"),
        ((snr >= 5) & (snr < 10), "gold", r"$5 \leq S/N < 10$"),
        ((snr >= 10), "red", r"$S/N \geq 10$"),
    ]
    kwargs = dict(angles="xy", scale_units="xy", scale=0.2, headwidth=0, headlength=0,
                  headaxislength=0, pivot="mid", width=0.003)
    for selection, color, label in bins:
        selection = selection & grid & np.isfinite(vx) & np.isfinite(vy)
        if np.any(selection):
            ax.quiver(xx[selection], yy[selection], vx[selection], vy[selection], color=color, label=label, **kwargs)
    fig.colorbar(image, ax=ax, pad=0.02, label=f"Stokes I [{data.original_bunit}]")
    ax.set_xlabel("RA (J2000)")
    ax.set_ylabel("Dec (J2000)")
    ax.set_title(f"{data.object_name}: S/N-coded polarization angles")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="best", facecolor="black", labelcolor="white", framealpha=0.75)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_segmentation(
    data: HawcData,
    regions: dict[str, np.ndarray],
    source_mask: np.ndarray,
    smin: float,
    parameters: PipelineParameters,
    output_path: Path,
) -> None:
    fig = plt.figure(figsize=(8.3, 7.2))
    ax = fig.add_subplot(111, projection=data.wcs)
    norm = ImageNormalize(data.I, interval=ZScaleInterval(), stretch=AsinhStretch(a=0.1))
    im = ax.imshow(data.I, origin="lower", cmap="viridis", norm=norm)
    for index, mask in enumerate(regions.values(), start=1):
        ax.contour(mask.astype(float), levels=[0.5], colors="white", linewidths=0.8)
        y, x = np.where(mask)
        if x.size:
            ax.text(np.median(x), np.median(y), str(index), color="white", fontsize=8,
                    bbox={"facecolor": "black", "alpha": 0.7, "pad": 1})
    if np.any(source_mask):
        ax.contour(source_mask.astype(float), levels=[0.5], colors="red", linewidths=0.7)
    ax.set_title(
        f"S/N >= {parameters.snr_threshold:g}; Smin={smin:.1f} px; "
        f"segments={len(regions)}; reliable sources in red"
    )
    ax.set_xlabel("RA (J2000)")
    ax.set_ylabel("Dec (J2000)")
    fig.colorbar(im, ax=ax, pad=0.02, label=f"Stokes I [{data.original_bunit}]")
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_four_row_comparison(
    products: dict[str, tuple[pd.DataFrame, object, dict]],
    object_name: str,
    segment_name: str,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(8.3, 20.5))
    for ax, variant in zip(axes[:3], VARIANTS):
        valid, result, extras = products[variant]
        if valid.empty:
            ax.text(0.5, 0.5, "No valid SF pairs", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue
        draw_sf_diagnostic(
            ax, valid, result, extras, SFAnalysisConfig(smooth_window=3),
            title=f"{segment_name}: {VARIANT_LABELS[variant]}",
            tail_start_px=result.tail_start_pix,
        )
    overlay = axes[3]
    for variant in VARIANTS:
        valid, _, _ = products[variant]
        if valid.empty:
            continue
        x = valid["R_pix"].to_numpy()
        y = valid["Dphi"].to_numpy()
        mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
        overlay.loglog(x[mask], y[mask], marker="o", ms=3, lw=1.4,
                       color=VARIANT_COLORS[variant], label=VARIANT_LABELS[variant])
    overlay.axhline(0.5, color="red", lw=1.5, label="White Noise Limit (0.5)")
    overlay.set_xlabel("Separation R [pixels]")
    overlay.set_ylabel(r"Structure Function $D_{\psi}(R)$")
    overlay.set_title(f"{segment_name}: all variants")
    overlay.grid(alpha=0.2, which="both")
    overlay.legend(loc="best")
    fig.suptitle(f"{object_name}: source-mask comparison", y=0.997, fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.992))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def analyze_object(
    prepared: PreparedObject,
    products: GetsourcesProducts,
    parameters: PipelineParameters,
) -> pd.DataFrame:
    data = prepared.data
    labels = np.rint(np.squeeze(fits.getdata(products.footprints, 0))).astype(int)
    if labels.shape != data.I.shape:
        raise ValueError(f"Footprint grid {labels.shape} != HAWC+ grid {data.I.shape}")
    source_ids = read_getsources_catalog_ids(products.catalog, allow_empty=True)
    source_mask = np.isin(labels, source_ids) if source_ids else np.zeros(data.I.shape, dtype=bool)
    regions, snr, pixels_per_beam, smin = segment_polarization_regions(data, parameters)
    if not regions:
        raise RuntimeError(f"{data.object_name}: no S/N segments above adaptive Smin={smin:.1f} px")

    _write_mask(prepared.report_dir / f"{data.object_name}_sources_mask.fits", source_mask, data.header)
    _plot_snr_colored_map(
        data, regions, snr, prepared.report_dir / f"{data.object_name}_snr_colored_map.pdf",
        parameters.vector_step,
    )
    _plot_segmentation(
        data, regions, source_mask, smin, parameters,
        prepared.report_dir / f"{data.object_name}_segmentation_sources.pdf",
    )

    sf_config = SFAnalysisConfig(
        min_points=8,
        min_npairs=30,
        tail_fraction=0.25,
        smooth_window=3,
        rel_tol=0.08,
        sigma_tol=2.0,
        require_consecutive_tail_points=3,
        start_offset_pix=3,
    )
    finite_qu = np.isfinite(data.Q) & np.isfinite(data.U)
    physical_pc_per_pixel = math.radians(data.pixel_scale_arcsec / 3600.0) * parameters.distance_pc
    rows: list[dict] = []

    for segment_name, region_mask in regions.items():
        ids_here = sorted(set(np.unique(labels[region_mask & source_mask]).tolist()) - {0})
        has_sources = bool(ids_here)
        variants = {"with_sources": region_mask}
        if has_sources:
            variants.update(
                {
                    "without_sources": region_mask & ~source_mask,
                    "sources_only": region_mask & source_mask,
                }
            )
        comparison_products: dict[str, tuple[pd.DataFrame, object, dict]] = {}

        for variant, mask in variants.items():
            mask = mask & finite_qu
            area, length, n_components = _mask_geometry(mask)
            mask_name = f"{segment_name}_{variant}"
            _write_mask(prepared.report_dir / f"{data.object_name}_{mask_name}_mask.fits", mask, data.header)
            base = {
                "Object": data.object_name,
                "Segment": segment_name,
                "Variant": variant,
                "MA": np.nan,
                "L_px": length,
                "Rmax_px": np.nan,
                "lA_px": np.nan,
                "Area_px": area,
                "Imax": np.nan,
                "Imin": np.nan,
                "Imed": np.nan,
                "Plateau": np.nan,
                "Plateau_status": "not_computed",
                "MA_status": "not_computed",
                "N_components": n_components,
                "N_sources_in_segment": len(ids_here),
                "Source_ids": ",".join(map(str, ids_here)),
                "Smin_px": smin,
                "pixels_per_beam": pixels_per_beam,
                "catalog_mode": products.catalog_mode_used,
                "catalog_path": str(products.catalog),
                "FWHM_arcsec": data.beam_arcsec,
                "pixel_scale_arcsec": data.pixel_scale_arcsec,
                "I_unit": data.original_bunit,
            }
            intensity = data.I[mask & np.isfinite(data.I)]
            if intensity.size:
                base.update(Imax=float(np.nanmax(intensity)), Imin=float(np.nanmin(intensity)), Imed=float(np.nanmedian(intensity)))
            if area < 2 or length < 1:
                base["Plateau_status"] = "not_enough_pixels"
                rows.append(base)
                comparison_products[variant] = (pd.DataFrame(), None, {})
                continue

            rmax = max(5, int(2 * length / 3))
            sf = compute_structure_function(
                np.where(mask, data.Q, np.nan),
                np.where(mask, data.U, np.nan),
                range(1, rmax + 1),
                pixels_per_beam,
            )
            base["Rmax_px"] = rmax
            if sf is None:
                base["Plateau_status"] = "no_valid_pairs"
                rows.append(base)
                comparison_products[variant] = (pd.DataFrame(), None, {})
                continue
            raw = pd.DataFrame(
                {
                    "R_pix": sf["R_px"],
                    "Dphi": sf["SF"],
                    "Dphi_err": sf["SF_err"],
                    "Npairs": sf["Npairs"],
                }
            )
            raw["R_arcsec"] = raw["R_pix"] * data.pixel_scale_arcsec
            raw["R_pc"] = raw["R_pix"] * physical_pc_per_pixel
            raw["object_name"] = data.object_name
            raw["mask_name"] = mask_name
            raw["dataset"] = variant
            raw.to_csv(prepared.report_dir / f"{data.object_name}_{mask_name}_sf.csv", index=False)

            result, valid, extras = analyze_sf_table(
                raw, sf_config, prepared.report_dir, l_pix_fallback=length, make_plots=False
            )
            result.object_name = data.object_name
            result.mask_name = mask_name
            result.dataset = variant
            l_a = result.saturation_scale_pix
            if n_components == 1:
                mach = estimate_mach_from_break_and_length(l_a, length, data.object_name, mask_name)
                ma_value = mach.MA_proxy if mach.status == "ok" else np.nan
                ma_status = mach.status
            else:
                ma_value = np.nan
                ma_status = "disconnected_mask"
            base.update(
                {
                    "MA": ma_value,
                    "lA_px": l_a,
                    "Plateau": result.plateau_level,
                    "Plateau_status": result.plateau_detected,
                    "MA_status": ma_status,
                    "SF_valid_points": result.n_points_valid,
                }
            )
            rows.append(base)
            comparison_products[variant] = (valid, result, extras)
            if not valid.empty:
                plot_sf_diagnostic(
                    valid,
                    result,
                    extras,
                    sf_config,
                    title=f"{data.object_name}: {segment_name} - {VARIANT_LABELS[variant]}",
                    tail_start_px=result.tail_start_pix,
                    save_path=prepared.report_dir / f"{data.object_name}_{mask_name}_sf_diagnostic.pdf",
                )

        if has_sources:
            for variant in VARIANTS:
                comparison_products.setdefault(variant, (pd.DataFrame(), None, {}))
            _plot_four_row_comparison(
                comparison_products,
                data.object_name,
                segment_name,
                prepared.report_dir / f"{data.object_name}_{segment_name}_sf_four_rows.pdf",
            )

    table = pd.DataFrame(rows)
    table["Source_area_fraction"] = 0.0
    for segment_name, group in table.groupby("Segment"):
        with_area = group.loc[group["Variant"] == "with_sources", "Area_px"]
        source_area = group.loc[group["Variant"] == "sources_only", "Area_px"]
        fraction = (
            float(source_area.iloc[0] / with_area.iloc[0])
            if len(with_area) and len(source_area) and float(with_area.iloc[0]) > 0
            else 0.0
        )
        table.loc[table["Segment"] == segment_name, "Source_area_fraction"] = fraction
    table.to_csv(prepared.report_dir / f"{data.object_name}_segment_summary.csv", index=False)
    metadata = {
        "object": data.object_name,
        "segments": len(regions),
        "adaptive_Smin_px": smin,
        "source_catalog_mode": products.catalog_mode_used,
        "source_catalog": str(products.catalog),
        "source_count": len(source_ids),
        "footprints": str(products.footprints),
        "parameters": asdict(parameters),
        "warnings": data.warnings,
    }
    (prepared.report_dir / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return table


def _tex_escape(value: object) -> str:
    text = str(value)
    for old, new in [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
    ]:
        text = text.replace(old, new)
    return text


def _fmt(value: object, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "--"
    if not np.isfinite(number):
        return "--"
    return f"{number:.{digits}f}"


def build_latex_report(output_root: str | Path) -> Path:
    output_root = Path(output_root).resolve()
    report_root = output_root / "report"
    report_root.mkdir(parents=True, exist_ok=True)
    object_dirs = sorted(path for path in (report_root / "objects").glob("*") if path.is_dir())
    all_tables: list[pd.DataFrame] = []
    lines = [
        r"\documentclass[10pt,a4paper]{article}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage[T2A]{fontenc}",
        r"\usepackage[russian,english]{babel}",
        r"\usepackage[margin=1.35cm]{geometry}",
        r"\usepackage{graphicx}",
        r"\usepackage{booktabs}",
        r"\usepackage{longtable}",
        r"\usepackage{array}",
        r"\usepackage{float}",
        r"\usepackage{placeins}",
        r"\usepackage[hidelinks]{hyperref}",
        r"\setlength{\parindent}{0pt}",
        r"\begin{document}",
        r"\title{Автоматический анализ поляризации HAWC+, источников и структурных функций}",
        r"\author{HAWC+ / GETSOURCES pipeline}",
        r"\date{\today}",
        r"\maketitle",
        r"\tableofcontents",
    ]
    for object_dir in object_dirs:
        summaries = list(object_dir.glob("*_segment_summary.csv"))
        metadata_path = object_dir / "analysis_metadata.json"
        if not summaries or not metadata_path.exists():
            continue
        table = pd.read_csv(summaries[0])
        all_tables.append(table.copy())
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        object_name = metadata["object"]
        lines.extend(
            [
                r"\clearpage",
                rf"\section{{{_tex_escape(object_name)}}}",
                rf"Адаптивный порог площади: $S_{{\min}}={metadata['adaptive_Smin_px']:.1f}$ пикс.; "
                rf"сегментов: {metadata['segments']}; источников в каталоге: {metadata['source_count']}; "
                rf"каталог: \texttt{{{_tex_escape(metadata['source_catalog_mode'])}}}.",
            ]
        )
        warnings = metadata.get("warnings", [])
        if warnings:
            lines.append(r"\textbf{Предупреждения:} " + "; ".join(_tex_escape(item) for item in warnings) + ".")
        high_coverage = (
            table.loc[table["Source_area_fraction"] > 0.7, ["Segment", "Source_area_fraction"]]
            .drop_duplicates()
            .sort_values("Segment")
        )
        if not high_coverage.empty:
            details = ", ".join(
                f"{_tex_escape(row.Segment)}={100.0 * row.Source_area_fraction:.1f}\\%"
                for row in high_coverage.itertuples()
            )
            lines.append(
                r"\textbf{Высокая доля источника:} " + details
                + r". Для этих сегментов вариант without sources содержит мало оставшихся пикселей."
            )
        snr_map = object_dir / f"{object_name}_snr_colored_map.pdf"
        segmentation = object_dir / f"{object_name}_segmentation_sources.pdf"
        for image, caption in [
            (snr_map, "Поляризационные углы, цвет которых кодирует S/N."),
            (segmentation, "Адаптивная сегментация и надёжные GETSOURCES-источники."),
        ]:
            if image.exists():
                lines.extend(
                    [
                        r"\begin{figure}[H]\centering",
                        rf"\includegraphics[width=0.88\textwidth]{{\detokenize{{{image}}}}}",
                        rf"\caption{{{caption}}}",
                        r"\end{figure}",
                    ]
                )

        lines.extend(
            [
                r"\subsection*{Детализация по областям}",
                r"\scriptsize",
                r"\setlength{\tabcolsep}{2.7pt}",
                r"\begin{longtable}{llrrrrrrrrrl}",
                r"\toprule",
                r"Сегмент & Маска & $M_A$ & $L$ & $R_{\max}$ & $l_A$ & Площадь & $I_{\max}$ & $I_{\min}$ & $I_{\rm med}$ & Плато & Статус \\",
                r"\midrule\endfirsthead",
                r"\toprule",
                r"Сегмент & Маска & $M_A$ & $L$ & $R_{\max}$ & $l_A$ & Площадь & $I_{\max}$ & $I_{\min}$ & $I_{\rm med}$ & Плато & Статус \\",
                r"\midrule\endhead",
            ]
        )
        for _, row in table.iterrows():
            lines.append(
                " & ".join(
                    [
                        _tex_escape(row.get("Segment", "")),
                        _tex_escape(row.get("Variant", "")),
                        _fmt(row.get("MA")),
                        _fmt(row.get("L_px"), 0),
                        _fmt(row.get("Rmax_px"), 0),
                        _fmt(row.get("lA_px")),
                        _fmt(row.get("Area_px"), 0),
                        _fmt(row.get("Imax")),
                        _fmt(row.get("Imin")),
                        _fmt(row.get("Imed")),
                        _fmt(row.get("Plateau"), 4),
                        _tex_escape(row.get("MA_status", "")),
                    ]
                )
                + r" \\"
            )
        lines.extend([r"\bottomrule", r"\end{longtable}", r"\normalsize"])

        for segment in table["Segment"].drop_duplicates().tolist():
            four = object_dir / f"{object_name}_{segment}_sf_four_rows.pdf"
            single = object_dir / f"{object_name}_{segment}_with_sources_sf_diagnostic.pdf"
            if four.exists():
                image, caption = four, f"{segment}: SF with, without and only sources, plus their overlay."
            elif single.exists():
                image, caption = single, f"{segment}: SF; no reliable source overlaps this segment."
            else:
                continue
            lines.extend(
                [
                    r"\clearpage",
                    rf"\subsection*{{{_tex_escape(segment)}}}",
                    r"\begin{figure}[H]\centering",
                    rf"\includegraphics[height=0.88\textheight,width=\textwidth,keepaspectratio]{{\detokenize{{{image}}}}}",
                    rf"\caption{{{_tex_escape(caption)}}}",
                    r"\end{figure}",
                ]
            )
        lines.append(r"\FloatBarrier")
    if all_tables:
        pd.concat(all_tables, ignore_index=True).to_csv(report_root / "all_segment_results.csv", index=False)
    lines.append(r"\end{document}")
    tex_path = report_root / "hawc_pipeline_report.tex"
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return tex_path


def compile_latex_report(tex_path: str | Path) -> Path:
    tex_path = Path(tex_path).resolve()
    command = shutil.which("latexmk")
    if command:
        args = [command, "-pdf", "-interaction=nonstopmode", "-halt-on-error", tex_path.name]
    else:
        command = shutil.which("pdflatex")
        if not command:
            raise FileNotFoundError("Neither latexmk nor pdflatex was found")
        args = [command, "-interaction=nonstopmode", "-halt-on-error", tex_path.name]
    result = subprocess.run(args, cwd=tex_path.parent, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if result.returncode != 0:
        raise RuntimeError("LaTeX compilation failed:\n" + "\n".join(result.stdout.splitlines()[-80:]))
    pdf = tex_path.with_suffix(".pdf")
    if not pdf.exists():
        raise FileNotFoundError(f"Compiled PDF not found: {pdf}")
    return pdf
