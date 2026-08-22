"""Compare polarization structure functions using GETSOURCES source masks.

The HAWC+ Stokes Q and U arrays are never altered.  GETSOURCES products are
used only to select pixels inside or outside compact-source footprints.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits

from SF_analysis import (
    SFAnalysisConfig,
    analyze_sf_table,
    estimate_mach_from_break_and_length,
    plot_sf_diagnostic,
)


VARIANT_LABELS = {
    "with_sources": "Filament region: with sources",
    "without_sources": "Filament region: without sources",
    "sources_only": "Sources only",
}


def read_getsources_catalog_ids(catalog_path: str | Path) -> list[int]:
    """Read source numbers (column NO) from a GETSOURCES ASCII catalog."""
    catalog_path = Path(catalog_path)
    ids: list[int] = []
    for line in catalog_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("!", "#")):
            continue
        try:
            ids.append(int(float(stripped.split()[0])))
        except (ValueError, IndexError) as exc:
            raise ValueError(
                f"Cannot read source number from {catalog_path}: {line!r}"
            ) from exc
    if not ids:
        raise ValueError(f"No sources found in catalog: {catalog_path}")
    return sorted(set(ids))


def _check_same_fits_grid(hawc_file: Path, footprints_file: Path, shape: tuple[int, int]) -> None:
    """Fail early if the footprint image cannot be applied pixel by pixel."""
    hawc_header = fits.getheader(hawc_file, 0)
    footprint_header = fits.getheader(footprints_file, 0)
    footprint_shape = np.squeeze(fits.getdata(footprints_file, 0)).shape
    if footprint_shape != shape:
        raise ValueError(
            f"Grid mismatch: HAWC+ has {shape}, footprints have {footprint_shape}. "
            "Reprojection is required before masking."
        )

    numeric_keys = ("CRPIX1", "CRPIX2", "CRVAL1", "CRVAL2", "CDELT1", "CDELT2")
    for key in numeric_keys:
        if key in hawc_header and key in footprint_header:
            if not np.isclose(hawc_header[key], footprint_header[key], rtol=0, atol=1e-9):
                raise ValueError(f"WCS mismatch in {key}: {hawc_header[key]} != {footprint_header[key]}")


def _mask_from_contour(shape: tuple[int, int], contour: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 1, thickness=cv2.FILLED)
    return mask.astype(bool)


def _mask_geometry(mask: np.ndarray) -> tuple[int, int, int]:
    """Return area, bounding-box length and number of connected components."""
    y, x = np.where(mask)
    if y.size == 0:
        return 0, 0, 0
    length = int(max(x.max() - x.min(), y.max() - y.min()))
    n_components = int(cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)[0] - 1)
    return int(y.size), length, n_components


def _write_mask(path: Path, mask: np.ndarray, header) -> None:
    fits.writeto(path, mask.astype(np.uint8), header=header, overwrite=True)


def _plot_mask_diagnostic(
    intensity: np.ndarray,
    region_mask: np.ndarray,
    source_mask_raw: np.ndarray,
    source_mask_buffered: np.ndarray,
    output_path: Path,
) -> None:
    finite = intensity[np.isfinite(intensity)]
    vmin, vmax = np.nanpercentile(finite, [2, 98])
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    axes[0].imshow(intensity, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
    axes[0].contour(region_mask, levels=[0.5], colors="cyan", linewidths=1)
    axes[0].contour(source_mask_raw, levels=[0.5], colors="red", linewidths=0.8)
    axes[0].set_title("Fil1 region (cyan) and sources (red)")
    axes[1].imshow(source_mask_raw, origin="lower", cmap="magma", vmin=0, vmax=1)
    axes[1].set_title("GETSOURCES footprints")
    axes[2].imshow(source_mask_buffered, origin="lower", cmap="magma", vmin=0, vmax=1)
    axes[2].set_title("Footprints used for exclusion")
    for ax in axes:
        ax.set_xlabel("x [pixel]")
        ax.set_ylabel("y [pixel]")
    fig.savefig(output_path)
    plt.close(fig)


def _plot_variant_comparison(
    tables: Dict[str, pd.DataFrame],
    title: str,
    output_path: Path,
) -> None:
    colors = {
        "with_sources": "black",
        "without_sources": "royalblue",
        "sources_only": "crimson",
    }
    fig, ax = plt.subplots(figsize=(8, 5))
    for variant, table in tables.items():
        if table.empty:
            continue
        x = table["R_pix"].to_numpy()
        y = table["Dphi"].to_numpy()
        yerr = table["Dphi_err"].to_numpy()
        valid = (x > 0) & (y > 0) & np.isfinite(y)
        if not np.any(valid):
            continue
        ax.errorbar(
            x[valid], y[valid], yerr=yerr[valid], marker="o", ms=3, lw=1,
            capsize=1.5, color=colors[variant], alpha=0.8,
            label=VARIANT_LABELS[variant],
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Separation R [pixels]")
    ax.set_ylabel("Structure function")
    ax.set_title(title)
    ax.grid(alpha=0.2, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def analyze_getsources_source_masks(
    analyzer,
    hawc_file: str | Path,
    footprints_file: str | Path,
    source_catalog: str | Path,
    distance_pc: float = 0.0,
    fwhm_fallback: float = 18.2,
    pix_scale_fallback: float = 4.55,
    snr_threshold: float = 3.0,
    min_beams: float = 10.0,
    source_buffer_beams: float = 0.5,
) -> pd.DataFrame:
    """Calculate SF with sources, without sources, and for sources only.

    Results are written to the same ``hawc_report/plots/<object>`` directory
    used by ``HawcPolarizationAnalyzer.process_hawc_file``.
    """
    plt.rcParams["text.usetex"] = False
    hawc_file = Path(hawc_file)
    footprints_file = Path(footprints_file)
    source_catalog = Path(source_catalog)
    object_name = hawc_file.stem

    I, Q, U, wcs, err_I, err_Q, err_U, qual_I, hdr_fwhm, hdr_scale = analyzer.load_hawc_data(hawc_file)
    _check_same_fits_grid(hawc_file, footprints_file, I.shape)

    scales = analyzer.get_physical_scales(wcs, distance_pc, I.shape)
    fwhm_arcsec = float(hdr_fwhm if hdr_fwhm is not None else fwhm_fallback)
    pix_scale_arcsec = float(scales.get("pix_scale_arcsec") or hdr_scale or pix_scale_fallback)
    pixels_per_beam = 1.133 * (fwhm_arcsec / pix_scale_arcsec) ** 2

    plots_dir = analyzer.base_dir / "hawc_report" / "plots" / object_name
    plots_dir.mkdir(parents=True, exist_ok=True)

    segmentation = analyzer.detect_regions_by_snr(
        I.copy(), Q.copy(), U.copy(), err_Q.copy(), err_U.copy(),
        None if qual_I is None else qual_I.copy(), wcs, distance_pc, object_name,
        fwhm_arcsec=fwhm_arcsec, pix_scale_fallback=pix_scale_arcsec,
        snr_threshold=snr_threshold, min_beams=min_beams,
        save_path=str(plots_dir / f"{object_name}_getsources_stat_segmentation.pdf"),
    )
    valid_contours, snr_map, err_Q_map, err_U_map, ppb = segmentation[:5]
    min_area_pixels = float(segmentation[5]) if len(segmentation) > 5 else np.nan
    if not valid_contours:
        raise RuntimeError(f"No valid polarization-SNR regions found for {object_name}")

    region_masks: Dict[str, np.ndarray] = {
        f"Core_{index}": _mask_from_contour(I.shape, contour)
        for index, contour in enumerate(valid_contours, start=1)
    }
    if len(region_masks) > 1:
        region_masks["Combined_All_Cores"] = np.logical_or.reduce(list(region_masks.values()))

    labels = np.rint(np.squeeze(fits.getdata(footprints_file, 0))).astype(int)
    source_ids = read_getsources_catalog_ids(source_catalog)
    source_mask_raw = np.isin(labels, source_ids)

    buffer_radius_px = int(np.ceil(source_buffer_beams * fwhm_arcsec / pix_scale_arcsec))
    if buffer_radius_px > 0:
        kernel_size = 2 * buffer_radius_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        source_mask_buffered = cv2.dilate(source_mask_raw.astype(np.uint8), kernel).astype(bool)
    else:
        source_mask_buffered = source_mask_raw.copy()

    hawc_header = fits.getheader(hawc_file, 0)
    _write_mask(plots_dir / f"{object_name}_getsources_sources_mask_raw.fits", source_mask_raw, hawc_header)
    _write_mask(plots_dir / f"{object_name}_getsources_sources_mask_buffered.fits", source_mask_buffered, hawc_header)

    combined_region = np.logical_or.reduce(list(region_masks.values()))
    _plot_mask_diagnostic(
        I, combined_region, source_mask_raw, source_mask_buffered,
        plots_dir / f"{object_name}_getsources_masks.pdf",
    )

    sf_config = SFAnalysisConfig(
        r_col="R_pix", dphi_col="Dphi", npairs_col="Npairs", r_pc_col="R_pc",
        min_points=8, min_npairs=30, tail_fraction=0.25, smooth_window=3,
        rel_tol=0.08, sigma_tol=2.0, require_consecutive_tail_points=3,
        start_offset_pix=3,
    )

    finite_qu = np.isfinite(Q) & np.isfinite(U)
    summary_rows: list[dict] = []

    for region_name, region_mask in region_masks.items():
        variants = {
            "with_sources": region_mask,
            "without_sources": region_mask & ~source_mask_buffered,
            "sources_only": region_mask & source_mask_buffered,
        }
        variant_tables: Dict[str, pd.DataFrame] = {}

        for variant, variant_mask in variants.items():
            variant_mask = variant_mask & finite_qu
            area_pix, length_pix, n_components = _mask_geometry(variant_mask)
            mask_name = f"{region_name}_{variant}"
            _write_mask(plots_dir / f"{object_name}_{mask_name}_mask.fits", variant_mask, hawc_header)

            base_summary = {
                "object_name": object_name,
                "mask_name": mask_name,
                "dataset": variant,
                "variant": variant,
                "Area_pix": area_pix,
                "L_fit_px": length_pix,
                "n_components": n_components,
                "source_catalog": str(source_catalog),
                "source_count_catalog": len(source_ids),
                "source_ids": ",".join(map(str, source_ids)),
                "source_buffer_beams": source_buffer_beams,
                "source_buffer_radius_px": buffer_radius_px,
                "FWHM_arcsec": fwhm_arcsec,
                "pixel_scale_arcsec": pix_scale_arcsec,
                "pixels_per_beam": pixels_per_beam,
                "polarization_snr_threshold": snr_threshold,
                "Min_Area_Threshold_px": min_area_pixels,
                "Distance": distance_pc,
            }

            if area_pix < 2 or length_pix < 1:
                base_summary.update({
                    "status": "not_enough_pixels",
                    "notes": "Mask contains too few valid Q/U pixels",
                    "MA_proxy": np.nan,
                    "lA_fit_px": np.nan,
                })
                pd.DataFrame([base_summary]).to_csv(
                    plots_dir / f"{object_name}_{mask_name}_analysis.csv", index=False
                )
                summary_rows.append(base_summary)
                variant_tables[variant] = pd.DataFrame()
                continue

            local_max_r = max(5, int(2 * length_pix / 3))
            sf_data = analyzer.compute_structure_function(
                np.where(variant_mask, Q, np.nan),
                np.where(variant_mask, U, np.nan),
                R_list=np.arange(1, local_max_r + 1),
                pixels_per_beam=ppb,
            )

            if sf_data is None:
                base_summary.update({
                    "status": "no_valid_pairs",
                    "notes": "No valid Q/U pairs for the structure function",
                    "MA_proxy": np.nan,
                    "lA_fit_px": np.nan,
                })
                pd.DataFrame([base_summary]).to_csv(
                    plots_dir / f"{object_name}_{mask_name}_analysis.csv", index=False
                )
                summary_rows.append(base_summary)
                variant_tables[variant] = pd.DataFrame()
                continue

            raw_table = pd.DataFrame({
                "R_pix": sf_data["R_px"],
                "Dphi": sf_data["SF"],
                "Dphi_err": sf_data["SF_err"],
                "Npairs": sf_data["Npairs"],
            })
            raw_table["R_arcsec"] = raw_table["R_pix"] * pix_scale_arcsec
            raw_table["R_pc"] = raw_table["R_pix"] * scales["phys_scale_pc"]
            raw_table["object_name"] = object_name
            raw_table["mask_name"] = mask_name
            raw_table["dataset"] = variant
            raw_table.to_csv(plots_dir / f"{object_name}_{mask_name}_sf.csv", index=False)
            variant_tables[variant] = raw_table

            result, valid, extras = analyze_sf_table(
                raw_table, sf_config, plots_dir, l_pix_fallback=length_pix
            )
            result.object_name = object_name
            result.mask_name = mask_name
            result.dataset = variant

            break_pix = result.saturation_scale_pix
            if n_components == 1:
                mach = estimate_mach_from_break_and_length(
                    break_pix, length_pix, object_name=object_name, mask_name=mask_name
                )
                ma_proxy = mach.MA_proxy if mach.status == "ok" else np.nan
                mach_status = mach.status
            else:
                ma_proxy = np.nan
                mach_status = "not_interpreted_for_disconnected_mask"

            result.ma_proxy = ma_proxy
            result.l_fit_px = length_pix
            valid_intensity = I[variant_mask & np.isfinite(I)]
            summary = result.__dict__.copy()
            summary.update(base_summary)
            summary.update({
                "status": "ok" if len(valid) >= sf_config.min_points else "limited_sf",
                "MA_proxy": ma_proxy,
                "MA_status": mach_status,
                "lA_fit_px": break_pix,
                "R_max_pix": float(raw_table["R_pix"].max()),
                "I_max": float(np.nanmax(valid_intensity)) if valid_intensity.size else np.nan,
                "I_min": float(np.nanmin(valid_intensity)) if valid_intensity.size else np.nan,
                "I_median": float(np.nanmedian(valid_intensity)) if valid_intensity.size else np.nan,
            })
            pd.DataFrame([summary]).to_csv(
                plots_dir / f"{object_name}_{mask_name}_analysis.csv", index=False
            )
            summary_rows.append(summary)

            if not valid.empty:
                plot_sf_diagnostic(
                    valid, result, extras, sf_config,
                    title=f"SF: {object_name} ({mask_name.replace('_', ' ')})",
                    save_path=plots_dir / f"{object_name}_sf_{mask_name}.pdf",
                )

        _plot_variant_comparison(
            variant_tables,
            title=f"{object_name}: {region_name.replace('_', ' ')}",
            output_path=plots_dir / f"{object_name}_{region_name}_sources_comparison.pdf",
        )

    summary_table = pd.DataFrame(summary_rows)
    summary_table.to_csv(
        plots_dir / f"{object_name}_getsources_source_mask_summary.csv", index=False
    )
    print(f"GETSOURCES mask analysis saved to: {plots_dir}")
    print(summary_table[["mask_name", "Area_pix", "n_components", "status", "MA_proxy"]].to_string(index=False))
    return summary_table
