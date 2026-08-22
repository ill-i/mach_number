#!/usr/bin/env python3
"""Command-line entry point for the automated HAWC+ analysis pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from hawc_pipeline import (
    PipelineParameters,
    analyze_object,
    build_latex_report,
    compile_latex_report,
    locate_getsources_products,
    prepare_object,
    run_getsources,
)


HERE = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare multi-extension HAWC+ FITS files, run the two-pass "
            "GETSOURCES extraction non-interactively, calculate polarization "
            "structure functions, and build a TeX/PDF report."
        )
    )
    parser.add_argument("--input", type=Path, default=HERE / "data_hawc", help="Directory with HAWC+ FITS files")
    parser.add_argument("--output", type=Path, default=HERE, help="Pipeline root (work/ and report/ are created here)")
    parser.add_argument(
        "--objects",
        nargs="*",
        default=None,
        help="Optional FITS stems or filenames to process; default: every *.fits",
    )
    parser.add_argument(
        "--getsources",
        type=Path,
        default=Path.home() / "bin/getsources/getsources",
        help="Path to the GETSOURCES executable",
    )
    parser.add_argument("--prepare-only", action="store_true", help="Only extract component and GETSOURCES input FITS files")
    parser.add_argument("--skip-getsources", action="store_true", help="Reuse final products already present under work/")
    parser.add_argument("--no-compile", action="store_true", help="Write TeX but do not compile PDF")
    parser.add_argument("--snr-threshold", type=float, default=3.0)
    parser.add_argument("--min-beams", type=float, default=10.0)
    parser.add_argument("--max-source-size", type=float, default=120.0, help="GETSOURCES maximum source FWHM [arcsec]")
    parser.add_argument("--sreliable", type=float, default=5.0)
    parser.add_argument("--stentative", type=float, default=3.0)
    parser.add_argument("--catalog", choices=("reliable", "relitent"), default="reliable")
    parser.add_argument("--distance-pc", type=float, default=0.0)
    return parser.parse_args()


def selected_files(input_dir: Path, requested: list[str] | None) -> list[Path]:
    files = sorted(input_dir.expanduser().resolve().glob("*.fits"))
    if not requested:
        return files
    wanted = {Path(item).name for item in requested} | {Path(item).stem for item in requested}
    return [path for path in files if path.name in wanted or path.stem in wanted]


def main() -> int:
    args = parse_args()
    parameters = PipelineParameters(
        snr_threshold=args.snr_threshold,
        min_beams=args.min_beams,
        max_source_size_arcsec=args.max_source_size,
        sreliable=args.sreliable,
        stentative=args.stentative,
        source_catalog_mode=args.catalog,
        distance_pc=args.distance_pc,
    )
    files = selected_files(args.input, args.objects)
    if not files:
        print(f"No FITS files selected in {args.input}", file=sys.stderr)
        return 2

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest: dict = {
        "input_directory": str(args.input.expanduser().resolve()),
        "output_directory": str(output),
        "parameters": asdict(parameters),
        "objects": [],
    }
    failures = 0
    analyzed = 0

    for index, path in enumerate(files, start=1):
        print(f"\n[{index}/{len(files)}] {path.name}")
        record = {"file": str(path), "status": "started"}
        try:
            prepared = prepare_object(path, output, parameters)
            record.update(
                {
                    "prepared_root": str(prepared.root),
                    "warnings": prepared.data.warnings,
                    "status": "prepared",
                }
            )
            if args.prepare_only:
                print(f"[PREPARED] {prepared.root}")
            else:
                products = (
                    locate_getsources_products(prepared, parameters)
                    if args.skip_getsources
                    else run_getsources(prepared, parameters, args.getsources)
                )
                summary = analyze_object(prepared, products, parameters)
                record.update(
                    {
                        "status": "analyzed",
                        "report_directory": str(prepared.report_dir),
                        "rows": len(summary),
                        "catalog": str(products.catalog),
                        "footprints": str(products.footprints),
                    }
                )
                analyzed += 1
                print(f"[ANALYZED] {prepared.report_dir}")
        except Exception as exc:
            failures += 1
            record.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
            print(f"[FAILED] {path.name}: {exc}", file=sys.stderr)
        manifest["objects"].append(record)
        (output / "pipeline_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if analyzed:
        tex = build_latex_report(output)
        manifest["tex_report"] = str(tex)
        print(f"[REPORT] {tex}")
        if not args.no_compile:
            pdf = compile_latex_report(tex)
            manifest["pdf_report"] = str(pdf)
            print(f"[REPORT] {pdf}")
        (output / "pipeline_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if failures:
        print(f"Completed with {failures} failed object(s); see pipeline_manifest.json", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
