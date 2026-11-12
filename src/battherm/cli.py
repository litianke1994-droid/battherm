"""Command line entry point:

    battherm synth        --out data/processed/synthetic.csv.gz
    battherm prepare-hg2  --raw-dir data/raw/LG_HG2 --out data/processed/hg2.csv.gz
    battherm run          --config configs/synthetic.toml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from battherm.data.schema import cycle_table, validate


def _synth(a: argparse.Namespace) -> None:
    from battherm.data.synthetic import generate_dataset

    df = generate_dataset(cycles_per_ambient=a.cycles_per_ambient, duration_s=a.duration_s, seed=a.seed)
    validate(df)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(cycle_table(df).to_string(index=False))
    print(f"\nwrote {len(df):,} rows to {a.out}")


def _download_hg2(a: argparse.Namespace) -> None:
    from battherm.data.download import download_hg2

    paths = download_hg2(a.out_dir)
    print(f"{len(paths)} files ready under {a.out_dir}")


def _prepare_hg2(a: argparse.Namespace) -> None:
    from battherm.data.hg2 import DRIVE_CYCLE_RE, load_hg2_dir, quality_filter

    raw, skipped = load_hg2_dir(a.raw_dir, dt_s=a.dt, include=a.include or DRIVE_CYCLE_RE, max_gap_s=a.max_gap_s)
    df, checks = quality_filter(raw)
    validate(df)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    audit = cycle_table(raw).merge(checks, on="cycle_id")
    excluded = audit[audit["excluded"] != ""]
    if len(excluded):
        print(f"excluded {len(excluded)} cycles by the quality rules:")
        print(excluded[["cycle_id", "duration_s", "current_rms_A", "start_offset_K", "excluded"]].to_string(index=False), "\n")
    audit_path = a.out.with_name(a.out.name.split(".")[0] + "_audit.csv")
    audit.to_csv(audit_path, index=False)
    print(audit.to_string(index=False))
    if skipped:
        print(f"\nskipped {len(skipped)} files:")
        print("\n".join("  " + s for s in skipped))
    print(f"\nwrote {len(df):,} rows to {a.out}; audit table in {audit_path}")


def _run(a: argparse.Namespace) -> None:
    from battherm.pipeline import run

    run(a.config)


def _sweep(a: argparse.Namespace) -> None:
    from battherm.experiments import sweep

    sweep(a.config, a.set, a.models)


def _final(a: argparse.Namespace) -> None:
    from battherm.experiments import final

    final(a.config, a.seeds)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="battherm", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("synth", help="generate the synthetic dataset")
    s.add_argument("--out", type=Path, default=Path("data/processed/synthetic.csv.gz"))
    s.add_argument("--cycles-per-ambient", type=int, default=9)
    s.add_argument("--duration-s", type=int, default=3600)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=_synth)

    d = sub.add_parser("download-hg2", help="download the LG 18650HG2 drive-cycle CSVs (~580 MB, CC BY 4.0)")
    d.add_argument("--out-dir", type=Path, default=Path("data/raw/LG_HG2"))
    d.set_defaults(func=_download_hg2)

    h = sub.add_parser("prepare-hg2", help="convert raw LG 18650HG2 CSVs to the canonical schema")
    h.add_argument("--raw-dir", type=Path, required=True)
    h.add_argument("--out", type=Path, default=Path("data/processed/hg2.csv.gz"))
    h.add_argument("--dt", type=float, default=1.0, help="resampling step in seconds")
    h.add_argument("--max-gap-s", type=float, default=5.0, help="split a file where logging pauses longer than this")
    h.add_argument("--include", type=str, default=None, help="regex on profile names (default: drive cycles)")
    h.set_defaults(func=_prepare_hg2)

    r = sub.add_parser("run", help="run one experiment from a TOML config")
    r.add_argument("--config", type=Path, required=True)
    r.set_defaults(func=_run)

    w = sub.add_parser("sweep", help="grid search on the VALIDATION split, e.g. --set lstm.hidden=32,64")
    w.add_argument("--config", type=Path, required=True)
    w.add_argument("--set", action="append", default=[], help="dotted.key=v1,v2 (repeatable; values are TOML literals)")
    w.add_argument("--models", nargs="*", default=None, help="restrict to these models")
    w.set_defaults(func=_sweep)

    f = sub.add_parser("final", help="evaluate the frozen config on the TEST split over several seeds")
    f.add_argument("--config", type=Path, required=True)
    f.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    f.set_defaults(func=_final)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
