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


def _prepare_hg2(a: argparse.Namespace) -> None:
    from battherm.data.hg2 import DRIVE_CYCLE_RE, load_hg2_dir

    df, skipped = load_hg2_dir(a.raw_dir, dt_s=a.dt, include=a.include or DRIVE_CYCLE_RE, max_gap_s=a.max_gap_s)
    validate(df)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    audit = cycle_table(df)
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


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="battherm", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("synth", help="generate the synthetic dataset")
    s.add_argument("--out", type=Path, default=Path("data/processed/synthetic.csv.gz"))
    s.add_argument("--cycles-per-ambient", type=int, default=9)
    s.add_argument("--duration-s", type=int, default=3600)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=_synth)

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

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
