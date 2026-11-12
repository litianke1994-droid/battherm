from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from battherm.data.hg2 import load_hg2_csv, parse_ambient, parse_profile
from battherm.data.schema import COLUMNS, cycle_table, resample_uniform, validate
from battherm.data.synthetic import CellParams, generate_dataset, simulate_cycle
from battherm.data.windows import Scalers, WindowSpec, add_features, make_windows, split_cycles


@pytest.fixture(scope="module")
def synth() -> pd.DataFrame:
    return generate_dataset(ambients_C=(0.0, 25.0), cycles_per_ambient=6, duration_s=900, seed=1)


def test_synthetic_matches_schema(synth):
    validate(synth)
    assert list(synth.columns) == COLUMNS
    assert synth["temperature_C"].between(-30, 120).all()


def test_synthetic_heating_scales_with_current():
    p = CellParams(temp_noise_C=0.0, volt_noise_V=0.0, curr_noise_A=0.0, h_a_jitter=0.0)
    rise = {}
    for amps in (3.0, 6.0):  # 1C and 2C discharge for 20 min at 25 C
        sig = simulate_cycle(np.full(1200, -amps), 25.0, p, np.random.default_rng(0), soc0=1.0)
        rise[amps] = sig["temperature_C"][-1] - sig["temperature_C"][0]
    assert rise[3.0] > 0.5
    assert 3.0 < rise[6.0] / rise[3.0] < 5.0  # Joule heat ~ I^2 -> roughly 4x the rise


def test_window_alignment_on_known_ramp():
    n = 50
    df = pd.DataFrame({"cycle_id": "c", "profile": "p", "time_s": np.arange(n, dtype=float),
                       "current_A": np.arange(n, dtype=float), "voltage_V": 3.7,
                       "temperature_C": np.arange(n, dtype=float) * 0.1, "ambient_C": 25.0})
    ws = make_windows(add_features(df), WindowSpec(past_steps=5, horizon_steps=3, stride_steps=1))
    t = 4  # first forecast origin
    assert ws.past[0, -1, 0] == t  # last past row is the origin itself
    np.testing.assert_allclose(ws.future[0, :, 0], [t, t + 1, t + 2])  # current applied during steps t..t+H-1
    np.testing.assert_allclose(ws.y[0], [0.1, 0.2, 0.3], atol=1e-6)  # T[t+h] - T[t]
    assert len(ws) == n - 5 - 3 + 1


def test_windows_never_cross_cycles(synth):
    ws = make_windows(add_features(synth), WindowSpec(60, 30, 7))
    for cid in pd.unique(ws.cycle_id)[:5]:
        g = synth[synth["cycle_id"] == cid]
        m = ws.cycle_id == cid
        # every past temperature value in this cycle's windows must come from this cycle
        assert np.isin(ws.past[m][:, :, 2].ravel(), g["temperature_C"].to_numpy(np.float32)).all()


def test_split_is_disjoint_and_covers_all(synth):
    cycles = cycle_table(synth)
    s = split_cycles(cycles, {"test_fraction": 0.2, "val_fraction": 0.2}, seed=0)
    all_ids = set(s["train"]) | set(s["val"]) | set(s["test"])
    assert all_ids == set(cycles["cycle_id"])
    assert sum(len(v) for v in s.values()) == len(all_ids)
    s2 = split_cycles(cycles, {"test_profiles": ["cc"], "val_profiles": ["pulse"]}, seed=0)
    assert set(cycles.set_index("cycle_id").loc[s2["test"], "profile"]) == {"cc"}


def test_scalers_use_train_statistics_only(synth):
    df = add_features(synth)
    cycles = cycle_table(df)
    s = split_cycles(cycles, {"test_fraction": 0.3, "val_fraction": 0.2}, seed=0)
    spec = WindowSpec(60, 30, 10)
    train = make_windows(df[df["cycle_id"].isin(s["train"])], spec)
    full = make_windows(df, spec)
    sc = Scalers.fit(train)
    np.testing.assert_allclose(sc.past.mean, train.past.reshape(-1, train.past.shape[-1]).mean(0), rtol=1e-4)
    assert not np.allclose(sc.past.mean, full.past.reshape(-1, full.past.shape[-1]).mean(0))


def test_resample_uniform_grid():
    t = np.array([0.0, 0.1, 0.25, 1.05, 1.9, 2.2, 3.0])
    cyc = pd.DataFrame({"cycle_id": "c", "profile": "p", "time_s": t, "current_A": 1.0, "voltage_V": 3.7,
                        "temperature_C": 20.0, "ambient_C": 20.0})
    r = resample_uniform(cyc, 1.0)
    np.testing.assert_allclose(r["time_s"], [0, 1, 2, 3])


def test_hg2_parsers():
    assert parse_ambient(Path("raw/n10degC/551_UDDS.csv")) == -10.0
    assert parse_ambient(Path("raw/25degC/551_UDDS.csv")) == 25.0
    assert parse_profile(Path("551_Mixed8.csv")) == "Mixed8"
    assert parse_profile(Path("549_US06.csv")) == "US06"


def test_hg2_csv_with_metadata_and_units_row(tmp_path):
    folder = tmp_path / "10degC"
    folder.mkdir()
    lines = ["Cycler Data", "Test Name,demo", "", "Time Stamp,Step,Status,Prog Time,Voltage,Current,Temperature",
             ",,,hh:mm:ss.ms,V,A,degC"]
    for k in range(40):
        lines.append(f"x,1,DCH,00:00:{k * 0.5:06.3f},{3.9 - 0.001 * k:.4f},-2.0,{10 + 0.01 * k:.3f}")
    (folder / "600_UDDS.csv").write_text("\n".join(lines))
    df = load_hg2_csv(folder / "600_UDDS.csv", dt_s=1.0)
    validate(df)
    assert df["ambient_C"].iloc[0] == 10.0 and df["profile"].iloc[0] == "UDDS"
    assert len(df) == 20 and (df["current_A"] == -2.0).all()
