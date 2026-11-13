import numpy as np
import pandas as pd
import pytest
import torch

from battherm.data.synthetic import CellParams, generate_dataset
from battherm.data.windows import FUTURE_FEATURES, PAST_FEATURES, Scalers, WindowSpec, add_features, make_windows
from battherm.models.baselines import Persistence
from battherm.models.nets import build_model
from battherm.models.physics import LumpedThermalModel
from battherm.train import Arrays, fit, predict


@pytest.fixture(scope="module")
def clean_data() -> pd.DataFrame:
    p = CellParams(temp_noise_C=0.0, volt_noise_V=0.0, curr_noise_A=0.0, h_a_jitter=0.0)
    return add_features(generate_dataset(ambients_C=(0.0, 25.0, 40.0), cycles_per_ambient=3, duration_s=1800, seed=3, params=p))


def test_physics_recovers_time_constant_and_beats_persistence():
    # Without polarisation heat (tiny R1) the simulator has the same structure as the model,
    # so the fitted time constant must match the true one.
    p = CellParams(r1_ref_ohm=1e-5, temp_noise_C=0.0, volt_noise_V=0.0, curr_noise_A=0.0, h_a_jitter=0.0)
    data = add_features(generate_dataset(ambients_C=(0.0, 25.0, 40.0), cycles_per_ambient=3, duration_s=1800, seed=3, params=p))
    m = LumpedThermalModel(dt_s=1.0).fit(data, fit_steps=30)
    true_tau = p.heat_capacity_J_per_K / p.h_a_W_per_K
    assert abs(m.params()["time_constant_s"] - true_tau) / true_tau < 0.05
    ws = make_windows(data, WindowSpec(60, 60, 20))
    i = ws.future[:, :, FUTURE_FEATURES.index("current_A")]
    rmse_phys = np.sqrt(np.mean((m.predict(ws.t_now, ws.ambient, i) - ws.y) ** 2))
    rmse_pers = np.sqrt(np.mean((Persistence().predict(ws) - ws.y) ** 2))
    assert rmse_phys < 0.5 * rmse_pers


def test_arrhenius_physics_is_monotone_and_extrapolates():
    # Fit on 0..25 C only; the true temperature coefficient of R at 25 C is Ea / T^2 ~ 0.028 1/K.
    p = CellParams(r1_ref_ohm=1e-5, temp_noise_C=0.0, volt_noise_V=0.0, curr_noise_A=0.0, h_a_jitter=0.0)
    data = add_features(generate_dataset(ambients_C=(0.0, 10.0, 25.0, 40.0), cycles_per_ambient=3, duration_s=1800, seed=3, params=p))
    train, hot = data[data["ambient_C"] < 30], data[data["ambient_C"] == 40]
    m = LumpedThermalModel(dt_s=1.0, heat="arrhenius").fit(train, fit_steps=30)
    true_beta = p.activation_K / 298.15**2
    assert abs(m.params()["beta_per_K"] - true_beta) < 0.01
    T = np.linspace(-20, 60, 81)
    heat = m.rate(np.full_like(T, 3.0), T, T) - m.rate(np.zeros_like(T), T, T)
    assert np.all(np.diff(heat) < 0)  # less Joule heat in a hotter cell, also outside the fit range
    ws = make_windows(hot, WindowSpec(60, 60, 20))
    i = ws.future[:, :, FUTURE_FEATURES.index("current_A")]
    rmse_arr = np.sqrt(np.mean((m.predict(ws.t_now, ws.ambient, i) - ws.y) ** 2))
    rmse_pers = np.sqrt(np.mean((Persistence().predict(ws) - ws.y) ** 2))
    assert rmse_arr < 0.5 * rmse_pers


@pytest.mark.parametrize("kind", ["lstm", "cnn"])
def test_net_shapes_and_causality(kind):
    torch.manual_seed(0)
    model = build_model(kind, len(PAST_FEATURES), len(FUTURE_FEATURES), {}).eval()
    past, future = torch.randn(4, 30, len(PAST_FEATURES)), torch.randn(4, 12, len(FUTURE_FEATURES))
    out = model(past, future)
    assert out.shape == (4, 12)
    future2 = future.clone()
    future2[:, 8:] += 5.0  # changing late future inputs must not change early forecasts
    np.testing.assert_allclose(model(past, future2)[:, :8].detach(), out[:, :8].detach(), atol=1e-5)


def test_lstm_training_reduces_validation_error(clean_data):
    ws = make_windows(clean_data, WindowSpec(60, 30, 15))
    sc = Scalers.fit(ws)
    arr = Arrays(sc.past.transform(ws.past), sc.future.transform(ws.future), ws.y / sc.y_std, np.zeros_like(ws.y))
    model = build_model("lstm", len(PAST_FEATURES), len(FUTURE_FEATURES), {"lstm": {"hidden": 16}})
    before = np.sqrt(np.mean((predict(model, arr, torch.device("cpu")) - arr.y) ** 2))
    cfg = {"lr": 3e-3, "batch_size": 64, "epochs": 5, "patience": 5}
    fit(model, arr, arr, cfg, torch.device("cpu"), seed=0, log=lambda _: None)
    after = np.sqrt(np.mean((predict(model, arr, torch.device("cpu")) - arr.y) ** 2))
    assert after < before
