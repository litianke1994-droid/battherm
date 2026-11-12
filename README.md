# battherm: physics-informed deep learning for Li-ion cell temperature forecasting

Forecast the surface temperature of a Li-ion cell **60 s ahead** from what a battery
management system (BMS) already measures, plus the planned load. The project compares a
grey-box physics model, linear and deep-learning forecasters, and a **hybrid model in which
an LSTM learns only the residual of the physics model**.

Why it matters for thermal management: a cooling controller that knows the temperature
rise a minute in advance can pre-cool before a fast-charge or a hill climb, instead of
reacting after the cell is already hot.

## Task

At forecast origin *t* (uniform step *dt*):

| | content | shape |
|---|---|---|
| past inputs | current, voltage, temperature, ambient, current² over the last 120 s | `(W, 5)` |
| future inputs | planned current and current² for the next 60 s | `(H, 2)` |
| target | temperature rise `T[t+h] − T[t]`, h = 1..H | `(H,)` |

Errors on the rise are identical to errors on absolute temperature (`T[t]` is known).

## Models

| name | idea |
|---|---|
| `persistence` | `T[t+h] = T[t]`; the bar every model must clear |
| `physics` | lumped thermal ODE `dT/dt = I²·R(T)/C − (T − T_amb)/τ`, 4 linear parameters fitted by least squares on the integral form, rolled forward with the future current |
| `ridge` | ridge regression on the flattened, standardised windows (alpha chosen on validation) |
| `lstm` | encoder–decoder LSTM: encoder reads the past, decoder consumes future current step by step; increment head + cumulative sum |
| `cnn` | dilated causal 1D-CNN encoder + causal CNN decoder over the horizon |
| `hybrid_lstm` | `physics + LSTM(residual)`: the network only corrects what the physics model misses |

## Data

1. **Synthetic** (`battherm synth`): 1-RC equivalent circuit + lumped thermal model with
   Arrhenius resistance, three load types (pulse, drive-like, constant-current) at
   −10/0/10/25/40 °C. Ground truth is known, so it is used to debug the pipeline and in unit
   tests. **Synthetic results are sanity checks, not evidence of real-world performance.**
2. **LG 18650HG2** (public, McMaster University; Kollmeyer, Vidal, Naguib, Skells, 2020,
   Mendeley Data, doi:10.17632/cp3473x7xv): drive cycles (UDDS, HWFET, LA92, US06,
   Mixed1–8) at several chamber temperatures with measured cell temperature. Download it
   yourself, check its licence, and place the original CSVs under `data/raw/LG_HG2/`
   (one folder per ambient, e.g. `25degC/`, `n10degC/`). Raw and processed data are
   git-ignored.

## Evaluation protocol (no leakage)

* **Whole cycles** are assigned to train/val/test; windows from one measurement never appear
  in two splits (overlapping neighbouring windows would otherwise leak).
* Two HG2 protocols: unseen **load profiles** (`hg2_profile_holdout.toml`: test US06 +
  Mixed8) and unseen **ambient temperature** (`hg2_ambient_holdout.toml`: test 40 °C).
* Scalers, the physics parameters and ridge alpha are fitted on **train/val only**; the test
  split is touched once, at the end.
* Reported in kelvin: RMSE, MAE, max error, RMSE per horizon step, per ambient and per
  profile, plus skill vs persistence at the final step.

## Quick start

```bash
python -m venv .venv
.venv/Scripts/activate            # Linux/macOS: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128   # or the CPU wheel
pip install -e .[dev]
pytest                            # unit tests incl. leakage and causality checks

battherm synth                    # -> data/processed/synthetic.csv.gz
battherm run --config configs/synthetic.toml

battherm prepare-hg2 --raw-dir data/raw/LG_HG2   # prints an audit table; read it
battherm run --config configs/hg2_profile_holdout.toml
battherm run --config configs/hg2_ambient_holdout.toml
```

Each run writes `outputs/<name>_<timestamp>/` with the config copy, split, scalers, model
weights, learning histories, metric tables and figures.

## Results

Synthetic sanity check (seed 42; see `docs/results_synthetic.md`). HG2 results: to be added
after the data audit — do not quote synthetic numbers as real performance.

## Repository layout

```
configs/                 experiment definitions (TOML)
src/battherm/
  data/schema.py         canonical columns, validation, resampling
  data/synthetic.py      electro-thermal simulator
  data/hg2.py            LG 18650HG2 loader
  data/windows.py        windowing, cycle-level split, train-only scaling
  models/physics.py      lumped thermal model
  models/baselines.py    persistence, ridge
  models/nets.py         encoder-decoder LSTM, causal CNN
  train.py               training loop, early stopping
  evaluate.py            metrics
  plots.py               figures
  pipeline.py            end-to-end experiment
  cli.py                 `battherm` command
tests/                   pytest suite
docs/                    design notes
```

## Limitations

* Single lumped node: surface temperature only, no core–surface gradient, no entropic heat.
* The future current is assumed known (planned load); with a predicted load the error grows.
* Resampling averages current, so `mean(I)²` slightly underestimates Joule heat for fast
  fluctuations.
* One cell type. Transfer to other chemistries or pack-level cooling is untested.

## License

MIT (code). Datasets keep their own licences.
