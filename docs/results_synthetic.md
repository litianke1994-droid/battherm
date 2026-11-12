# Synthetic sanity check (not a real-world result)

Run: `battherm run --config configs/synthetic.toml`, seed 42, single run, RTX 4060 Laptop.
Data: 45 simulated cycles (5 ambients × 9), split by whole cycle: 30 train / 5 val / 10 test.
Task: 120 s history + known 60 s future current → temperature rise over the next 60 s.

| model | RMSE (K) | MAE (K) | max abs (K) | RMSE @ 60 s (K) | skill @ 60 s |
|---|---:|---:|---:|---:|---:|
| persistence | 0.676 | 0.350 | 7.52 | 1.125 | 0.000 |
| physics | 0.153 | 0.095 | 1.80 | 0.228 | 0.959 |
| ridge | 0.197 | 0.119 | 3.30 | 0.338 | 0.910 |
| lstm | 0.091 | 0.068 | 0.99 | 0.127 | 0.987 |
| cnn | 0.083 | 0.065 | 0.57 | 0.101 | 0.992 |
| hybrid_lstm | 0.077 | 0.060 | 0.66 | 0.093 | 0.993 |

What this does and does not show:

* The pipeline works end to end and every model beats persistence; the ranking is plausible.
* The data come from a model whose structure is close to the physics baseline, so the physics
  model is unusually strong here. The fitted time constant is 635 s against 562 s used in the
  simulator: the polarisation heat `V1²/R1` is absorbed into the fitted parameters.
* Differences between lstm / cnn / hybrid (~0.01 K) are within what a different seed could
  change. Run several seeds before drawing conclusions.
* Ridge picked the smallest alpha in the grid (0.01) and sklearn warned about an
  ill-conditioned matrix; the grid edge should be checked.
* Real conclusions require the LG 18650HG2 protocols (profile and ambient hold-out).
