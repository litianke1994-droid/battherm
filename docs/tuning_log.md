# Tuning log (validation split only)

Protocol `hg2_profile_holdout`: train on HWFET, LA92, UDDS, Mixed1–6; validate on Mixed7
(5 cycles, −20…25 °C); US06 + Mixed8 are the test set and were not built or evaluated in any
run below. Single seed (42). RMSE in K over all 60 horizon steps.

Validation has only 5 cycles, so differences below ~0.003 K are within noise. The tuning was
deliberately small (4 rounds) to avoid over-fitting the validation set.

## Round 0: defaults (past 120 s, LSTM 64, CNN 32, constant LR)

| persistence | physics | ridge | lstm | cnn | hybrid_lstm |
|---:|---:|---:|---:|---:|---:|
| 0.1707 | 0.1287 | 0.1152 | 0.0925 | 0.0932 | 0.0959 |

## Round 1: history length

The fitted thermal time constant is ~530 s, so 120 s of history may be too short.

| past_s | persistence | physics | ridge | lstm | cnn | hybrid_lstm |
|---:|---:|---:|---:|---:|---:|---:|
| 120 | 0.1707 | 0.1287 | 0.1152 | 0.0925 | 0.0929 | 0.0959 |
| **300** | 0.1673 | 0.1289 | 0.1066 | **0.0895** | **0.0889** | 0.0952 |
| 600 | 0.1662 | 0.1276 | 0.1046 | 0.0986 | 0.0889 | 0.0964 |

(Persistence changes slightly because a longer history drops the first windows of each cycle.)
Chosen: 300 s.

## Round 2: hybrid input

| physics forecast also fed to the decoder | hybrid_lstm |
|---|---:|
| no (residual only) | 0.0952 |
| **yes** | **0.0947** |

No meaningful difference; kept "yes" (the network can see what it corrects).

## Round 3: LSTM width × learning-rate decay (patience 10)

| hidden | LR halving on plateau | lstm | hybrid_lstm |
|---:|---|---:|---:|
| 64 | no | 0.0895 | 0.0938 |
| 64 | yes | 0.0896 | 0.0932 |
| 128 | no | 0.0906 | 0.0939 |
| **128** | **yes** | **0.0867** | 0.0937 |

Chosen: hidden 128 with LR decay; epochs raised to 80 because several runs reached the 60-epoch cap.

## Round 4: CNN width (with LR decay, 80 epochs)

| channels | cnn |
|---:|---:|
| 32 | 0.0883 |
| **64** | **0.0874** |

## Frozen configuration

Committed before the first test evaluation (git commit "Freeze HG2 hyper-parameters…"):
past 300 s, horizon 60 s, LSTM hidden 128, CNN channels 64, Adam lr 1e-3 with halving on
plateau (patience 3), early stopping patience 10, ≤ 80 epochs, batch 512. The same settings are
used unchanged for the ambient hold-out protocol, which was never tuned.

Raw sweep tables: `outputs/sweeps/*.csv` (not committed; reproducible with `battherm sweep`).
