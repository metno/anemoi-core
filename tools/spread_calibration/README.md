# Spread-calibration tools

Builders for the inputs of `anemoi.training.spread` (see the user guide page
"Spread calibration of ensemble models" in the anemoi-training documentation).

| Script | Output | Used by |
|---|---|---|
| `build_smoothers.py` | Gaussian smoothing matrices on the model grid, scipy CSR `.npz` | `multiscale_config.loss_matrices`, `training.target_noise.smoother` |
| `make_noise_tables.py` | one amplitude table per rollout step, JSON `{"rmse": {variable: amplitude}}` | `training.target_noise.amplitude_table` |

A dose map is an `.npz` with `names` (nvar), `w` (nvar, n_grid; area mean of `w²` equal to 1 per
variable) and `lat` (degrees). It is built from per-location in-sample and out-of-sample errors of
the model, which depends on how forecasts are produced and verified; no builder is included.
