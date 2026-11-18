#!/usr/bin/env python
# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Target-noise amplitude tables, one per rollout step, from a run's spread–skill epoch dump.

The dump is the JSON-lines file written by ``anemoi.training.spread.callbacks.SpreadSkillLogger``
(``dump_path``): one record per split, epoch and rollout step with the epoch means of the squared
error of the ensemble mean (``ensmse``) and of the ensemble variance (``ensvar``) for every
model-output variable, in normalised units.

    A_v(r)² = c_v(r) · MSE_val,v(r) − MSE_train,v(r),      c_v(r) = <s²>_train / <s²>_val,

and 0 where that is negative. ``c_v`` removes the part of the validation-minus-training error
difference that comes from the two periods having different spread (the model's spread differs
between the training years and the validation year); ``--no-spread-correction`` gives the plain
difference ``MSE_val − MSE_train``.

The source run must have trained against CLEAN targets (no target noise: its training-side error
is inflated by the noise) and should have the rollout depth the tables are for, e.g. a short
rollout fine-tune of the model to be calibrated with the plain loss.

    python tools/spread_calibration/make_noise_tables.py --dump spread_skill_epochs_<id>.jsonl \\
        --steps 1 2 3 4 --last-n 3 --out-prefix tables/target_noise_step

Output per step: ``<prefix><step>.json`` = {"meta": {...}, "rmse": {variable: amplitude}} — the
``rmse`` mapping is what ``training.target_noise.amplitude_table`` reads.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def epoch_mean(records: dict[int, dict], epochs: list[int], key: str, variable: str) -> float:
    return sum(float(records[e][key][variable]) for e in epochs) / len(epochs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", required=True, help="spread_skill_epochs*.jsonl of the source run")
    ap.add_argument("--steps", type=int, nargs="+", required=True, help="1-based rollout steps (one table each)")
    ap.add_argument("--last-n", type=int, default=3, help="use the last N epochs that carry both splits")
    ap.add_argument("--epochs", type=int, nargs="*", default=None, help="explicit epochs instead of the last N")
    ap.add_argument("--out-prefix", required=True, help="tables are written to <prefix><step>.json")
    ap.add_argument("--dataset", default="data")
    ap.add_argument("--label", default="", help="free text stored in the tables' metadata")
    ap.add_argument("--no-spread-correction", action="store_true", help="A² = MSE_val − MSE_train (c_v = 1)")
    a = ap.parse_args()

    dump = Path(a.dump)
    recs = [json.loads(line) for line in dump.read_text().splitlines() if line.strip()]
    for step in a.steps:
        by: dict[str, dict[int, dict]] = {"train": {}, "val": {}}
        for r in recs:
            if int(r.get("rollout_step", 1)) == step and r.get("dataset", "data") == a.dataset:
                by[r["split"]][int(r["epoch"])] = r
        common = sorted(set(by["train"]) & set(by["val"]))
        chosen = a.epochs if a.epochs else common[-a.last_n :]
        missing = [e for e in chosen if e not in common]
        if missing or not chosen:
            msg = f"step {step}: epochs {missing or chosen} lack both splits (available {common})"
            raise SystemExit(msg)
        names = [k for k in by["val"][chosen[0]]["ensmse"] if k != "all"]
        table: dict[str, float] = {}
        c_factor: dict[str, float] = {}
        for v in names:
            mse_val = epoch_mean(by["val"], chosen, "ensmse", v)
            mse_train = epoch_mean(by["train"], chosen, "ensmse", v)
            c = 1.0
            if not a.no_spread_correction:
                var_val = epoch_mean(by["val"], chosen, "ensvar", v)
                var_train = epoch_mean(by["train"], chosen, "ensvar", v)
                c = var_train / var_val if var_val > 0 else 1.0
            c_factor[v] = c
            gap = c * mse_val - mse_train
            table[v] = math.sqrt(gap) if gap > 0 else 0.0
        out = Path(f"{a.out_prefix}{step}.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "source_dump": str(dump),
            "source_label": a.label,
            "epochs": chosen,
            "rollout_step": step,
            "space": "normalised",
            "definition": "A_v(r)^2 = c_v * MSE_val,v(r) - MSE_train,v(r), c_v = <s^2>_train / <s^2>_val "
            "(c_v = 1 with --no-spread-correction); 0 where negative. Use with amplitude_fraction 1.0.",
            "c_v": {v: round(c, 4) for v, c in c_factor.items()},
        }
        out.write_text(json.dumps({"meta": meta, "rmse": table}, indent=1))
        positive = sum(1 for x in table.values() if x > 0)
        print(f"wrote {out}: step {step}, epochs {chosen}, {positive}/{len(table)} variables with a positive amplitude")


if __name__ == "__main__":
    main()
