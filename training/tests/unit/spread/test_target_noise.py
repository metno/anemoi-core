# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests of the per-variable epoch dump (SpreadSkillLogger.dump_path) and of target-noise training."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp
import torch
from omegaconf import DictConfig

from .conftest import TOY_VARIABLES
from .conftest import make_index_collection
from .conftest import ring_moving_average
from anemoi.training.spread import SpreadScaledCRPS
from anemoi.training.spread import TrainingDiagnosticsLossWrapper
from anemoi.training.spread import spread_skill_ratio
from anemoi.training.spread.callbacks import EpochAccumulator
from anemoi.training.spread.callbacks import SpreadSkillLogger
from anemoi.training.spread.target_noise import TargetNoise
from anemoi.training.spread.target_noise import TargetNoiseEnsembleTraining
from anemoi.training.spread.target_noise import build_target_noise
from anemoi.training.spread.target_noise import load_amplitude_table

PINNED = {"tp": 1.0, "ssrd": 1.0, "tcw": 1.0, "q_500": 1.0, "tcc": 1.0}  # bounded subset of the toy variables


class FakeModule:
    def __init__(self, loss, n_steps: int = 1, nens: int = 2) -> None:  # noqa: ANN001
        self.loss = torch.nn.ModuleDict({"data": loss})
        self.task = SimpleNamespace(steps=lambda mode="training": tuple({"rollout_step": i} for i in range(n_steps)))
        self.nens_per_group = nens
        self.logger_enabled = True
        self.logged: dict[str, float] = {}

    def log(self, name, value, **kwargs):  # noqa: ANN001, ANN003, ANN201
        self.logged[name] = float(value)


class FakeTrainer:
    def __init__(self, global_step: int = 0, max_steps: int = 1000) -> None:
        self.callback_metrics: dict[str, torch.Tensor] = {}
        self.sanity_checking = False
        self.global_step = global_step
        self.current_epoch = 0
        self.max_steps = max_steps
        self.is_global_zero = True
        self.global_rank = 0


def _single_scale_wrapper(n_grid: int, a: float = 1.125, pinned: dict | None = None):  # noqa: ANN202
    """TrainingDiagnosticsLossWrapper(LossVariableMapper(SpreadScaledCRPS)) built through the factory with real indices."""
    from anemoi.training.losses.loss import get_loss_function

    indices = make_index_collection()
    cfg = DictConfig(
        {
            "_target_": "anemoi.training.spread.TrainingDiagnosticsLossWrapper",
            "loss": {
                "_target_": "anemoi.training.spread.SpreadScaledCRPS",
                "alpha": 0.95,
                "spread_adjustment": a,
                "per_variable_spread_adjustment": dict(PINNED if pinned is None else pinned),
                "scalers": ["node_weights"],
            },
        },
    )
    scalers = {"node_weights": (3, torch.full((n_grid,), 1.0 / n_grid))}
    return get_loss_function(cfg, scalers=scalers, data_indices=indices), indices


def _val_batch(indices, n_grid: int, ssr_by_var: dict[str, float]) -> tuple[torch.Tensor, torch.Tensor]:  # noqa: ANN001
    """(pred MODEL_OUTPUT, target DATA_FULL) with an exact per-variable corrected SSR:
    members = y + 1 ± s_v  → ⟨(x̄−y)²⟩ = 1, ⟨s²⟩ = 2 s_v² (M = 2, unbiased) → SSR = sqrt(3)·s_v."""
    out_names = list(indices.model.output.ordered_names)
    n_full = len(indices.name_to_index)
    target = torch.randn(1, 1, 1, n_grid, n_full)
    y_out = target[..., [indices.name_to_index[n] for n in out_names]]
    s = torch.tensor([ssr_by_var.get(n, 1.0) / np.sqrt(3.0) for n in out_names], dtype=torch.float32).view(1, 1, 1, 1, -1)
    pred = torch.cat([y_out + 1.0 + s, y_out + 1.0 - s], dim=2)
    return pred, target


LAYOUTS = {"pred_layout": "model_output", "target_layout": "data_full"}


# --------------------------------------------------------------------------- accumulator + dump
def test_epoch_accumulator_means() -> None:
    acc = EpochAccumulator()
    acc.add("data", 1, {"ensvar": torch.tensor([1.0, 2.0]), "ensmse": torch.tensor([4.0, 4.0])})
    acc.add("data", 1, {"ensvar": torch.tensor([3.0, 2.0]), "ensmse": torch.tensor([0.0, 4.0])})
    acc.add("data", 2, {"ensvar": torch.tensor([9.0, 9.0]), "ensmse": torch.tensor([1.0, 1.0])})
    red = acc.reduce()
    assert red[("data", 1)]["count"] == 2 and red[("data", 2)]["count"] == 1
    torch.testing.assert_close(red[("data", 1)]["ensvar"], torch.tensor([2.0, 2.0]))
    torch.testing.assert_close(red[("data", 1)]["ensmse"], torch.tensor([2.0, 4.0]))
    torch.testing.assert_close(red[("data", 2)]["ensvar"], torch.tensor([9.0, 9.0]))


def test_spread_skill_logger_dumps_all_variables(tmp_path: Path, n_grid: int) -> None:
    wrapped, indices = _single_scale_wrapper(n_grid)
    module, trainer = FakeModule(wrapped), FakeTrainer(global_step=5)
    dump = tmp_path / "spread_skill_epochs.jsonl"
    cb = SpreadSkillLogger(variables=["z_500"], dump_path=str(dump))
    want = {"z_500": 0.5, "t_850": 1.2, "u_850": 0.9, "tp": 0.7}
    for b in range(3):
        cb.on_validation_batch_start(trainer, module, {}, b)
        pred, target = _val_batch(indices, n_grid, want)
        wrapped(pred, target, **LAYOUTS)
        cb.on_validation_batch_end(trainer, module, None, {}, b)
    cb.on_validation_epoch_end(trainer, module)
    recs = [json.loads(line) for line in dump.read_text().splitlines()]
    assert len(recs) == 1 and recs[0]["split"] == "val" and recs[0]["n_batches"] == 3 and recs[0]["global_step"] == 5
    rec = recs[0]
    assert set(rec["ssr"]) == set(indices.model.output.ordered_names) | {"all"}
    for name, ssr in want.items():
        assert rec["ssr"][name] == pytest.approx(ssr, rel=1e-4)
        assert rec["ensmse"][name] == pytest.approx(1.0, rel=1e-4)
        assert rec["ensvar"][name] == pytest.approx(2 * (ssr / np.sqrt(3)) ** 2, rel=1e-4)
    assert rec["ssr"]["all"] == pytest.approx(spread_skill_ratio(rec["ensvar"]["all"], rec["ensmse"]["all"], 2))
    # the accumulator is reset after the dump; the train side dumps too
    assert not cb._acc["val"].sums
    cb.on_train_batch_start(trainer, module, {}, 0)
    wrapped(pred, target, **LAYOUTS)
    cb.on_train_batch_end(trainer, module, None, {"data": torch.zeros(1, 1, 1, 1, 1)}, 0)
    cb.on_train_epoch_end(trainer, module)
    recs = [json.loads(line) for line in dump.read_text().splitlines()]
    assert [r["split"] for r in recs] == ["val", "train"] and recs[1]["n_batches"] == 1


# --------------------------------------------------------------------------- adaptive per-variable a
def test_wrapper_resolves_single_scale_overrides(n_grid: int) -> None:
    wrapped, indices = _single_scale_wrapper(n_grid, a=1.3)
    leaf = wrapped.spread_leaves[0]
    assert isinstance(leaf, SpreadScaledCRPS) and leaf.variable_index == indices.model.output.name_to_index
    pred, target = _val_batch(indices, n_grid, {})
    out = wrapped(pred, target, **LAYOUTS)
    # reference: widen every non-pinned variable by 1.3, pinned ones by 1.0, then plain CRPS
    ref = SpreadScaledCRPS(alpha=0.95)
    ref.add_scaler(3, torch.full((n_grid,), 1.0 / n_grid), name="node_weights")
    names = indices.model.output.ordered_names
    a = torch.tensor([PINNED.get(n, 1.3) for n in names]).view(1, 1, 1, 1, -1)
    y_out = target[..., [indices.name_to_index[n] for n in names]]
    torch.testing.assert_close(out, ref(SpreadScaledCRPS.widen(pred, a, dim=2), y_out, squash_mode="avg"))  # the mapper forces the variable *mean*


# --------------------------------------------------------------------------- target noise
@pytest.fixture
def ring_smoother(n_grid: int) -> sp.csr_matrix:
    return sp.csr_matrix(ring_moving_average(n_grid, 9))


def test_target_noise_amplitude_shards_and_determinism(ring_smoother: sp.csr_matrix, n_grid: int) -> None:
    indices = make_index_collection()
    table = {n: 2.0 for n in indices.data.output.ordered_names}
    table["z_500"] = 4.0
    tn = build_target_noise(
        {"smoother": ring_smoother, "amplitude_table": table, "amplitude_fraction": 0.5, "exclude_variables": ["u_850"], "seed": 7},
        data_indices=indices,
    )
    n2i = indices.name_to_index
    assert tn.amplitudes_full[n2i["z_500"]] == 2.0 and tn.amplitudes_full[n2i["t_850"]] == 1.0
    assert tn.amplitudes_full[n2i["lsm"]] == 0.0 and tn.amplitudes_full[n2i["u_850"]] == 0.0  # forcing / excluded

    # marginal std of the coloured unit noise is 1 at every node (statistical, 20000 fields)
    field = tn.coloured_unit_noise(20000, device=torch.device("cpu"), seed=1)
    assert field.shape == (n_grid, 20000)
    assert torch.allclose(field.std(dim=1), torch.ones(n_grid), atol=0.03)
    # spatially correlated (neighbours) — moving average of width 9
    corr = torch.corrcoef(torch.stack([field[0], field[1]]))[0, 1].item()
    assert 0.8 < corr < 0.95

    y = torch.randn(2, 1, 1, n_grid, len(n2i))
    out = tn.perturb(y, global_step=3, rollout_step=0, group_id=0)
    assert torch.equal(out[..., n2i["lsm"]], y[..., n2i["lsm"]]) and torch.equal(out[..., n2i["u_850"]], y[..., n2i["u_850"]])
    assert not torch.equal(out[..., n2i["z_500"]], y[..., n2i["z_500"]])
    torch.testing.assert_close(out, tn.perturb(y, global_step=3, rollout_step=0, group_id=0))  # forward == recompute
    assert not torch.equal(out, tn.perturb(y, global_step=4, rollout_step=0, group_id=0))
    assert not torch.equal(out, tn.perturb(y, global_step=3, rollout_step=1, group_id=0))
    assert not torch.equal(out, tn.perturb(y, global_step=3, rollout_step=0, group_id=1))
    # perturbation is identical for every target "member" slot and has the configured std (many samples)
    d = torch.stack([tn.perturb(y, global_step=s, rollout_step=0, group_id=0)[..., n2i["z_500"]] - y[..., n2i["z_500"]] for s in range(300)])
    assert d.std().item() == pytest.approx(2.0, rel=0.05)
    # shard consistency: the shard's noise equals the slice of the full-grid noise
    sl = slice(10, 40)
    shard = tn.perturb(y[:, :, :, sl], global_step=3, rollout_step=0, group_id=0, grid_shard_slice=sl)
    torch.testing.assert_close(shard, out[:, :, :, sl])
    with pytest.raises(ValueError):
        tn.perturb(y[:, :, :, sl], global_step=3, rollout_step=0, group_id=0)  # size mismatch without a slice
    # annealing
    tn.anneal_steps = 100
    assert tn.amplitude_factor(0) == 1.0 and tn.amplitude_factor(50) == 0.5 and tn.amplitude_factor(1000) == 0.0
    torch.testing.assert_close(tn.perturb(y, global_step=1000, rollout_step=0, group_id=0), y)


def test_target_noise_resampled_every_visit(ring_smoother: sp.csr_matrix, n_grid: int) -> None:
    """r2 item L: eta must be independent across visits of the SAME sample, and must not depend on the
    sample's content/identity — only on (global_step, rollout_step, group)."""
    indices = make_index_collection()
    tn = build_target_noise({"smoother": ring_smoother, "amplitude_table": {n: 1.0 for n in indices.data.output.ordered_names}}, data_indices=indices)
    # bs=32 so the smoothed fields carry enough effective dof (~9-node correlation length on a 96-ring)
    y = torch.randn(32, 1, 1, n_grid, len(indices.name_to_index))
    eta = {s: tn.perturb(y, global_step=s, rollout_step=0, group_id=0) - y for s in (100, 5000, 90000)}
    # different visits (steps) -> independent noise (uncorrelated, not merely unequal)
    for a in eta.values():
        for b in eta.values():
            if a is b:
                continue
            c = torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1].abs().item()
            assert c < 0.15, c
    # the noise must NOT depend on the sample: two different batches at the same step get the SAME eta
    y2 = torch.randn_like(y) * 3.0
    torch.testing.assert_close(tn.perturb(y2, global_step=100, rollout_step=0, group_id=0) - y2, eta[100])
    # and a global torch.manual_seed cannot make two visits collide (private generator)
    torch.manual_seed(0)
    e1 = tn.perturb(y, global_step=7, rollout_step=0, group_id=0) - y
    torch.manual_seed(0)
    e2 = tn.perturb(y, global_step=8, rollout_step=0, group_id=0) - y
    assert not torch.allclose(e1, e2)


def test_target_noise_table_validation(ring_smoother: sp.csr_matrix, tmp_path: Path) -> None:
    indices = make_index_collection()
    with pytest.raises(KeyError, match="lacks"):
        build_target_noise({"smoother": ring_smoother, "amplitude_table": {"z_500": 1.0}}, data_indices=indices)
    with pytest.raises(TypeError, match="unknown keys"):
        build_target_noise({"smoother": ring_smoother, "amplitude_table": {"z_500": 1.0}, "sigma": 3}, data_indices=indices)
    f = tmp_path / "t.json"
    f.write_text(json.dumps({"meta": {"x": 1}, "rmse": {"a": 1.5, "b": "2.5"}}))
    assert load_amplitude_table(f) == {"a": 1.5, "b": 2.5}


def test_target_noise_method_only_perturbs_training(ring_smoother: sp.csr_matrix, n_grid: int, monkeypatch: pytest.MonkeyPatch) -> None:
    """The EnsembleTraining override: noise in training mode only, never in validation; state threaded through."""
    from anemoi.training.train.methods.ensemble import EnsembleTraining

    indices = make_index_collection()
    tn = build_target_noise({"smoother": ring_smoother, "amplitude_table": {n: 1.0 for n in indices.data.output.ordered_names}}, data_indices=indices)
    obj = TargetNoiseEnsembleTraining.__new__(TargetNoiseEnsembleTraining)  # no anemoi model needed for the hook
    obj.target_noise_enabled = True
    obj.target_noise = {"data": tn}
    obj.grid_shard_slice = {"data": None}
    obj.ens_comm_group_id = 2
    monkeypatch.setattr(TargetNoiseEnsembleTraining, "global_step", property(lambda self: 11))
    seen = {}

    def fake_super(self, y_pred, y, dataset_name, rollout_step=None, validation_mode=False, **kw):  # noqa: ANN001, ANN202
        seen["y"] = y
        return "ok"

    monkeypatch.setattr(EnsembleTraining, "compute_dataset_loss_metrics", fake_super)
    y = torch.randn(1, 1, 1, n_grid, len(indices.name_to_index))
    obj.training = True
    assert obj.compute_dataset_loss_metrics(None, y, "data", rollout_step=0, validation_mode=False) == "ok"
    torch.testing.assert_close(seen["y"], tn.perturb(y, global_step=11, rollout_step=0, group_id=2))
    obj.compute_dataset_loss_metrics(None, y, "data", rollout_step=0, validation_mode=True)
    assert seen["y"] is y  # validation: untouched
    obj.training = False
    obj.compute_dataset_loss_metrics(None, y, "data", rollout_step=0, validation_mode=False)
    assert seen["y"] is y


def test_target_noise_include_variables(ring_smoother: sp.csr_matrix) -> None:
    """``include_variables`` concentrates the noise on the gap family — amplitude 0 for every
    other output variable (no table entry needed for them), mutually exclusive with exclude, names checked."""
    indices = make_index_collection()
    n2i = indices.name_to_index
    out = list(indices.data.output.ordered_names)
    family = ["z_500"] + ([v for v in out if v.startswith("z_") and v != "z_500"][:1])
    table = {v: 2.0 for v in family}
    table["z_500"] = 4.0
    tn = build_target_noise(
        {"smoother": ring_smoother, "amplitude_table": table, "amplitude_fraction": 0.5, "include_variables": family, "seed": 3},
        data_indices=indices,
    )
    assert tn.amplitudes_full[n2i["z_500"]] == 2.0
    for v in out:
        if v not in family:
            assert tn.amplitudes_full[n2i[v]] == 0.0, v
    assert set(int(i) for i in tn.active_full_idx) == {n2i[v] for v in family}
    with pytest.raises(ValueError, match="OR exclude"):
        build_target_noise({"smoother": ring_smoother, "amplitude_table": {"z_500": 1.0}, "include_variables": ["z_500"], "exclude_variables": ["t_850"]}, data_indices=indices)
    with pytest.raises(KeyError, match="not model outputs"):
        build_target_noise({"smoother": ring_smoother, "amplitude_table": {"z_500": 1.0}, "include_variables": ["z_500", "nonesuch"]}, data_indices=indices)


def test_target_noise_per_step_tables(ring_smoother: sp.csr_matrix) -> None:
    """A LIST amplitude_table gives per-rollout-step amplitudes — row selected by the
    0-based rollout step, clamped to the last beyond; active set = union over steps; single table
    unchanged (back-compat)."""
    indices = make_index_collection()
    n2i = indices.name_to_index
    out = list(indices.data.output.ordered_names)
    t0 = {"z_500": 2.0}
    t1 = {"z_500": 4.0}
    tn = build_target_noise(
        {"smoother": ring_smoother, "amplitude_table": [t0, t1], "amplitude_fraction": 1.0,
         "include_variables": ["z_500"], "seed": 5},
        data_indices=indices,
    )
    assert tn.n_amp_steps == 2
    assert tn.amplitudes_full[n2i["z_500"]] == 2.0  # step-0 accessor
    assert set(int(i) for i in tn.active_full_idx) == {n2i["z_500"]}
    y = torch.zeros(1, 1, 2, ring_smoother.shape[0], max(n2i.values()) + 1)
    outs = {}
    for r in (0, 1, 5):  # 5 -> clamp to last (row 1)
        # pool over many global steps: each step draws a fresh correlated field (few effective dof),
        # so a single realisation's std is noisy — the amplitude ratio holds in expectation
        vals = [tn.perturb(y, global_step=g, rollout_step=r, group_id=0)[..., n2i["z_500"]] for g in range(100)]
        outs[r] = torch.stack(vals).std().item()
        assert vals[0].abs().max().item() > 0.0
        assert tn.perturb(y, global_step=0, rollout_step=r, group_id=0)[..., n2i["t_850"]].abs().max().item() == 0.0
    assert outs[1] / outs[0] == pytest.approx(2.0, rel=0.1)  # 4.0 vs 2.0 amplitude
    # clamp-to-last: same amplitude row (seed differs, distribution identical)
    assert outs[5] == pytest.approx(outs[1], rel=0.1)
    # union active set: a variable dosed only at step 1
    tn2 = build_target_noise(
        {"smoother": ring_smoother, "amplitude_table": [{"z_500": 2.0, "t_850": 0.0}, {"z_500": 2.0, "t_850": 3.0}],
         "amplitude_fraction": 1.0, "include_variables": ["z_500", "t_850"], "seed": 5},
        data_indices=indices,
    )
    assert set(int(i) for i in tn2.active_full_idx) == {n2i["z_500"], n2i["t_850"]}
    p0 = tn2.perturb(y, global_step=3, rollout_step=0, group_id=0)
    p1 = tn2.perturb(y, global_step=3, rollout_step=1, group_id=0)
    assert p0[..., n2i["t_850"]].abs().max().item() == 0.0  # zero-dosed at step 0
    assert p1[..., n2i["t_850"]].abs().max().item() > 0.0


def test_target_noise_dose_map(ring_smoother: sp.csr_matrix, n_grid: int, tmp_path: Path) -> None:
    """A static spatial dose map w_v(x) multiplies the perturbation; w ≡ 1 reproduces
    today's behaviour BITWISE; a map of 2 on one half of the grid doubles the perturbation there (same seed); the
    shard path applies the shard's rows; every perturbed variable needs a map; the .npz round trip works."""
    indices = make_index_collection()
    n2i = indices.name_to_index
    out = list(indices.data.output.ordered_names)
    base = {"smoother": ring_smoother, "amplitude_table": {n: 1.0 for n in out}, "amplitude_fraction": 1.0, "seed": 11}
    tn0 = build_target_noise(base, data_indices=indices)
    y = torch.randn(2, 1, 2, n_grid, max(n2i.values()) + 1)
    ref = tn0.perturb(y, global_step=3, rollout_step=1, group_id=0)
    # w ≡ 1 → bitwise identical (the map is dropped at construction)
    ones = {"names": out, "w": np.ones((len(out), n_grid), dtype=np.float32), "lat": np.zeros(n_grid)}
    tn1 = build_target_noise({**base, "dose_map": ones}, data_indices=indices)
    assert tn1._dose_map_cpu is None
    assert torch.equal(tn1.perturb(y, global_step=3, rollout_step=1, group_id=0), ref)
    # a map of 2 on the first half, 0.5 on the second: eta_map == eta * w node by node (same seed → same field)
    w = np.ones((len(out), n_grid), dtype=np.float32)
    w[:, : n_grid // 2] = 2.0
    w[:, n_grid // 2:] = 0.5
    w[out.index("t_850")] = 1.0
    tn2 = build_target_noise({**base, "dose_map": {"names": out, "w": w, "lat": np.zeros(n_grid)}}, data_indices=indices)
    got = tn2.perturb(y, global_step=3, rollout_step=1, group_id=0)
    eta_ref = ref - y
    eta_got = got - y
    wz = torch.from_numpy(w[out.index("z_500")])
    torch.testing.assert_close(eta_got[..., n2i["z_500"]], eta_ref[..., n2i["z_500"]] * wz)
    torch.testing.assert_close(eta_got[..., n2i["t_850"]], eta_ref[..., n2i["t_850"]])
    assert torch.equal(got[..., n2i["lsm"]], y[..., n2i["lsm"]])
    # shard path
    sl = slice(3, n_grid // 2 + 5)
    shard = tn2.perturb(y[:, :, :, sl], global_step=3, rollout_step=1, group_id=0, grid_shard_slice=sl)
    torch.testing.assert_close(shard, got[:, :, :, sl])
    # .npz round trip (the dose-map builder's layout) and the missing-variable error
    f = tmp_path / "map.npz"
    np.savez(f, names=np.array(out), w=w, lat=np.zeros(n_grid), lon=np.zeros(n_grid))
    tn3 = build_target_noise({**base, "dose_map": str(f)}, data_indices=indices)
    torch.testing.assert_close(tn3.perturb(y, global_step=3, rollout_step=1, group_id=0), got)
    with pytest.raises(KeyError):
        build_target_noise({**base, "dose_map": {"names": ["z_500"], "w": w[:1], "lat": np.zeros(n_grid)}}, data_indices=indices)
    # an excluded variable needs no map
    build_target_noise({**base, "include_variables": ["z_500"], "dose_map": {"names": ["z_500"], "w": w[:1], "lat": np.zeros(n_grid)}}, data_indices=indices)
    with pytest.raises(ValueError):
        TargetNoise(ring_smoother, torch.ones(max(n2i.values()) + 1), dose_map=torch.ones(n_grid + 1, max(n2i.values()) + 1))


