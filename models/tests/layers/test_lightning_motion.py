"""Exercise a configured motion head, its member outputs, and checkpoint reload."""

import numpy as np
import pytest
import torch
from hydra.utils import instantiate

pytest.importorskip("cv2")
pyproj = pytest.importorskip("pyproj")


def test_history_motion_and_checkpoint_reload(tmp_path):
    projection = "+proj=laea +lat_0=0 +lon_0=0 +ellps=WGS84 +units=m +no_defs"
    head = instantiate({
        "_target_": "anemoi.models.layers.lightning_motion.LightningMotionHead",
        "projection": projection, "logit_scale": 0.01, "grid_size": 1000.0,
    })
    yy, xx = np.indices((32, 32), dtype=np.float32)
    inverse = pyproj.Transformer.from_crs(projection, "EPSG:4326", always_xy=True)
    longitude, latitude = inverse.transform(((xx + 0.5) * 1000).ravel(), ((yy + 0.5) * 1000).ravel())
    coordinates = {"radar_latitudes": latitude, "radar_longitudes": longitude,
                   "lightning_latitudes": latitude, "lightning_longitudes": longitude}
    head.set_grid(**coordinates)
    radar = np.stack([8 * np.exp(-((xx - 11 - step * 0.6) ** 2 + (yy - 16) ** 2) / 8)
                      for step in range(7)]).reshape(7, -1)
    lightning = np.zeros((5, 32, 32), dtype=np.float32)
    for step in range(5):
        lightning[step, 16, round(11 + (step + 2) * 0.6)] = 1
    lightning = torch.from_numpy(lightning.reshape(5, -1))
    prediction = torch.zeros((25, 2, 1024), dtype=torch.float32)
    prediction[:, 1] = 0.01
    original = prediction.clone()
    rng_state = torch.get_rng_state()
    leads = np.arange(0, 121, 5)
    with torch.inference_mode():
        score = head(prediction, torch.from_numpy(radar), lightning, leads)
    assert torch.equal(prediction, original)
    assert torch.equal(torch.get_rng_state(), rng_state)
    assert torch.isnan(score[0]).all() and torch.isnan(score[7:]).all()
    assert torch.isfinite(score[1:7]).all()
    assert score[1:7].min() >= 0 and score[1:7].max() <= 1
    assert not torch.equal(score[1:7, 0], score[1:7, 1])
    excess = score[1:7, 0].numpy().reshape(6, 32, 32)
    excess = excess - excess.min(axis=(1, 2), keepdims=True)
    centroid = (excess * xx).sum(axis=(1, 2)) / excess.sum(axis=(1, 2))
    assert centroid[-1] - centroid[0] > 1
    path = tmp_path / "motion_head.ckpt"
    torch.save(head, path)
    restored = torch.load(path, weights_only=False)
    assert restored._lightning_index is None and restored._radar_index is None
    restored.set_grid(**coordinates)
    with torch.inference_mode():
        reloaded_score = restored(prediction, torch.from_numpy(radar), lightning, leads)
    torch.testing.assert_close(score, reloaded_score, rtol=0, atol=0, equal_nan=True)
