import json
from pathlib import Path
import unittest

import numpy as np
import torch

from panprompt3d import InteractivePredictor, build_panprompt3d
from panprompt3d.clicks import get_next_click3D_torch_2


class ModelTests(unittest.TestCase):
    def test_full_model_layout(self):
        with torch.device("meta"):
            model = build_panprompt3d()
        state = model.state_dict()
        self.assertEqual(len(state), 382)
        self.assertEqual(sum(t.numel() for t in state.values()), 100643524)
        self.assertTrue(any("triplet_attention" in name for name in state))
        self.assertTrue(any(".dsc." in name for name in state))

    def test_simulated_training_click(self):
        np.random.seed(3407)
        gt = torch.zeros((1, 1, 8, 8, 8), dtype=torch.long)
        gt[:, :, 2:6, 2:6, 2:6] = 1
        points, labels = get_next_click3D_torch_2(torch.zeros_like(gt), gt)
        point = points[0][0, 0]
        self.assertEqual(labels[0].item(), 1)
        self.assertEqual(gt[(0, 0) + tuple(point.tolist())].item(), 1)

    def test_click_requires_an_image(self):
        class Model:
            def to(self, _):
                return self

            def eval(self):
                return self

        predictor = InteractivePredictor(Model(), "cpu")
        with self.assertRaises(RuntimeError):
            predictor.click([64, 64, 64])

    def test_public_config_is_full(self):
        config = json.loads(
            (
                Path(__file__).resolve().parents[1] / "configs/full_merged.json"
            ).read_text()
        )
        self.assertTrue(config["use_cf_loss"])
        self.assertEqual(config["effective_interactions_per_batch"], 11)


if __name__ == "__main__":
    unittest.main()
