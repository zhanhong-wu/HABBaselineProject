import tempfile
import unittest
from pathlib import Path

import torch
import torch.optim as optim

from hab_baseline.checkpoints import load_best_model, load_checkpoint, save_best_model, save_checkpoint
from hab_baseline.models import BASELINE_MODEL_NAMES, build_model


class ProjectTests(unittest.TestCase):
    def test_fifteen_legacy_baselines_are_registered(self):
        self.assertEqual(len(BASELINE_MODEL_NAMES), 15)
        self.assertEqual(len(set(BASELINE_MODEL_NAMES)), 15)

    def test_best_and_checkpoint_roundtrip(self):
        model = build_model("SimpleCNN_custom", in_channels=12, num_classes=2)
        optimizer = optim.AdamW(model.parameters(), lr=1e-4)
        images = torch.rand(1, 12, 32, 32)
        targets = torch.zeros(1, 32, 32, dtype=torch.long)
        loss = torch.nn.functional.cross_entropy(model(images), targets)
        loss.backward()
        optimizer.step()

        with tempfile.TemporaryDirectory() as directory:
            best_path = Path(directory) / "best.pth"
            checkpoint_path = Path(directory) / "checkpoint.pth"
            save_best_model(model, best_path)
            save_checkpoint(model, optimizer, epoch=6, best_composite_score=0.75,
                            is_best=True, path=checkpoint_path)

            restored = build_model("SimpleCNN_custom", in_channels=12, num_classes=2)
            best_info = load_best_model(restored, best_path)
            self.assertEqual(best_info["missing_keys"], [])
            self.assertEqual(best_info["unexpected_keys"], [])
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, restored.state_dict()[key]))

            restored_optimizer = optim.AdamW(restored.parameters(), lr=1e-4)
            checkpoint_info = load_checkpoint(restored, restored_optimizer, checkpoint_path)
            self.assertEqual(checkpoint_info["next_epoch"], 7)
            self.assertEqual(checkpoint_info["best_composite_score"], 0.75)
            self.assertGreater(checkpoint_info["optimizer_state_entries"], 0)


if __name__ == "__main__":
    unittest.main()
