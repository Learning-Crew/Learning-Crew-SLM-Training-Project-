import math
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import Dataset

from configs.model_config import ModelConfig
from configs.training_config import TrainingConfig
from training.checkpoint import save_checkpoint
from training.trainer import train


class TinyDataset(Dataset):
    def __init__(self, size=5):
        self.size = size
        self.visited = []

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        self.visited.append(index)
        tokens = torch.arange(index, index + 4)
        return tokens[:-1], tokens[1:]


class TrainingTests(unittest.TestCase):
    def setUp(self):
        temp_dir = TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.directory = Path(temp_dir.name)
        self.model_config = ModelConfig(vocab_size=8, hidden_size=4, num_layers=1, num_heads=1)
        self.config = TrainingConfig(batch_size=2, max_steps=7, checkpoint_every=2)

    def make_model(self):
        return nn.Sequential(nn.Embedding(8, 4), nn.Dropout(0.5), nn.Linear(4, 8))

    def train(self, model, dataset, **overrides):
        options = {"device": "cpu", "dataset_id": "tiny-v1", "checkpoint_dir": self.directory}
        options.update(overrides)
        config = options.pop("training_config", self.config)
        return train(model, dataset, self.model_config, config, **options)

    def test_training_and_resume_match_uninterrupted_run(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(73)
            dataset = TinyDataset()
            model = self.make_model()
            initial_weight = model[0].weight.detach().clone()

            with self.assertLogs("training.trainer", level="INFO") as full_logs:
                self.assertEqual(self.train(model, dataset), 7)

            self.assertFalse(torch.equal(model[0].weight, initial_weight))
            self.assertEqual(len(dataset.visited), 12)
            self.assertCountEqual(dataset.visited[:5], range(5))
            self.assertCountEqual(dataset.visited[5:10], range(5))
            self.assertEqual(
                sorted(path.name for path in self.directory.iterdir()),
                ["step_2.pt", "step_4.pt", "step_6.pt", "step_7.pt"],
            )
            logged_lrs = [float(line.rsplit("lr=", 1)[1]) for line in full_logs.output]
            self.assertEqual(len(logged_lrs), self.config.max_steps)
            for step, lr in enumerate(logged_lrs):
                expected_lr = self.config.learning_rate * (1 + math.cos(math.pi * step / 7)) / 2
                self.assertAlmostEqual(lr, expected_lr, places=12)
            final_state = torch.load(self.directory / "step_7.pt", weights_only=True)
            self.assertEqual(final_state["optimizer"]["param_groups"][0]["lr"], 0)

            for step, samples_seen in [(2, 4), (6, 10)]:
                with self.subTest(resume_step=step):
                    resumed_model = self.make_model()
                    resumed_dataset = TinyDataset()
                    with self.assertLogs("training.trainer", level="INFO") as resumed_logs:
                        self.assertEqual(
                            self.train(
                                resumed_model,
                                resumed_dataset,
                                checkpoint_dir=self.directory / f"resume_{step}",
                                resume_from=self.directory / f"step_{step}.pt",
                            ),
                            7,
                        )
                    self.assertEqual(resumed_dataset.visited, dataset.visited[samples_seen:])
                    self.assertEqual(resumed_logs.output, full_logs.output[step:])
                    for key, weight in model.state_dict().items():
                        torch.testing.assert_close(
                            resumed_model.state_dict()[key], weight, rtol=0, atol=0
                        )

    def test_empty_dataset_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "dataset must not be empty"):
            self.train(self.make_model(), TinyDataset(size=0))

    def test_resume_rejects_changed_config_and_data(self):
        self.train(self.make_model(), TinyDataset())
        with self.assertRaisesRegex(
            ValueError, r"training_config.max_steps, dataset_id, dataset_length"
        ):
            self.train(
                self.make_model(),
                TinyDataset(size=6),
                dataset_id="tiny-v2",
                training_config=replace(self.config, max_steps=8),
                resume_from=self.directory / "step_2.pt",
            )

    def test_failed_save_preserves_existing_checkpoint(self):
        path = self.directory / "step_2.pt"
        save_checkpoint(path, {"global_step": 2})

        def fail_save(state, temporary_path):
            temporary_path.write_bytes(b"partial checkpoint")
            raise OSError("save failed")

        with patch("training.checkpoint.torch.save", side_effect=fail_save):
            with self.assertRaisesRegex(OSError, "save failed"):
                save_checkpoint(path, {"global_step": 3})

        self.assertEqual(torch.load(path, weights_only=True), {"global_step": 2})
        self.assertEqual(list(self.directory.iterdir()), [path])
