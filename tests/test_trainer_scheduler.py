import unittest

import torch.nn as nn

from src.trainer import build_optimizer_and_scheduler


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(4, 1)

    def get_param_groups(self, lr_head, lr_backbone):
        return [
            {"params": [self.layer.weight], "lr": lr_backbone},
            {"params": [self.layer.bias], "lr": lr_head},
        ]

    def freeze_epoch(self, epoch):
        return None


class SchedulerSetupTest(unittest.TestCase):
    def test_build_optimizer_and_scheduler_initializes_lr(self):
        model = DummyModel()
        optimizer, scheduler = build_optimizer_and_scheduler(
            model,
            lr_head=1e-3,
            lr_backbone=1e-4,
            weight_decay=1e-4,
            T_0=3,
            T_mult=2,
            last_epoch=2,
        )

        self.assertIn("initial_lr", optimizer.param_groups[0])
        self.assertEqual(optimizer.param_groups[0]["initial_lr"], optimizer.param_groups[0]["lr"])
        self.assertGreaterEqual(scheduler.last_epoch, 2)


if __name__ == "__main__":
    unittest.main()
