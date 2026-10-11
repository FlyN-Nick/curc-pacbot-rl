import unittest

import torch

from models import DebugMLPQNet, NetV2, QNet, QNetV2
from pacbot_rl_models import models as package_models


class ModelPackageCompatibilityTests(unittest.TestCase):
    def test_legacy_names_are_reexports_of_installed_package_models(self):
        self.assertIs(QNet, package_models.QNet)
        self.assertIs(QNetV2, package_models.QNetV2)
        self.assertIs(NetV2, package_models.NetV2)
        self.assertIs(DebugMLPQNet, package_models.DebugMLPQNet)

    def test_debug_model_runs_on_cpu(self):
        model = DebugMLPQNet(torch.Size((8,)), action_count=4)
        output = model(torch.zeros((2, 8), device="cpu"))

        self.assertEqual(tuple(output.shape), (2, 4))


if __name__ == "__main__":
    unittest.main()
