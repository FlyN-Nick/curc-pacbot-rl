import math
import torch
from torch import nn


def init_orthogonal(module: nn.Module):
    """
    Initializes model weights orthogonally.
    This has been shown to greatly improve training efficiency.
    """
    with torch.no_grad():
        for param in module.parameters():
            if len(param.size()) >= 2:
                torch.nn.init.orthogonal_(param.data)


class QNet(nn.Module):
    def __init__(self, obs_shape: torch.Size, action_count: int) -> None:
        super().__init__()

        def conv_act(in_channels: int, out_channels: int, kernel_size: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, padding="same"),
                # nn.BatchNorm2d(out_channels),
                nn.SiLU(),
            )

        obs_channels, _, _ = obs_shape
        self.backbone = nn.Sequential(
            conv_act(obs_channels, 16, 5),
            conv_act(16, 32, 3),
            nn.Conv2d(32, 64, 3, padding="same"),
        )
        self.fc = nn.Sequential(
            nn.SiLU(),
            nn.Linear(64, 128),
            nn.SiLU(),
        )
        self.value_head = nn.Sequential(
            # nn.Linear(128, 128),
            # nn.SiLU(),
            nn.Linear(128, 1),
        )
        self.advantage_head = nn.Sequential(
            # nn.Linear(128, 128),
            # nn.SiLU(),
            nn.Linear(128, action_count),
        )

        init_orthogonal(self)

        with torch.no_grad():
            # Have both heads output zeros at the start of training.
            for linear_layer in [self.value_head[-1], self.advantage_head[-1]]:
                linear_layer.weight.fill_(0)
                linear_layer.bias.fill_(0)

    def forward(self, input_batch: torch.Tensor) -> torch.Tensor:
        backbone_features = self.backbone(input_batch).amax(dim=(-2, -1))  # max pooling
        fc_features = self.fc(backbone_features)
        value = self.value_head(fc_features)
        advantages = self.advantage_head(fc_features)
        return value - advantages.mean(dim=1, keepdim=True) + advantages


class QNetV2(nn.Module):
    def __init__(self, obs_shape: torch.Size, action_count: int) -> None:
        super().__init__()

        def conv_block_pool(in_channels: int, out_channels: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding="same"),
                nn.MaxPool2d(2, ceil_mode=True),
                nn.SiLU(),
            )

        obs_channels, _, _ = obs_shape
        self.backbone = nn.Sequential(
            nn.Conv2d(obs_channels, 16, 5, padding="same"),
            nn.SiLU(),
            *conv_block_pool(16, 32),
            *conv_block_pool(32, 64),
            *conv_block_pool(64, 128),
            nn.Conv2d(128, 128, 3, groups=128 // 16, padding="same"),
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.SiLU(),
            nn.Linear(128, 256),
            nn.SiLU(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(256, 1),
        )
        self.advantage_head = nn.Sequential(
            nn.Linear(256, action_count),
        )

        init_orthogonal(self)

        with torch.no_grad():
            # Have both heads output zeros at the start of training.
            for linear_layer in [self.value_head[-1], self.advantage_head[-1]]:
                linear_layer.weight.fill_(0)
                linear_layer.bias.fill_(0)

    def forward(self, input_batch: torch.Tensor) -> torch.Tensor:
        backbone_features = self.backbone(input_batch)
        value = self.value_head(backbone_features)
        advantages = self.advantage_head(backbone_features)
        return value - advantages.mean(dim=1, keepdim=True) + advantages


class NetV2(nn.Module):
    def __init__(self, obs_shape: torch.Size, output_dim: int) -> None:
        super().__init__()

        def conv_block_pool(in_channels: int, out_channels: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding="same"),
                nn.MaxPool2d(2, ceil_mode=True),
                nn.SiLU(),
            )

        obs_channels, _, _ = obs_shape
        self.network = nn.Sequential(
            nn.Conv2d(obs_channels, 16, 5, padding="same"),
            nn.SiLU(),
            *conv_block_pool(16, 32),
            *conv_block_pool(32, 64),
            *conv_block_pool(64, 128),
            nn.Conv2d(128, 128, 3, groups=128 // 16, padding="same"),
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.SiLU(),
            nn.Linear(128, 256),
            nn.SiLU(),
            nn.Linear(256, output_dim),
        )

        init_orthogonal(self)

        with torch.no_grad():
            # Have the final layer output zeros at the start of training.
            linear_layer = self.network[-1]
            linear_layer.weight.fill_(0)
            linear_layer.bias.fill_(0)

    def forward(self, input_batch: torch.Tensor) -> torch.Tensor:
        return self.network(input_batch)


class DebugMLPQNet(nn.Module):
    def __init__(self, obs_shape: torch.Size, action_count: int) -> None:
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(math.prod(obs_shape), 128),
            nn.SiLU(),
            nn.Linear(128, action_count),
        )
        with torch.no_grad():
            # Have both heads output zeros at the start of training.
            self.mlp[-1].weight.fill_(0)
            self.mlp[-1].bias.fill_(0)

    def forward(self, input_batch: torch.Tensor) -> torch.Tensor:
        return self.mlp(input_batch.view(input_batch.shape[0], -1))


def _v2_backbone(obs_channels: int, widths: tuple[int, int, int, int], fc_width: int,
                 extra_convs: bool = False) -> nn.Sequential:
    """QNetV2's backbone with configurable widths; extra_convs adds a 3x3 conv to each stage."""

    def stage(in_channels: int, out_channels: int) -> list[nn.Module]:
        layers: list[nn.Module] = [nn.Conv2d(in_channels, out_channels, 3, padding="same")]
        if extra_convs:
            layers += [nn.SiLU(), nn.Conv2d(out_channels, out_channels, 3, padding="same")]
        return layers + [nn.MaxPool2d(2, ceil_mode=True), nn.SiLU()]

    w0, w1, w2, w3 = widths
    return nn.Sequential(
        nn.Conv2d(obs_channels, w0, 5, padding="same"),
        nn.SiLU(),
        *stage(w0, w1),
        *stage(w1, w2),
        *stage(w2, w3),
        nn.Conv2d(w3, w3, 3, groups=w3 // 16, padding="same"),
        nn.AdaptiveMaxPool2d((1, 1)),
        nn.Flatten(),
        nn.SiLU(),
        nn.Linear(w3, fc_width),
        nn.SiLU(),
    )


class _DuelingV2(nn.Module):
    """QNetV2's design (backbone, then dueling value and advantage heads) at a given size."""

    WIDTHS = (16, 32, 64, 128)
    FC_WIDTH = 256
    EXTRA_CONVS = False

    def __init__(self, obs_shape: torch.Size, action_count: int) -> None:
        super().__init__()
        self.backbone = _v2_backbone(obs_shape[0], self.WIDTHS, self.FC_WIDTH, self.EXTRA_CONVS)
        self.value_head = nn.Sequential(nn.Linear(self.FC_WIDTH, 1))
        self.advantage_head = nn.Sequential(nn.Linear(self.FC_WIDTH, action_count))
        init_orthogonal(self)
        with torch.no_grad():
            # Have both heads output zeros at the start of training.
            for linear_layer in [self.value_head[-1], self.advantage_head[-1]]:
                linear_layer.weight.fill_(0)
                linear_layer.bias.fill_(0)

    def forward(self, input_batch: torch.Tensor) -> torch.Tensor:
        features = self.backbone(input_batch)
        value = self.value_head(features)
        advantages = self.advantage_head(features)
        return value - advantages.mean(dim=1, keepdim=True) + advantages


class QNetV2Wide(_DuelingV2):
    """QNetV2 with twice the channels and a twice as wide fully connected layer."""

    WIDTHS = (32, 64, 128, 256)
    FC_WIDTH = 512


class QNetV2Deep(_DuelingV2):
    """QNetV2Wide with a second 3x3 convolution in each stage."""

    WIDTHS = (32, 64, 128, 256)
    FC_WIDTH = 512
    EXTRA_CONVS = True


class QRNetV2(nn.Module):
    """
    Quantile regression DQN (QR-DQN) with QNetV2's backbone: for each action, the network
    predicts NUM_QUANTILES quantiles of the return (at the midpoints (2i + 1) / 2N), dueling
    per quantile. forward() returns their mean, the Q values, so playing works as with any Q
    network; training uses quantiles().
    """

    NUM_QUANTILES = 32
    WIDTHS = (16, 32, 64, 128)
    FC_WIDTH = 256

    def __init__(self, obs_shape: torch.Size, action_count: int) -> None:
        super().__init__()
        self.action_count = action_count
        n = self.NUM_QUANTILES
        self.backbone = _v2_backbone(obs_shape[0], self.WIDTHS, self.FC_WIDTH)
        self.value_head = nn.Sequential(nn.Linear(self.FC_WIDTH, n))
        self.advantage_head = nn.Sequential(nn.Linear(self.FC_WIDTH, action_count * n))
        init_orthogonal(self)
        with torch.no_grad():
            for linear_layer in [self.value_head[-1], self.advantage_head[-1]]:
                linear_layer.weight.fill_(0)
                linear_layer.bias.fill_(0)

    def quantiles(self, input_batch: torch.Tensor) -> torch.Tensor:
        """Shape (batch, actions, NUM_QUANTILES)."""
        features = self.backbone(input_batch)
        value = self.value_head(features).unsqueeze(1)
        advantages = self.advantage_head(features).view(-1, self.action_count, self.NUM_QUANTILES)
        return value - advantages.mean(dim=1, keepdim=True) + advantages

    def forward(self, input_batch: torch.Tensor) -> torch.Tensor:
        return self.quantiles(input_batch).mean(dim=2)


class PVNetV2(nn.Module):
    """
    A policy and value network for AlphaZero-style training (src/algorithms/train_alphazero.py),
    with QNetV2's backbone: forward returns (policy logits (B, action_count), value (B,)).
    Both heads output zeros at the start (a uniform policy, value 0).
    """

    WIDTHS = (16, 32, 64, 128)
    FC_WIDTH = 256

    def __init__(self, obs_shape: torch.Size, action_count: int) -> None:
        super().__init__()
        self.backbone = _v2_backbone(obs_shape[0], self.WIDTHS, self.FC_WIDTH)
        self.policy_head = nn.Linear(self.FC_WIDTH, action_count)
        self.value_head = nn.Linear(self.FC_WIDTH, 1)
        init_orthogonal(self)
        with torch.no_grad():
            for linear_layer in (self.policy_head, self.value_head):
                linear_layer.weight.fill_(0)
                linear_layer.bias.fill_(0)

    def forward(self, input_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(input_batch)
        return self.policy_head(features), self.value_head(features).squeeze(1)


class PVNetV2Wide(PVNetV2):
    """PVNetV2 with QNetV2Wide's backbone: twice the channels and a twice as wide fully connected layer."""

    WIDTHS = (32, 64, 128, 256)
    FC_WIDTH = 512
