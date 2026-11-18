"""Compatibility imports for the original training code."""

from pacbot_rl_models.models import DebugMLPQNet, NetV2, QNet, QNetV2, QNetV2Deep, QNetV2Wide, QRNetV2, PVNetV2, init_orthogonal

__all__ = ["QNet", "QNetV2", "QNetV2Wide", "QNetV2Deep", "QRNetV2", "NetV2", "DebugMLPQNet", "init_orthogonal", "PVNetV2"]
