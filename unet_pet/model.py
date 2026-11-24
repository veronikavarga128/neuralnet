"""U-Net architecture built from scratch (Ronneberger et al., 2015).

``(B, in_channels, H, W)`` image -> ``(B, num_classes, H, W)`` logits. No pretrained
weights and no external segmentation library are used (course rule, see AGENTS.md).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from unet_pet.config import Config


class DoubleConv(nn.Module):
    """(Conv3x3 -> BatchNorm -> ReLU) x 2, size-preserving (``padding=1``).

    The convolutions have no bias: the following BatchNorm subtracts the per-channel mean
    anyway, so a bias would be a redundant parameter.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """Encoder step: 2x2 max-pool (halves H, W) followed by ``DoubleConv``."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    """Decoder step: upsample x2, concatenate the skip connection, ``DoubleConv``.

    Both upsampling modes halve the channel count before the concatenation, so the
    ``DoubleConv`` always sees ``in_channels`` (= upsampled half + skip half):

    - ``bilinear=False``: learnable ``ConvTranspose2d`` (the original U-Net's "up-conv");
    - ``bilinear=True``: fixed bilinear interpolation + 1x1 conv; fewer parameters and no
      checkerboard artefacts, at the cost of a less flexible upsampling.
    """

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = False) -> None:
        super().__init__()
        half = in_channels // 2
        if bilinear:
            self.up: nn.Module = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(in_channels, half, kernel_size=1),
            )
        else:
            self.up = nn.ConvTranspose2d(in_channels, half, kernel_size=2, stride=2)
        self.conv = DoubleConv(half + out_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # When H or W was odd somewhere in the encoder, max-pooling floored it, so the
        # upsampled map can be 1 px smaller than the skip. Pad (right/bottom) to match.
        dh, dw = skip.shape[-2] - x.shape[-2], skip.shape[-1] - x.shape[-1]
        if dh or dw:
            x = F.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
        return self.conv(torch.cat([skip, x], dim=1))


class UNet(nn.Module):
    """U-Net: ``depth`` encoder steps, a bottleneck and ``depth`` decoder steps with skips.

    Channel widths double at every encoder step: ``base_channels * 2**i``; with the
    defaults (32, depth 4) that is 32 -> 64 -> 128 -> 256 -> 512 (bottleneck).

    The output are raw **logits**: ``CrossEntropyLoss`` applies log-softmax itself, and for
    prediction ``argmax`` over logits equals ``argmax`` over probabilities.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 3,
        base_channels: int = 32,
        depth: int = 4,
        bilinear: bool = False,
    ) -> None:
        super().__init__()
        if depth < 1 or base_channels < 1 or num_classes < 1 or in_channels < 1:
            raise ValueError("depth, base_channels, num_classes and in_channels must be >= 1")
        self.depth = depth
        widths = [base_channels * 2**i for i in range(depth + 1)]

        self.inc = DoubleConv(in_channels, widths[0])
        self.downs = nn.ModuleList(Down(widths[i], widths[i + 1]) for i in range(depth))
        self.ups = nn.ModuleList(
            Up(widths[i + 1], widths[i], bilinear) for i in reversed(range(depth))
        )
        self.head = nn.Conv2d(widths[0], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        min_size = 2**self.depth
        if min(x.shape[-2:]) < min_size:
            raise ValueError(
                f"Input {tuple(x.shape[-2:])} is too small for depth {self.depth}: "
                f"H and W must be >= {min_size}"
            )
        skips = [self.inc(x)]
        for down in self.downs:
            skips.append(down(skips[-1]))
        out = skips.pop()  # bottleneck
        for up in self.ups:
            out = up(out, skips.pop())
        return self.head(out)


def build_model(cfg: Config) -> UNet:
    """Create a ``UNet`` from the ``model`` section of the config."""
    m = cfg.model
    return UNet(
        in_channels=m.in_channels,
        num_classes=m.num_classes,
        base_channels=m.base_channels,
        depth=m.depth,
        bilinear=m.bilinear,
    )


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Number of (trainable) parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)
