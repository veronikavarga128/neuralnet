"""Tests for unet_pet.model (CPU, small random tensors, a few seconds)."""

import pytest
import torch
from torch import nn

from unet_pet.config import load_config
from unet_pet.model import DoubleConv, Down, UNet, Up, build_model, count_parameters
from unet_pet.utils import set_seed


def test_blocks_shapes() -> None:
    x = torch.randn(2, 8, 32, 32)
    assert DoubleConv(8, 16)(x).shape == (2, 16, 32, 32)
    assert Down(8, 16)(x).shape == (2, 16, 16, 16)
    skip = torch.randn(2, 8, 64, 64)
    for bilinear in (False, True):
        assert Up(16, 8, bilinear)(torch.randn(2, 16, 32, 32), skip).shape == (2, 8, 64, 64)


@pytest.mark.parametrize("bilinear", [False, True])
def test_output_shape_default_size(bilinear: bool) -> None:
    model = UNet(base_channels=8, depth=4, bilinear=bilinear)
    assert model(torch.randn(2, 3, 128, 128)).shape == (2, 3, 128, 128)


@pytest.mark.parametrize("size", [(100, 100), (97, 131), (16, 16)])
def test_output_matches_any_input_size(size: tuple[int, int]) -> None:
    model = UNet(base_channels=8, depth=4).eval()
    with torch.no_grad():
        assert model(torch.randn(1, 3, *size)).shape == (1, 3, *size)


def test_too_small_input_raises() -> None:
    with pytest.raises(ValueError, match="too small"):
        UNet(base_channels=8, depth=4)(torch.randn(1, 3, 8, 8))


def test_output_is_logits_not_probabilities() -> None:
    set_seed(0)
    out = UNet(base_channels=8, depth=2).eval()(torch.randn(1, 3, 32, 32))
    assert not torch.allclose(out.sum(dim=1), torch.ones(1, 32, 32))


def test_default_config_parameter_count() -> None:
    model = build_model(load_config())
    n = count_parameters(model)
    assert 7_000_000 <= n <= 8_000_000, n
    bilinear = build_model(load_config(overrides={"model.bilinear": True}))
    assert count_parameters(bilinear) < n


def test_build_model_follows_config() -> None:
    cfg = load_config(overrides={"model.base_channels": 4, "model.depth": 2})
    model = build_model(cfg)
    assert isinstance(model, UNet) and len(model.downs) == 2
    assert model.head.out_channels == cfg.model.num_classes


def test_no_pretrained_or_external_modules() -> None:
    allowed = {"torch.nn.modules", "unet_pet.model"}
    for module in build_model(load_config()).modules():
        assert any(type(module).__module__.startswith(a) for a in allowed), type(module)


def test_every_parameter_receives_gradient() -> None:
    model = UNet(base_channels=4, depth=3)
    model(torch.randn(2, 3, 32, 32)).sum().backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, name


def test_can_overfit_a_single_batch() -> None:
    """A correct U-Net must memorise one small batch: loss drops and pixels are learned."""
    set_seed(0)
    images = torch.randn(2, 3, 32, 32)
    masks = torch.zeros(2, 32, 32, dtype=torch.long)
    masks[:, 8:24, 8:24] = 1  # a square "pet"
    masks[:, 8:24, 8] = 2  # with a border column
    model = UNet(base_channels=8, depth=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.CrossEntropyLoss()

    first = None
    for _ in range(60):
        optimizer.zero_grad()
        loss = loss_fn(model(images), masks)
        loss.backward()
        optimizer.step()
        first = loss.item() if first is None else first
    model.eval()
    with torch.no_grad():
        accuracy = (model(images).argmax(dim=1) == masks).float().mean().item()
    assert loss.item() < 0.2 * first
    assert accuracy > 0.95
