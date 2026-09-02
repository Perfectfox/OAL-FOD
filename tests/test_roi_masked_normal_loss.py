import torch

from fod_recon_ad.native_losses import compute_roi_masked_normal_loss


def test_roi_masked_normal_loss_has_zero_gradient_outside_roi():
    target = torch.randn(1, 4, 4, 4)
    prediction = torch.randn(1, 4, 4, 4, requires_grad=True)
    roi = torch.zeros(1, 1, 4, 4)
    roi[:, :, 1:3, 1:3] = 1.0

    loss, _ = compute_roi_masked_normal_loss(
        ([target], [prediction]),
        roi,
        architecture="inpformer",
        step=1,
        normal_loss="native",
        prototype_loss_weight=0.0,
    )
    loss.backward()

    outside = ~roi.expand_as(prediction).bool()
    inside = roi.expand_as(prediction).bool()
    assert torch.count_nonzero(prediction.grad[outside]) == 0
    assert torch.count_nonzero(prediction.grad[inside]) > 0


def test_roi_masked_gather_uses_only_fully_valid_tokens():
    target = torch.randn(1, 4, 2, 2)
    prediction = target.detach().clone().requires_grad_(True)
    gather_distance = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0]], requires_grad=True
    )
    roi = torch.zeros(1, 1, 4, 4)
    roi[:, :, :2, :2] = 1.0

    loss, diagnostics = compute_roi_masked_normal_loss(
        ([target], [prediction], gather_distance.mean()),
        roi,
        architecture="inpformer",
        step=1,
        normal_loss="native",
        prototype_loss_weight=1.0,
        gather_distance=gather_distance,
    )
    loss.backward()

    assert diagnostics["roi_masked_gather_loss"] == 1.0
    assert gather_distance.grad[0, 0] == 1.0
    assert torch.count_nonzero(gather_distance.grad[0, 1:]) == 0


def test_roi_masked_gather_ignores_scalar_model_gather():
    target = torch.randn(1, 4, 2, 2)
    prediction = target.detach().clone().requires_grad_(True)
    native_distance = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0]], requires_grad=True
    )
    unused_model_gather = torch.tensor(9.0, requires_grad=True)
    roi = torch.zeros(1, 1, 4, 4)
    roi[:, :, :2, :2] = 1.0

    loss, diagnostics = compute_roi_masked_normal_loss(
        ([target], [prediction], unused_model_gather),
        roi,
        architecture="inpformer",
        step=1,
        normal_loss="native",
        prototype_loss_weight=0.2,
        gather_distance=native_distance,
    )
    loss.backward()

    assert diagnostics["roi_masked_gather_loss"] == 1.0
    assert torch.isclose(native_distance.grad[0, 0], torch.tensor(0.2))
    assert torch.count_nonzero(native_distance.grad[0, 1:]) == 0
    assert unused_model_gather.grad is None


def test_explicit_roi_aware_guided_gather_takes_precedence():
    target = torch.randn(1, 4, 2, 2)
    prediction = target.detach().clone().requires_grad_(True)
    native_distance = torch.ones(1, 4, requires_grad=True)
    guided_loss = torch.tensor(2.5, requires_grad=True)
    roi = torch.ones(1, 1, 4, 4)

    loss, diagnostics = compute_roi_masked_normal_loss(
        ([target], [prediction], guided_loss),
        roi,
        architecture="inpformer",
        step=1,
        normal_loss="native",
        prototype_loss_weight=0.2,
        gather_distance=native_distance,
        roi_aware_gather_loss=guided_loss,
    )
    loss.backward()

    assert diagnostics["roi_masked_gather_loss"] == 2.5
    assert diagnostics["roi_aware_guided_gather"] == 1.0
    assert torch.isclose(guided_loss.grad, torch.tensor(0.2))
    assert native_distance.grad is None
