import torch


def _make_model(num_classes: int = 7):
    from covl_seg.model.boundary_detect import BoundaryDetector
    from covl_seg.model.continual_backbone import ContinualBackbone
    from covl_seg.model.covl_seg_model import COVLSegModel
    from covl_seg.model.fusion import FusionHead
    from covl_seg.model.hciba_head import HCIBAHead

    backbone = ContinualBackbone(in_channels=3, hidden_dim=64, out_dim=32)
    head = HCIBAHead(in_dim=32, out_dim=16)
    boundary = BoundaryDetector(threshold=0.1)
    fusion = FusionHead(alpha=0.6, tau=0.7)
    return COVLSegModel(
        backbone=backbone,
        hciba_head=head,
        boundary_detector=boundary,
        fusion_head=fusion,
        num_classes=num_classes,
        text_dim=16,
    )


def test_covl_seg_model_forward_shape():
    model = _make_model(num_classes=5)
    images = torch.randn(2, 3, 48, 48)
    text_embeddings = torch.randn(5, 16)
    clip_logits = torch.randn(2, 5, 48, 48)

    outputs = model(images=images, text_embeddings=text_embeddings, clip_logits=clip_logits)
    assert "logits" in outputs
    assert outputs["logits"].shape == (2, 5, 48, 48)
    assert outputs["boundary_map"].shape == (2, 1, 48, 48)


def test_backward_no_nan():
    model = _make_model(num_classes=4)
    images = torch.randn(1, 3, 32, 32)
    text_embeddings = torch.randn(4, 16)
    targets = torch.randint(0, 4, (1, 32, 32))

    outputs = model(images=images, text_embeddings=text_embeddings, targets=targets)
    loss = outputs["loss"]
    assert torch.isfinite(loss)

    loss.backward()
    grad_norm = torch.stack(
        [
            param.grad.detach().norm()
            for param in model.parameters()
            if param.grad is not None
        ]
    ).sum()
    assert torch.isfinite(grad_norm)
