import torch


def test_covl_training_model_forward_returns_finite_loss():
    from covl_seg.engine.detectron2_runner import build_covl_training_model

    model = build_covl_training_model(num_classes=4, text_dim=16, seed=0)
    batch = [
        {
            "image": torch.randn(3, 32, 32),
            "sem_seg": torch.randint(0, 4, (32, 32)),
        },
        {
            "image": torch.randn(3, 32, 32),
            "sem_seg": torch.randint(0, 4, (32, 32)),
        },
    ]
    outputs = model(batch)
    assert "loss_total" in outputs
    assert torch.isfinite(outputs["loss_total"])
