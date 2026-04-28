import torch
import torch.nn as nn


def _make_model():
    return nn.Linear(4, 2)


def _make_loader(n_batches=5, batch_size=3):
    class _DS(torch.utils.data.Dataset):
        def __init__(self, length):
            self.length = length

        def __len__(self):
            return self.length

        def __getitem__(self, idx):
            return torch.randn(4), torch.randint(0, 2, (1,)).item()

    ds = _DS(n_batches * batch_size)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size)


def _loss_fn(outputs, targets):
    return nn.functional.cross_entropy(outputs, targets)


def test_ewc_computes_fisher():
    from covl_seg.continual.ewc import EWCRegularizer

    model = _make_model()
    reg = EWCRegularizer(model, lambda_ewc=10.0)
    loader = _make_loader()
    reg.compute_fisher(loader, _loss_fn, n_samples=5)

    fisher_vals = torch.cat([v.flatten() for v in reg._fisher.values()])
    assert (fisher_vals > 0).any(), "Fisher diagonal should be non-zero for trainable params"


def test_ewc_no_penalty_before_consolidate():
    from covl_seg.continual.ewc import EWCRegularizer

    model = _make_model()
    reg = EWCRegularizer(model, lambda_ewc=10.0)
    loader = _make_loader()
    reg.compute_fisher(loader, _loss_fn, n_samples=5)

    penalty = reg.penalty(model)
    assert penalty.item() == 0.0, "Penalty must be 0 before consolidate"


def test_ewc_loss_penalizes_drift():
    from covl_seg.continual.ewc import EWCRegularizer

    model = _make_model()
    reg = EWCRegularizer(model, lambda_ewc=10.0)
    loader = _make_loader()
    reg.compute_fisher(loader, _loss_fn, n_samples=5)
    reg.consolidate()

    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.5)

    penalty = reg.penalty(model)
    assert penalty.item() > 0, "Penalty must be positive after weight drift"


def test_ewc_state_dict_roundtrip():
    from covl_seg.continual.ewc import EWCRegularizer

    model = _make_model()
    reg = EWCRegularizer(model, lambda_ewc=10.0)
    loader = _make_loader()
    reg.compute_fisher(loader, _loss_fn, n_samples=5)
    reg.consolidate()

    penalty_before = reg.penalty(model).item()

    state = reg.state_dict()

    model2 = _make_model()
    reg2 = EWCRegularizer(model2, lambda_ewc=10.0)
    reg2.load_state_dict(state)

    with torch.no_grad():
        for p in model2.parameters():
            p.add_(torch.randn_like(p) * 0.5)

    penalty_after = reg2.penalty(model2).item()
    assert penalty_after > 0, "Loaded regularizer should still compute positive penalty"