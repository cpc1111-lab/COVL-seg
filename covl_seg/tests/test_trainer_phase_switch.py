import torch


def test_phase_switch_freeze_unfreeze_policy():
    from covl_seg.engine.trainer import Phase, PhaseController

    f_s = torch.nn.Linear(8, 8)
    phi = torch.nn.Linear(8, 8)
    mine = torch.nn.Linear(8, 1)

    controller = PhaseController(f_s=f_s, phi=phi, mine=mine)

    controller.set_phase(Phase.PHASE1)
    assert all(not p.requires_grad for p in f_s.parameters())
    assert all(p.requires_grad for p in phi.parameters())
    assert all(p.requires_grad for p in mine.parameters())

    controller.set_phase(Phase.PHASE2)
    assert all(p.requires_grad for p in f_s.parameters())
    assert all(p.requires_grad for p in phi.parameters())
    assert all(not p.requires_grad for p in mine.parameters())
