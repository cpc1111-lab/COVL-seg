from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


def _load_runtime_continual_losses_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "vendor"
        / "covl_seg_d2_runtime"
        / "cat_seg"
        / "continual_losses.py"
    )
    spec = importlib.util.spec_from_file_location("runtime_continual_losses", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_fake_detectron2_modules(monkeypatch):
    detectron2_mod = types.ModuleType("detectron2")

    config_mod = types.ModuleType("detectron2.config")
    config_mod.configurable = lambda fn: fn

    data_mod = types.ModuleType("detectron2.data")
    data_mod.MetadataCatalog = object()

    class _Registry:
        def register(self):
            def _deco(cls):
                return cls

            return _deco

    modeling_mod = types.ModuleType("detectron2.modeling")
    modeling_mod.META_ARCH_REGISTRY = _Registry()
    modeling_mod.build_backbone = lambda cfg: None
    modeling_mod.build_sem_seg_head = lambda cfg, shape: None

    backbone_mod = types.ModuleType("detectron2.modeling.backbone")
    backbone_mod.Backbone = nn.Module

    post_mod = types.ModuleType("detectron2.modeling.postprocessing")
    post_mod.sem_seg_postprocess = lambda output, image_size, height, width: output

    structures_mod = types.ModuleType("detectron2.structures")

    class _ImageList:
        def __init__(self, tensor, image_sizes):
            self.tensor = tensor
            self.image_sizes = image_sizes

        @staticmethod
        def from_tensors(tensors, size_divisibility):
            del size_divisibility
            return _ImageList(
                torch.stack(tensors, dim=0),
                [tuple(t.shape[-2:]) for t in tensors],
            )

    structures_mod.ImageList = _ImageList

    memory_mod = types.ModuleType("detectron2.utils.memory")
    memory_mod._ignore_torch_cuda_oom = lambda fn: fn

    utils_mod = types.ModuleType("detectron2.utils")

    einops_mod = types.ModuleType("einops")

    def _rearrange(tensor, pattern, **axes_lengths):
        if pattern == "B (H W) C -> B C H W":
            batch_size, flattened, channels = tensor.shape
            height = int(axes_lengths["H"])
            width = flattened // max(height, 1)
            return tensor.reshape(batch_size, height, width, channels).permute(0, 3, 1, 2)
        if pattern == "(H W) B C -> B C H W":
            flattened, batch_size, channels = tensor.shape
            height = int(axes_lengths["H"])
            width = flattened // max(height, 1)
            return tensor.reshape(height, width, batch_size, channels).permute(2, 3, 0, 1)
        raise NotImplementedError(pattern)

    einops_mod.rearrange = _rearrange

    monkeypatch.setitem(sys.modules, "detectron2", detectron2_mod)
    monkeypatch.setitem(sys.modules, "detectron2.config", config_mod)
    monkeypatch.setitem(sys.modules, "detectron2.data", data_mod)
    monkeypatch.setitem(sys.modules, "detectron2.modeling", modeling_mod)
    monkeypatch.setitem(sys.modules, "detectron2.modeling.backbone", backbone_mod)
    monkeypatch.setitem(sys.modules, "detectron2.modeling.postprocessing", post_mod)
    monkeypatch.setitem(sys.modules, "detectron2.structures", structures_mod)
    monkeypatch.setitem(sys.modules, "detectron2.utils", utils_mod)
    monkeypatch.setitem(sys.modules, "detectron2.utils.memory", memory_mod)
    monkeypatch.setitem(sys.modules, "einops", einops_mod)


def _load_runtime_cat_seg_model_module(monkeypatch):
    _install_fake_detectron2_modules(monkeypatch)

    cat_seg_root = (
        Path(__file__).resolve().parents[1]
        / "vendor"
        / "covl_seg_d2_runtime"
        / "cat_seg"
    )
    module_path = cat_seg_root / "cat_seg_model.py"

    package_name = "runtime_cat_seg"
    package_mod = types.ModuleType(package_name)
    package_mod.__path__ = [str(cat_seg_root)]
    monkeypatch.setitem(sys.modules, package_name, package_mod)

    utils_package = types.ModuleType(f"{package_name}.utils")
    utils_package.__path__ = [str(cat_seg_root / "utils")]
    monkeypatch.setitem(sys.modules, f"{package_name}.utils", utils_package)

    module_name = f"{package_name}.cat_seg_model"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_runtime_training_model(module):
    cat_seg_cls = module.CATSeg
    model = cat_seg_cls.__new__(cat_seg_cls)
    nn.Module.__init__(model)

    model.pixel_mean = torch.zeros(3, 1, 1)
    model.pixel_std = torch.ones(3, 1, 1)
    model.clip_pixel_mean = torch.zeros(3, 1, 1)
    model.clip_pixel_std = torch.ones(3, 1, 1)
    model.size_divisibility = 1
    model.sliding_window = False
    model.clip_resolution = (24, 24)
    model.proj_dim = 4
    model.upsample1 = nn.Identity()
    model.upsample2 = nn.Identity()
    model.layers = []

    model.visible_class_indexes = None
    model.old_class_indexes = []
    model.lambda_old_kd = 0.0
    model.distill_temp = 1.0
    model.enable_ciba = False
    model.enable_ctr = False
    model.beta_star = 0.0
    model.gamma_clip = 0.0
    model.lambda0_ctr = 0.1
    model.background_ids = []
    model.mine_critic = None
    model.old_teacher_weights = ""
    model._old_teacher = None
    model._old_teacher_load_attempted = False

    class _FakeClipModel(nn.Module):
        def __init__(self, owner):
            super().__init__()
            self._owner = owner

        def encode_image(self, images, dense=True):
            del dense
            batch_size = images.shape[0]
            token_count = 1 + 24 * 24
            channel_count = self._owner.proj_dim
            self._owner.layers = [
                torch.ones(token_count, batch_size, channel_count),
                torch.ones(token_count, batch_size, channel_count),
            ]
            return torch.zeros(batch_size, token_count, channel_count)

    class _FakeSemSegHead(nn.Module):
        def __init__(self, owner):
            super().__init__()
            self.ignore_value = 255
            self.num_classes = 3
            self.teacher_scale = nn.Parameter(torch.tensor(1.0))
            self.predictor = types.SimpleNamespace(
                clip_model=_FakeClipModel(owner),
                text_features=torch.randn(3, owner.proj_dim),
            )

        def forward(self, clip_features, features):
            del clip_features, features
            return self.teacher_scale * torch.randn(1, self.num_classes, 24, 24)

    model.sem_seg_head = _FakeSemSegHead(model)
    model.train()
    return model


def _build_training_batch():
    return [
        {
            "image": torch.randn(3, 24, 24),
            "sem_seg": torch.randint(0, 3, (24, 24), dtype=torch.long),
        }
    ]


def test_kd_loss_on_class_indexes_is_finite_and_non_negative_for_valid_indexes():
    module = _load_runtime_continual_losses_module()

    student_logits = torch.randn(2, 5, 3, 3)
    teacher_logits = torch.randn(2, 5, 3, 3)
    class_indexes = [0, 2, 4]

    loss = module.kd_loss_on_class_indexes(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        class_indexes=class_indexes,
        temperature=2.0,
    )

    assert torch.isfinite(loss)
    assert float(loss.item()) >= 0.0


def test_kd_loss_on_class_indexes_matches_per_pixel_batchmean_for_4d_logits():
    module = _load_runtime_continual_losses_module()

    student_logits = torch.randn(2, 5, 4, 3)
    teacher_logits = torch.randn(2, 5, 4, 3)
    class_indexes = [1, 3, 4]
    temperature = 2.5

    loss = module.kd_loss_on_class_indexes(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        class_indexes=class_indexes,
        temperature=temperature,
    )

    student_selected = student_logits.index_select(dim=1, index=torch.tensor(class_indexes))
    teacher_selected = teacher_logits.index_select(dim=1, index=torch.tensor(class_indexes))
    student_flat = student_selected.permute(0, 2, 3, 1).reshape(-1, len(class_indexes))
    teacher_flat = teacher_selected.permute(0, 2, 3, 1).reshape(-1, len(class_indexes))
    expected = F.kl_div(
        F.log_softmax(student_flat / temperature, dim=1),
        F.softmax(teacher_flat / temperature, dim=1),
        reduction="batchmean",
    ) * (temperature ** 2)

    assert torch.allclose(loss, expected, atol=1e-6, rtol=1e-6)


def test_kd_loss_on_class_indexes_returns_zero_for_empty_indexes():
    module = _load_runtime_continual_losses_module()

    student_logits = torch.randn(1, 4, 2, 2)
    teacher_logits = torch.randn(1, 4, 2, 2)

    loss = module.kd_loss_on_class_indexes(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        class_indexes=[],
        temperature=1.0,
    )

    assert float(loss.item()) == 0.0


def test_continual_losses_module_exposes_kd_api():
    module = _load_runtime_continual_losses_module()

    assert hasattr(module, "kd_loss_on_class_indexes")
    assert callable(module.kd_loss_on_class_indexes)


def test_cat_seg_training_skips_old_kd_without_teacher_or_old_indexes(monkeypatch):
    module = _load_runtime_cat_seg_model_module(monkeypatch)
    model = _build_runtime_training_model(module)
    model.lambda_old_kd = 0.7
    model.old_class_indexes = []

    losses = model(_build_training_batch())

    assert "loss_sem_seg" in losses
    assert torch.isfinite(losses["loss_sem_seg"])
    assert "loss_old_kd" not in losses


def test_cat_seg_training_emits_old_kd_when_teacher_state_is_compatible(monkeypatch, tmp_path):
    module = _load_runtime_cat_seg_model_module(monkeypatch)
    model = _build_runtime_training_model(module)
    model.lambda_old_kd = 0.5
    model.old_class_indexes = [0, 1]
    checkpoint_path = tmp_path / "teacher_ok.pth"
    teacher_state = {
        f"sem_seg_head.{key}": value.clone()
        for key, value in model.sem_seg_head.state_dict().items()
    }
    torch.save({"model": teacher_state}, checkpoint_path)
    model.old_teacher_weights = str(checkpoint_path)

    losses = model(_build_training_batch())

    assert "loss_sem_seg" in losses
    assert "loss_old_kd" in losses
    assert torch.isfinite(losses["loss_old_kd"])


def test_cat_seg_training_keeps_visible_mask_and_optional_ciba_ctr_paths(monkeypatch):
    module = _load_runtime_cat_seg_model_module(monkeypatch)
    model = _build_runtime_training_model(module)
    model.visible_class_indexes = torch.tensor([0, 2], dtype=torch.long)
    model.enable_ciba = True
    model.mine_critic = object()
    model.enable_ctr = True
    model.background_ids = [0]

    calls = {"mask": 0}
    original_mask = module.mask_logits_and_targets_to_visible_classes

    def _mask_with_tracking(logits, targets, visible_class_indexes):
        calls["mask"] += 1
        return original_mask(
            logits=logits,
            targets=targets,
            visible_class_indexes=visible_class_indexes,
        )

    monkeypatch.setattr(module, "mask_logits_and_targets_to_visible_classes", _mask_with_tracking)
    monkeypatch.setattr(model, "_compute_ciba_loss", lambda res3, targets: torch.tensor(0.2))
    monkeypatch.setattr(model, "_compute_ctr_loss", lambda res3: torch.tensor(0.3))

    losses = model(_build_training_batch())

    assert calls["mask"] == 1
    assert "loss_sem_seg" in losses
    assert "loss_ciba" in losses
    assert "loss_ctr" in losses
    assert torch.isfinite(losses["loss_ciba"])
    assert torch.isfinite(losses["loss_ctr"])


def test_cat_seg_training_tolerates_invalid_old_index_file_parse(monkeypatch, tmp_path):
    module = _load_runtime_cat_seg_model_module(monkeypatch)
    model = _build_runtime_training_model(module)

    bad_indexes = tmp_path / "old_indexes_bad.json"
    bad_indexes.write_text("{not valid json", encoding="utf-8")
    loaded_old_indexes = module.load_visible_class_indexes(str(bad_indexes), num_classes=3)
    model.old_class_indexes = [] if loaded_old_indexes is None else loaded_old_indexes.tolist()
    model.lambda_old_kd = 0.7

    checkpoint_path = tmp_path / "teacher_ok.pth"
    teacher_state = {
        f"sem_seg_head.{key}": value.clone()
        for key, value in model.sem_seg_head.state_dict().items()
    }
    torch.save({"model": teacher_state}, checkpoint_path)
    model.old_teacher_weights = str(checkpoint_path)

    losses = model(_build_training_batch())

    assert loaded_old_indexes is None
    assert "loss_sem_seg" in losses
    assert "loss_old_kd" not in losses


def test_get_old_teacher_returns_none_for_empty_teacher_path(monkeypatch):
    module = _load_runtime_cat_seg_model_module(monkeypatch)
    cat_seg_cls = module.CATSeg
    model = cat_seg_cls.__new__(cat_seg_cls)
    nn.Module.__init__(model)

    model.pixel_mean = torch.zeros(1)
    model.sem_seg_head = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
    model._old_teacher = None
    model._old_teacher_load_attempted = False
    model.old_teacher_weights = ""

    assert model._get_old_teacher() is None


def test_get_old_teacher_fails_closed_on_incompatible_state(monkeypatch, tmp_path, caplog):
    module = _load_runtime_cat_seg_model_module(monkeypatch)
    cat_seg_cls = module.CATSeg
    model = cat_seg_cls.__new__(cat_seg_cls)
    nn.Module.__init__(model)

    model.pixel_mean = torch.zeros(1)
    model.sem_seg_head = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
    model._old_teacher = None
    model._old_teacher_load_attempted = False

    checkpoint_path = tmp_path / "bad_teacher.pth"
    bad_state = {"sem_seg_head.fake_weight": torch.randn(2, 2)}
    torch.save({"model": bad_state}, checkpoint_path)
    model.old_teacher_weights = str(checkpoint_path)

    with caplog.at_level("WARNING"):
        teacher = model._get_old_teacher()

    assert teacher is None
    assert "Skipping old teacher load" in caplog.text
    assert "strict state mismatch" in caplog.text
