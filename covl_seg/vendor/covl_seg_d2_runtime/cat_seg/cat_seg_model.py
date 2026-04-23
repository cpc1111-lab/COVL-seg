# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import json
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList
from detectron2.utils.memory import _ignore_torch_cuda_oom

from einops import rearrange

from .continual_losses import kd_loss_on_class_indexes


def _load_class_indexes(index_path: str) -> List[int]:
    if not index_path:
        return []
    path = Path(index_path)
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [int(idx) for idx in data]
    return []

@META_ARCH_REGISTRY.register()
class CATSeg(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        size_divisibility: int,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        clip_pixel_mean: Tuple[float],
        clip_pixel_std: Tuple[float],
        train_class_json: str,
        test_class_json: str,
        train_class_indexes: str,
        train_old_class_indexes: str,
        old_teacher_weights: str,
        distill_temp: float,
        lambda_old_kd: float,
        sliding_window: bool,
        clip_finetune: str,
        backbone_multiplier: float,
        clip_pretrained: str,
    ):
        """
        Args:
            sem_seg_head: a module that predicts semantic segmentation from backbone features
        """
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        if size_divisibility < 0:
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        self.register_buffer("clip_pixel_mean", torch.Tensor(clip_pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("clip_pixel_std", torch.Tensor(clip_pixel_std).view(-1, 1, 1), False)
        
        self.train_class_json = train_class_json
        self.test_class_json = test_class_json
        self.train_class_indexes = _load_class_indexes(train_class_indexes)
        self.old_class_indexes = _load_class_indexes(train_old_class_indexes)
        self.old_teacher_weights = old_teacher_weights
        self.distill_temp = float(distill_temp)
        self.lambda_old_kd = float(lambda_old_kd)
        self._old_teacher: Optional[nn.Module] = None
        self._old_teacher_load_attempted = False

        self.clip_finetune = clip_finetune
        for name, params in self.sem_seg_head.predictor.clip_model.named_parameters():
            if "transformer" in name:
                if clip_finetune == "prompt":
                    params.requires_grad = True if "prompt" in name else False
                elif clip_finetune == "attention":
                    if "attn" in name:
                        # QV fine-tuning for attention blocks
                        params.requires_grad = True if "q_proj" in name or "v_proj" in name else False
                    elif "position" in name:
                        params.requires_grad = True
                    else:
                        params.requires_grad = False
                elif clip_finetune == "full":
                    params.requires_grad = True
                else:
                    params.requires_grad = False
            else:
                params.requires_grad = False

        self.sliding_window = sliding_window
        self.clip_resolution = (384, 384) if clip_pretrained == "ViT-B/16" else (336, 336)

        self.proj_dim = 768 if clip_pretrained == "ViT-B/16" else 1024
        self.upsample1 = nn.ConvTranspose2d(self.proj_dim, 256, kernel_size=2, stride=2)
        self.upsample2 = nn.ConvTranspose2d(self.proj_dim, 128, kernel_size=4, stride=4)

        self.layer_indexes = [3, 7] if clip_pretrained == "ViT-B/16" else [7, 15] 
        self.layers = []
        for l in self.layer_indexes:
            self.sem_seg_head.predictor.clip_model.visual.transformer.resblocks[l].register_forward_hook(lambda m, _, o: self.layers.append(o))


    @classmethod
    def from_config(cls, cfg):
        backbone = None
        sem_seg_head = build_sem_seg_head(cfg, None)
        
        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "clip_pixel_mean": cfg.MODEL.CLIP_PIXEL_MEAN,
            "clip_pixel_std": cfg.MODEL.CLIP_PIXEL_STD,
            "train_class_json": cfg.MODEL.SEM_SEG_HEAD.TRAIN_CLASS_JSON,
            "test_class_json": cfg.MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON,
            "train_class_indexes": cfg.MODEL.SEM_SEG_HEAD.TRAIN_CLASS_INDEXES,
            "train_old_class_indexes": cfg.MODEL.SEM_SEG_HEAD.TRAIN_OLD_CLASS_INDEXES,
            "old_teacher_weights": cfg.MODEL.SEM_SEG_HEAD.OLD_TEACHER_WEIGHTS,
            "distill_temp": cfg.MODEL.SEM_SEG_HEAD.DISTILL_TEMP,
            "lambda_old_kd": cfg.MODEL.SEM_SEG_HEAD.LAMBDA_OLD_KD,
            "sliding_window": cfg.TEST.SLIDING_WINDOW,
            "clip_finetune": cfg.MODEL.SEM_SEG_HEAD.CLIP_FINETUNE,
            "backbone_multiplier": cfg.SOLVER.BACKBONE_MULTIPLIER,
            "clip_pretrained": cfg.MODEL.SEM_SEG_HEAD.CLIP_PRETRAINED,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def _get_old_teacher(self) -> Optional[nn.Module]:
        if self._old_teacher is not None:
            return self._old_teacher
        if self._old_teacher_load_attempted:
            return None
        self._old_teacher_load_attempted = True
        if not self.old_teacher_weights:
            return None

        teacher_path = Path(self.old_teacher_weights)
        if not teacher_path.exists():
            return None

        teacher = copy.deepcopy(self.sem_seg_head)
        checkpoint = torch.load(teacher_path, map_location="cpu")
        state_dict = checkpoint.get("model", checkpoint)
        teacher_state = {
            key.replace("sem_seg_head.", "", 1): value
            for key, value in state_dict.items()
            if key.startswith("sem_seg_head.")
        }
        load_state = teacher_state if teacher_state else state_dict
        teacher.load_state_dict(load_state, strict=False)
        teacher.to(self.device)
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad = False
        self._old_teacher = teacher
        return self._old_teacher
    
    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
        """
        
        images = [x["image"].to(self.device) for x in batched_inputs]
        if not self.training and self.sliding_window:
            return self.inference_sliding_window(batched_inputs)

        clip_images = [(x - self.clip_pixel_mean) / self.clip_pixel_std for x in images]
        clip_images = ImageList.from_tensors(clip_images, self.size_divisibility)

        self.layers = []

        clip_images_resized = F.interpolate(clip_images.tensor, size=self.clip_resolution, mode='bilinear', align_corners=False, )
        clip_features = self.sem_seg_head.predictor.clip_model.encode_image(clip_images_resized, dense=True)

        image_features = clip_features[:, 1:, :]

        # CLIP ViT features for guidance
        res3 = rearrange(image_features, "B (H W) C -> B C H W", H=24)
        res4 = rearrange(self.layers[0][1:, :, :], "(H W) B C -> B C H W", H=24)
        res5 = rearrange(self.layers[1][1:, :, :], "(H W) B C -> B C H W", H=24)
        res4 = self.upsample1(res4)
        res5 = self.upsample2(res5)
        features = {'res5': res5, 'res4': res4, 'res3': res3,}

        outputs = self.sem_seg_head(clip_features, features)
        if self.training:
            targets = torch.stack([x["sem_seg"].to(self.device) for x in batched_inputs], dim=0)
            outputs = F.interpolate(outputs, size=(targets.shape[-2], targets.shape[-1]), mode="bilinear", align_corners=False)
            student_outputs = outputs

            teacher_outputs = None
            old_teacher = self._get_old_teacher()
            if old_teacher is not None and self.old_class_indexes:
                with torch.no_grad():
                    teacher_outputs = old_teacher(clip_features, features)
                    teacher_outputs = F.interpolate(
                        teacher_outputs,
                        size=(targets.shape[-2], targets.shape[-1]),
                        mode="bilinear",
                        align_corners=False,
                    )
            
            num_classes = student_outputs.shape[1]
            mask = targets != self.sem_seg_head.ignore_value

            outputs = student_outputs.permute(0,2,3,1)
            _targets = torch.zeros(outputs.shape, device=self.device)
            _onehot = F.one_hot(targets[mask], num_classes=num_classes).float()
            _targets[mask] = _onehot
            
            loss = F.binary_cross_entropy_with_logits(outputs, _targets)
            losses = {"loss_sem_seg" : loss}
            if teacher_outputs is not None and self.lambda_old_kd > 0:
                losses["loss_old_kd"] = self.lambda_old_kd * kd_loss_on_class_indexes(
                    student_logits=student_outputs,
                    teacher_logits=teacher_outputs,
                    class_indexes=self.old_class_indexes,
                    temperature=self.distill_temp,
                )
            return losses

        else:
            outputs = outputs.sigmoid()
            image_size = clip_images.image_sizes[0]
            height = batched_inputs[0].get("height", image_size[0])
            width = batched_inputs[0].get("width", image_size[1])

            output = sem_seg_postprocess(outputs[0], image_size, height, width)
            processed_results = [{'sem_seg': output}]
            return processed_results


    @torch.no_grad()
    def inference_sliding_window(self, batched_inputs, kernel=384, overlap=0.333, out_res=[640, 640]):
        images = [x["image"].to(self.device, dtype=torch.float32) for x in batched_inputs]
        stride = int(kernel * (1 - overlap))
        unfold = nn.Unfold(kernel_size=kernel, stride=stride)
        fold = nn.Fold(out_res, kernel_size=kernel, stride=stride)

        image = F.interpolate(images[0].unsqueeze(0), size=out_res, mode='bilinear', align_corners=False).squeeze()
        image = rearrange(unfold(image), "(C H W) L-> L C H W", C=3, H=kernel)
        global_image = F.interpolate(images[0].unsqueeze(0), size=(kernel, kernel), mode='bilinear', align_corners=False)
        image = torch.cat((image, global_image), dim=0)

        images = (image - self.pixel_mean) / self.pixel_std
        clip_images = (image - self.clip_pixel_mean) / self.clip_pixel_std
        clip_images = F.interpolate(clip_images, size=self.clip_resolution, mode='bilinear', align_corners=False, )
        
        self.layers = []
        clip_features = self.sem_seg_head.predictor.clip_model.encode_image(clip_images, dense=True)
        res3 = rearrange(clip_features[:, 1:, :], "B (H W) C -> B C H W", H=24)
        res4 = self.upsample1(rearrange(self.layers[0][1:, :, :], "(H W) B C -> B C H W", H=24))
        res5 = self.upsample2(rearrange(self.layers[1][1:, :, :], "(H W) B C -> B C H W", H=24))

        features = {'res5': res5, 'res4': res4, 'res3': res3,}
        outputs = self.sem_seg_head(clip_features, features)

        outputs = F.interpolate(outputs, size=kernel, mode="bilinear", align_corners=False)
        outputs = outputs.sigmoid()
        
        global_output = outputs[-1:]
        global_output = F.interpolate(global_output, size=out_res, mode='bilinear', align_corners=False,)
        outputs = outputs[:-1]
        outputs = fold(outputs.flatten(1).T) / fold(unfold(torch.ones([1] + out_res, device=self.device)))
        outputs = (outputs + global_output) / 2.

        height = batched_inputs[0].get("height", out_res[0])
        width = batched_inputs[0].get("width", out_res[1])
        output = sem_seg_postprocess(outputs[0], out_res, height, width)
        return [{'sem_seg': output}]
