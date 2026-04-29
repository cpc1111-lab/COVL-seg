from collections import OrderedDict
from typing import Callable, Optional

import torch
import torch.nn as nn


class EWCRegularizer:
    """Elastic Weight Consolidation regularizer.

    Computes a diagonal-Fisher penalty that discourages important parameters
    from drifting away from values learned on previous tasks.
    """

    def __init__(self, model: nn.Module, lambda_ewc: float = 10.0):
        self._model = model
        self.lambda_ewc = lambda_ewc
        self._device = next(model.parameters()).device
        self._trainable_names = [
            n for n, p in model.named_parameters() if p.requires_grad
        ]
        self._fisher: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._optimal_params: Optional[OrderedDict[str, torch.Tensor]] = None

    def compute_fisher(
        self,
        data_loader: torch.utils.data.DataLoader,
        loss_fn: Callable,
        n_samples: int = 200,
    ) -> None:
        self._model.eval()
        accum = OrderedDict((n, None) for n in self._trainable_names)
        count = 0

        for batch in data_loader:
            if count >= n_samples:
                break

            if isinstance(batch, dict):
                images = batch["image"].to(self._device, non_blocking=True)
                targets = batch["sem_seg"].to(self._device, non_blocking=True)
            elif isinstance(batch, (list, tuple)):
                images, targets = batch
                images = images.to(self._device, non_blocking=True)
                targets = targets.to(self._device, non_blocking=True)
            else:
                images, targets = batch
                images = images.to(self._device, non_blocking=True)
                targets = targets.to(self._device, non_blocking=True)
            self._model.zero_grad()
            with torch.enable_grad():
                loss = loss_fn(images, targets)
            loss.backward()

            for name in self._trainable_names:
                p = dict(self._model.named_parameters())[name]
                if p.grad is None:
                    continue
                grad_sq = (p.grad ** 2).detach()
                if accum[name] is None:
                    accum[name] = grad_sq.clone()
                else:
                    accum[name].add_(grad_sq)

            count += 1

        for name in self._trainable_names:
            if accum[name] is not None:
                accum[name].div_(count)

        self._fisher = accum

    def consolidate(self) -> None:
        self._optimal_params = OrderedDict(
            (n, p.data.clone()) for n, p in self._model.named_parameters() if p.requires_grad
        )

    def penalty(self, model: nn.Module) -> torch.Tensor:
        if self._optimal_params is None:
            return torch.tensor(0.0, device=next(model.parameters()).device)

        loss = torch.tensor(0.0, device=next(model.parameters()).device)
        for name, p in model.named_parameters():
            if not p.requires_grad or name not in self._fisher:
                continue
            fisher_diag = self._fisher[name].to(p.device)
            optimal = self._optimal_params[name].to(p.device)
            loss = loss + (fisher_diag * (p - optimal) ** 2).sum()
        return self.lambda_ewc * loss

    def state_dict(self):
        return {
            "lambda_ewc": self.lambda_ewc,
            "fisher": OrderedDict(
                (k, v.clone()) for k, v in self._fisher.items() if v is not None
            ),
            "optimal_params": OrderedDict(
                (k, v.clone()) for k, v in self._optimal_params.items()
            )
            if self._optimal_params is not None
            else None,
            "fisher_names": list(
                k for k, v in self._fisher.items() if v is not None
            ),
        }

    def load_state_dict(self, state):
        self.lambda_ewc = state["lambda_ewc"]
        self._fisher = OrderedDict((k, v) for k, v in state["fisher"].items())
        if state["optimal_params"] is not None:
            self._optimal_params = OrderedDict(
                (k, v) for k, v in state["optimal_params"].items()
            )
        else:
            self._optimal_params = None