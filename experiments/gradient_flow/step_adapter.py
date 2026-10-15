"""Observe unscaled gradients before the unchanged optimizer update."""

import torch


def backward_step(loss, optimizer, monitor, scaler=None, clip_grad=None,
                  clip_mode="norm", parameters=None, create_graph=False, update=True):
    if scaler is None:
        loss.backward(create_graph=create_graph)
    else:
        scaler.scale(loss).backward(create_graph=create_graph)
        # Exactly one explicit unscale; GradScaler.step does not unscale again.
        scaler.unscale_(optimizer)
    valid = bool(torch.isfinite(loss.detach()).all()) and all(
        bool(torch.isfinite(p.grad).all()) for group in optimizer.param_groups
        for p in group["params"] if p.grad is not None)
    records = monitor.finish(valid=valid, reason=None if valid else "nonfinite loss/parameter gradient")
    if clip_grad is not None:
        from timm.utils import dispatch_clip_grad
        dispatch_clip_grad(parameters, value=clip_grad, mode=clip_mode)
    if update:
        if scaler is None:
            # Match ordinary FP32 training, including its nonfinite behavior.
            optimizer.step()
        else:
            scaler.step(optimizer)
            scaler.update()
    return records


class DiagnosticScaler:
    """timm scaler protocol, including a passive FP32 adapter."""
    state_dict_key = "amp_scaler"

    def __init__(self, monitor, native=None, sink=None):
        self.monitor, self.native, self.sink = monitor, native, sink
        self.epoch, self.step = 0, 0

    def before_forward(self, module, inputs):
        if module.training:
            scale = self.native.get_scale() if self.native is not None else 1.0
            self.monitor.begin(self.epoch, self.step, scale)

    def __call__(self, loss, optimizer, **kwargs):
        records = backward_step(loss, optimizer, self.monitor, self.native, **kwargs)
        if self.sink is not None:
            self.sink(records)
        self.step += 1

    def state_dict(self):
        return self.native.state_dict() if self.native is not None else {}

    def load_state_dict(self, state):
        if self.native is not None:
            self.native.load_state_dict(state)
