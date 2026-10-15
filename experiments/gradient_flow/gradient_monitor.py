"""Bounded, detached observations at complete residual-block boundaries."""

import math

import torch
from spikingjelly.clock_driven.neuron import MultiStepLIFNode

from .alpha_scope import BlockAlphas


def scalar(value):
    value = float(value.detach().cpu()) if torch.is_tensor(value) else float(value)
    return value if math.isfinite(value) else None


class GradientMonitor:
    def __init__(self, model, beta=0.95, eps=1e-12, enabled=True):
        if not 0 <= beta < 1 or eps <= 0 or not math.isfinite(eps):
            raise ValueError("Require 0 <= beta < 1 and finite eps > 0")
        self.model = getattr(model, "module", model)
        self.alphas = BlockAlphas(self.model)
        self.beta, self.eps, self.enabled = beta, eps, enabled
        self.references = [{} for _ in self.model.block]
        self.handles, self.tensor_handles = [], []
        self.active = False
        if enabled:
            for block_id, block in enumerate(self.model.block):
                self.handles.append(block.register_forward_hook(self._block_hook(block_id)))
                for node in block.modules():
                    if isinstance(node, MultiStepLIFNode):
                        self.handles.append(node.register_forward_hook(self._spike_hook(block_id)))

    def begin(self, epoch, step, loss_scale=1.0):
        self.abort()
        if not self.enabled:
            return
        if not math.isfinite(loss_scale) or loss_scale <= 0:
            raise ValueError("Loss scale must be positive and finite")
        self.epoch, self.step, self.loss_scale = int(epoch), int(step), float(loss_scale)
        self.applied_alphas = self.alphas.get()
        self.batch = [dict(input=None, output=None, spikes=None, elements=0) for _ in self.model.block]
        self.active = True

    def _block_hook(self, block_id):
        def hook(module, inputs, output):
            if not self.active:
                return
            tensors = (inputs[0], output[0] if isinstance(output, tuple) else output)
            for name, tensor in zip(("input", "output"), tensors):
                if tensor.requires_grad:
                    def record(grad, name=name):
                        # Scalar reduction only; never retain the activation/gradient graph.
                        self.batch[block_id][name] = grad.detach().float().norm(2) / self.loss_scale
                        return None
                    self.tensor_handles.append(tensor.register_hook(record))
            return None
        return hook

    def _spike_hook(self, block_id):
        def hook(module, inputs, output):
            if self.active:
                count = output.detach().float().sum()
                item = self.batch[block_id]
                item["spikes"] = count if item["spikes"] is None else item["spikes"] + count
                item["elements"] += output.numel()
            return None
        return hook

    def finish(self, valid=True, reason=None):
        if not self.enabled:
            return []
        if not self.active:
            raise RuntimeError("begin() must precede forward/backward")
        records, updates = [], []
        try:
            for block_id, block in enumerate(self.model.block):
                values = self.batch[block_id]
                grads = {name: p.grad.detach().float() for name, p in block.named_parameters()
                         if p.grad is not None}
                norm = torch.stack([g.square().sum() for g in grads.values()]).sum().sqrt() if grads else None
                gi = scalar(values["input"]) if values["input"] is not None else None
                go = scalar(values["output"]) if values["output"] is not None else None
                pn = scalar(norm) if norm is not None else None
                good = valid and gi is not None and go is not None and pn is not None
                reference = self.references[block_id]
                consistency = None
                if good and reference and reference.keys() == grads.keys():
                    refs = {name: reference[name].to(g.device) for name, g in grads.items()}
                    rn = torch.stack([r.square().sum() for r in refs.values()]).sum().sqrt()
                    if pn > self.eps and scalar(rn) is not None and float(rn) > self.eps:
                        dot = torch.stack([(g * refs[name]).sum() for name, g in grads.items()]).sum()
                        consistency = scalar((dot / (norm * rn)).clamp(-1, 1))
                updates.append({name: (self.beta * reference[name].to(g.device) + (1 - self.beta) * g
                                      if name in reference else g.clone()).detach()
                                for name, g in grads.items()} if good else None)
                rate = scalar(values["spikes"] / values["elements"]) if values["elements"] else None
                records.append(dict(epoch=self.epoch, step=self.step, block_id=block_id,
                                    alpha_l=self.applied_alphas[block_id], grad_input_norm=gi,
                                    grad_output_norm=go, G_l=math.log((gi + self.eps) / (go + self.eps)) if good else None,
                                    parameter_grad_norm=pn, directional_consistency=consistency,
                                    firing_rate=rate, valid=bool(good), loss_scale=self.loss_scale,
                                    invalid_reason=reason if not valid else (None if good else "missing/nonfinite gradient")))
            # A nonfinite/missing block invalidates this observation's EMA updates globally.
            if all(record["valid"] for record in records):
                self.references = updates
            return records
        finally:
            self.abort()

    def abort(self):
        self.active = False
        for handle in self.tensor_handles:
            handle.remove()
        self.tensor_handles.clear()
        self.batch = []

    def state_dict(self):
        return dict(beta=self.beta, eps=self.eps, references=[{name: value.detach().cpu().clone()
                    for name, value in block.items()} for block in self.references])

    def load_state_dict(self, state):
        if state["beta"] != self.beta or state["eps"] != self.eps or len(state["references"]) != len(self.references):
            raise ValueError("Diagnostic state configuration mismatch")
        self.references = [{name: value.detach().clone() for name, value in block.items()}
                           for block in state["references"]]

    def close(self):
        self.abort()
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
