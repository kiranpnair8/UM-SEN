"""Shared online epoch lifecycle, detached logging and fail-closed pre-step checks."""

import copy
import csv
import math
from pathlib import Path
import statistics
import time

import torch
from spikingjelly.clock_driven import functional

from .alpha_scope import BlockAlphas
from .common import write_json
from .gradient_monitor import GradientMonitor
from .online_policy import OnlinePolicy
from .online_probe import OnlineProbe
from .step_adapter import DiagnosticScaler
from analysis.gradient_flow.online_summary import mean, summarize_condition


def synchronize(model):
    if next(model.parameters()).device.type == "cuda":
        torch.cuda.synchronize(next(model.parameters()).device)


class TrainingMetrics:
    """Detached expected target mass in top-k (Mixup/smoothing), not hard-label accuracy."""
    def __init__(self, loss_fn):
        self.loss_fn, self.count, self.correct = loss_fn, 0, [0., 0.]

    def __call__(self, output, target):
        with torch.no_grad():
            indices = output.detach().topk(min(5, output.shape[1]), dim=1).indices
            mass = (target.detach() if target.ndim == 2 else
                    torch.nn.functional.one_hot(target, output.shape[1]).to(output.dtype))
            hits = mass.gather(1, indices).float()
            self.correct[0] += float(hits[:, :1].sum())
            self.correct[1] += float(hits.sum())
            self.count += len(target)
        return self.loss_fn(output, target)

    def finish(self, loss):
        if not self.count or not math.isfinite(loss):
            raise ValueError("Empty/nonfinite training epoch")
        return dict(loss=float(loss), top1=100*self.correct[0]/self.count,
                    top5=100*self.correct[1]/self.count,
                    accuracy_definition="expected smoothed/Mixup target mass in top-k; hard labels when no Mixup")


class SafeOnlineScaler(DiagnosticScaler):
    def __call__(self, loss, optimizer, clip_grad=None, clip_mode="norm", parameters=None, create_graph=False):
        if not bool(torch.isfinite(loss.detach()).all()):
            raise ValueError("Nonfinite training loss; no optimizer update")
        if self.native is None:
            loss.backward(create_graph=create_graph)
        else:
            self.native.scale(loss).backward(create_graph=create_graph)
            self.native.unscale_(optimizer)  # Exactly once, prior to measurement/clip/step.
        finite = all(bool(torch.isfinite(p.grad).all()) for group in optimizer.param_groups
                     for p in group["params"] if p.grad is not None)
        records = self.monitor.finish(valid=finite, reason=None if finite else "nonfinite gradient")
        if not finite or not records or not all(r["valid"] for r in records):
            raise ValueError("Invalid training gradients; no optimizer update")
        self.sink(records)  # Safety checks on unscaled metrics BEFORE clip/optimizer step.
        if clip_grad is not None:
            from timm.utils import dispatch_clip_grad
            dispatch_clip_grad(parameters, value=clip_grad, mode=clip_mode)
        if self.native is None:
            optimizer.step()
        else:
            self.native.step(optimizer)
            self.native.update()
        self.step += 1


class OnlineRuntime:
    def __init__(self, model, optimizer, scheduler, scaler, batch, method, settings, destination,
                 identity, matched=None):
        self.model, self.optimizer = model, optimizer
        self.settings, self.destination, self.identity = settings, Path(destination), identity
        self.scope = BlockAlphas(model)
        self.scope.set_all([4.]*len(model.block))
        if any(a != 4. for name, a in self.scope.snapshot().items() if name.startswith("patch_embed.")):
            raise ValueError("SPS must remain alpha=4")
        self.policy = OnlinePolicy(len(model.block), method, settings)
        self.monitor = GradientMonitor(model, settings["diagnostic_beta"], settings["diagnostic_eps"])
        self.adapter = SafeOnlineScaler(self.monitor, scaler, self.accept_gradients)
        self.handle = model.register_forward_pre_hook(self.adapter.before_forward)
        self.probe = OnlineProbe(model, batch, self.policy.config, optimizer, scheduler, scaler)
        self.rows, self.matched, self.gradient_rows = [], matched, []
        self.controller_seconds = 0.

    def accept_gradients(self, records):
        c = self.policy.config
        for row in records:
            if (row["G_l"] is None or abs(row["G_l"]) > c.max_abs_G or
                    any(row[k] is None or not math.isfinite(row[k]) or row[k] > c.max_gradient_norm
                        for k in ("grad_input_norm", "grad_output_norm", "parameter_grad_norm")) or
                    row["firing_rate"] is None or not 0 <= row["firing_rate"] <= 1):
                raise ValueError("Unsafe online gradient observation; stopping before optimizer step")
        self.gradient_rows.extend(records)  # Python scalars only, discarded at epoch end.

    def start_epoch(self, epoch):
        if epoch != len(self.rows):
            raise ValueError("Resume is epoch-boundary only; missing/duplicate epoch")
        self.monitor.abort()
        self.gradient_rows = []
        synchronize(self.model)
        self.tick = time.perf_counter()
        self.epoch = epoch
        self.adapter.epoch = epoch
        start = time.perf_counter()
        observation = self.probe(self.policy.alpha,
            estimate=self.policy.method != "fixed" and self.policy.due(epoch))
        matched = None if self.matched is None else self.matched[epoch]["controller"]
        self.control = self.policy.advance(epoch, observation, matched)
        self.scope.set_all(self.policy.alpha)
        # Keep the raw calibration (not model tensors) separately auditable.
        self.control["calibration"] = observation
        synchronize(self.model)
        self.probe_seconds = time.perf_counter()-start
        self.controller_seconds += self.probe_seconds

    def finish_epoch(self, training, validation):
        synchronize(self.model)
        if any(not math.isfinite(float(v)) for v in validation.values()):
            raise ValueError("Nonfinite validation metric")
        gradient = []
        for block in range(len(self.model.block)):
            rows = [r for r in self.gradient_rows if r["block_id"] == block]
            if not rows:
                raise ValueError("Missing online training block metrics")
            norms = [r["parameter_grad_norm"] for r in rows]
            gradient.append(dict(block_id=block, observations=len(rows),
                **{key: mean([r[key] for r in rows]) for key in
                   ("G_l", "grad_input_norm", "grad_output_norm", "parameter_grad_norm", "directional_consistency", "firing_rate")},
                parameter_grad_cv=statistics.pstdev(norms)/abs(mean(norms)) if abs(mean(norms)) > 1e-12 else None))
        row = dict(epoch=self.epoch, train=training, validation={k: float(v) for k, v in validation.items()},
            controller=copy.deepcopy(self.control), gradient=gradient,
            controller_seconds=self.probe_seconds, cumulative_controller_seconds=self.controller_seconds,
            epoch_seconds=time.perf_counter()-self.tick,
            learning_rates=[float(g["lr"]) for g in self.optimizer.param_groups])
        self.rows.append(row)
        self.gradient_rows = []
        return row

    def state_dict(self):
        return dict(version=1, identity=self.identity, rows=copy.deepcopy(self.rows),
            policy=self.policy.state_dict(), monitor=self.monitor.state_dict(),
            step=self.adapter.step, controller_seconds=self.controller_seconds)

    def load_state_dict(self, state):
        if state["version"] != 1 or state["identity"] != self.identity:
            raise ValueError("Online resume provenance/config mismatch")
        self.policy.load_state_dict(state["policy"])
        self.monitor.load_state_dict(state["monitor"])
        self.rows = copy.deepcopy(state["rows"])
        if [r["epoch"] for r in self.rows] != list(range(len(self.rows))) or len(self.policy.history) != len(self.rows):
            raise ValueError("Incomplete online checkpoint history")
        self.adapter.step = state["step"]
        self.controller_seconds = state["controller_seconds"]
        self.scope.set_all(self.policy.alpha)

    def persist(self, complete=False):
        d = self.destination
        write_json(d/"metrics.json", self.rows)
        write_json(d/"controller_metrics.json", [r["controller"] for r in self.rows])
        write_json(d/"timing.json", dict(epoch_seconds=sum(r["epoch_seconds"] for r in self.rows),
            controller_seconds=self.controller_seconds,
            definition="train+validation+controller epoch time; checkpoint I/O excluded"))
        summary = summarize_condition(self.rows, self.settings)
        write_json(d/"summary.json", summary)
        fields = ["epoch", "train_loss", "train_top1", "train_top5", "val_loss", "val_top1", "val_top5", "epoch_seconds", "controller_seconds"]
        temp = d/"metrics.csv.tmp"
        with temp.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for r in self.rows:
                writer.writerow(dict(epoch=r["epoch"], epoch_seconds=r["epoch_seconds"], controller_seconds=r["controller_seconds"],
                    **{"train_"+k: r["train"][k] for k in ("loss", "top1", "top5")},
                    **{"val_"+k: r["validation"][k] for k in ("loss", "top1", "top5")}))
        temp.replace(d/"metrics.csv")
        write_json(d/"status.json", dict(status="complete" if complete else "running", epochs=len(self.rows), identity=self.identity))

    def close(self):
        self.handle.remove()
        self.monitor.close()
        functional.reset_net(self.model)
