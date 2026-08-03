#!/usr/bin/env python3
"""UM-SEN wrapper for the official CIFAR-10 Spikformer trainer."""

import json
import math
import os
import types

import torch
from spikingjelly.clock_driven.neuron import MultiStepLIFNode

import train as official_train


STATE = {
    "args": None,
    "controller": None,
}


official_train.parser.add_argument("--umsen", action="store_true", default=False,
                                   help="enable UM-SEN adaptive surrogate alpha controller")
official_train.parser.add_argument("--umsen-entropy-temperature", type=float, default=0.25,
                                   help="temperature for UM-SEN attention entropy distribution")
official_train.parser.add_argument("--umsen-ema-beta", type=float, default=0.95,
                                   help="EMA beta for UM-SEN entropy dispersion")
official_train.parser.add_argument("--umsen-warmup-steps", type=int, default=100,
                                   help="minimum number of training steps with fixed alpha=4.0")
official_train.parser.add_argument("--umsen-alpha-min", type=float, default=3.0,
                                   help="lower clamp for UM-SEN surrogate alpha")
official_train.parser.add_argument("--umsen-alpha-max", type=float, default=5.0,
                                   help="upper clamp for UM-SEN surrogate alpha")


def set_surrogate_alpha(module, alpha):
    for node in module.modules():
        if isinstance(node, MultiStepLIFNode):
            surrogate = getattr(node, "surrogate_function", None)
            if surrogate is not None and hasattr(surrogate, "alpha"):
                current = getattr(surrogate, "alpha")
                if torch.is_tensor(current):
                    current.data.fill_(float(alpha))
                else:
                    setattr(surrogate, "alpha", float(alpha))


def summarize(values):
    if not values:
        return {"min": None, "mean": None, "max": None}
    return {
        "min": float(min(values)),
        "mean": float(sum(values) / len(values)),
        "max": float(max(values)),
    }


class BlockController:
    def __init__(self, block_id, args):
        self.block_id = block_id
        self.beta = args.umsen_ema_beta
        self.temperature = args.umsen_entropy_temperature
        self.alpha_min = args.umsen_alpha_min
        self.alpha_max = args.umsen_alpha_max
        self.raw_dispersion = 0.0
        self.ema_dispersion = 0.5
        self.normalized_z = 0.0
        self.centered_z = 0.0
        self.alpha = 4.0
        self.initialized = False
        self.running_count = 0
        self.running_mean = 0.0
        self.running_m2 = 0.0
        self.raw_history = []
        self.ema_history = []
        self.z_history = []
        self.centered_z_history = []
        self.alpha_history = []

    def observe(self, attn):
        with torch.no_grad():
            probs = torch.softmax(attn.detach() / self.temperature, dim=-1)
            entropy = -(probs * (probs + 1e-12).log()).sum(dim=-1)
            num_tokens = attn.shape[-1]
            if num_tokens > 1:
                entropy = entropy / math.log(num_tokens)
            per_head_entropy = entropy.mean(dim=(0, 1, 3))
            raw_value = float(per_head_entropy.std(unbiased=False).detach().cpu().item())

            if self.initialized:
                self.ema_dispersion = self.beta * self.ema_dispersion + (1.0 - self.beta) * raw_value
            else:
                self.ema_dispersion = raw_value
                self.initialized = True
            self.raw_dispersion = raw_value

            self.running_count += 1
            delta = self.ema_dispersion - self.running_mean
            self.running_mean += delta / self.running_count
            delta2 = self.ema_dispersion - self.running_mean
            self.running_m2 += delta * delta2
            running_variance = self.running_m2 / max(self.running_count - 1, 1)
            running_std = math.sqrt(max(running_variance, 0.0))
            self.normalized_z = (self.ema_dispersion - self.running_mean) / (running_std + 1e-6)

            self.raw_history.append(self.raw_dispersion)
            self.ema_history.append(self.ema_dispersion)
            self.z_history.append(self.normalized_z)

    def set_centered_alpha(self, centered_z):
        self.centered_z = float(centered_z)
        if abs(self.centered_z) < 0.25:
            alpha = 4.0
        else:
            alpha = 4.0 + 0.5 * math.tanh(self.centered_z)
        self.alpha = min(max(alpha, self.alpha_min), self.alpha_max)
        self.centered_z_history.append(self.centered_z)
        self.alpha_history.append(self.alpha)

    def epoch_summary(self):
        return {
            "raw_entropy_dispersion": summarize(self.raw_history),
            "ema_entropy_dispersion": summarize(self.ema_history),
            "normalized_z": summarize(self.z_history),
            "centered_z": summarize(self.centered_z_history),
            "alpha": summarize(self.alpha_history),
        }

    def current(self):
        return {
            "applied_alpha": float(self.alpha),
            "raw_entropy_dispersion": float(self.raw_dispersion),
            "ema_entropy_dispersion": float(self.ema_dispersion),
            "running_mean": float(self.running_mean),
            "normalized_z": float(self.normalized_z),
            "centered_z": float(self.centered_z),
        }

    def clear_epoch_history(self):
        self.raw_history.clear()
        self.ema_history.clear()
        self.z_history.clear()
        self.centered_z_history.clear()
        self.alpha_history.clear()


class UMSENController:
    def __init__(self, model, args):
        self.model = model
        self.controllers = [
            BlockController(block_id=i, args=args)
            for i, _ in enumerate(getattr(model, "block"))
        ]
        self.step = 0
        self.current_epoch = 0
        self.warmup_steps = args.umsen_warmup_steps
        self.history = []
        self.install_attention_hooks()
        self.wrap_forward()
        self.apply_step_alphas()

    def install_attention_hooks(self):
        for idx, block in enumerate(getattr(self.model, "block")):
            attn_module = block.attn
            controller = self.controllers[idx]

            def record_attention(self_attn, attn, block_controller=controller):
                if self_attn.training:
                    block_controller.observe(attn)

            attn_module._record_attention_entropy = types.MethodType(record_attention, attn_module)

    def wrap_forward(self):
        original_forward = self.model.forward
        controller = self

        def forward_with_umsen(*args, **kwargs):
            if controller.model.training:
                controller.apply_step_alphas()
            output = original_forward(*args, **kwargs)
            if controller.model.training:
                controller.update_centered_alphas()
                controller.finish_step()
            return output

        self.model.forward = forward_with_umsen

    def start_epoch(self, epoch):
        self.current_epoch = epoch
        for controller in self.controllers:
            controller.clear_epoch_history()

    def current_alphas(self):
        if self.current_epoch == 0 or self.step < self.warmup_steps:
            return [4.0 for _ in self.controllers]
        return [controller.alpha for controller in self.controllers]

    def apply_step_alphas(self):
        for block, alpha in zip(getattr(self.model, "block"), self.current_alphas()):
            set_surrogate_alpha(block, alpha)

    def update_centered_alphas(self):
        z_values = [controller.normalized_z for controller in self.controllers]
        mean_z = float(sum(z_values) / len(z_values)) if z_values else 0.0
        for controller in self.controllers:
            controller.set_centered_alpha(controller.normalized_z - mean_z)

    def finish_step(self):
        self.step += 1

    def epoch_record(self, epoch):
        record = {
            "epoch": int(epoch),
            "step": int(self.step),
            "blocks": {
                str(controller.block_id): controller.epoch_summary()
                for controller in self.controllers
            },
            "current": {
                str(controller.block_id): controller.current()
                for controller in self.controllers
            },
        }
        self.history.append(record)

    def save_history(self, output_dir):
        if output_dir is None:
            return
        with open(os.path.join(output_dir, "umsen_controller.json"), "w") as f:
            json.dump({"enabled": True, "epochs": self.history}, f, indent=2)


ORIGINAL_PARSE_ARGS = official_train._parse_args
ORIGINAL_CREATE_MODEL = official_train.create_model
ORIGINAL_TRAIN_ONE_EPOCH = official_train.train_one_epoch


def parse_args_with_umsen():
    args, args_text = ORIGINAL_PARSE_ARGS()
    STATE["args"] = args
    return args, args_text


def create_model_with_umsen(*args, **kwargs):
    model = ORIGINAL_CREATE_MODEL(*args, **kwargs)
    parsed_args = STATE["args"]
    if parsed_args is not None and parsed_args.umsen:
        STATE["controller"] = UMSENController(model, parsed_args)
    return model


def train_one_epoch_with_umsen(epoch, model, loader, optimizer, loss_fn, args, **kwargs):
    controller = STATE["controller"]
    if controller is not None:
        controller.start_epoch(epoch)
    metrics = ORIGINAL_TRAIN_ONE_EPOCH(epoch, model, loader, optimizer, loss_fn, args, **kwargs)
    if controller is not None and args.rank == 0:
        controller.epoch_record(epoch)
        controller.save_history(kwargs.get("output_dir"))
    return metrics


official_train._parse_args = parse_args_with_umsen
official_train.create_model = create_model_with_umsen
official_train.train_one_epoch = train_one_epoch_with_umsen


if __name__ == "__main__":
    official_train.main()
