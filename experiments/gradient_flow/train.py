"""Budget-limited passive diagnostics around the existing official CIFAR trainer."""

import os
import sys

import torch

from .alpha_scope import BlockAlphas
from .common import CIFAR, ROOT, LimitedLoader, output_directory, provenance, write_json
from .gradient_monitor import GradientMonitor
from .metrics_io import MetricWriter
from .step_adapter import DiagnosticScaler


def main():
    sys.path.insert(0, str(CIFAR))
    import train as official
    official.config_parser.set_defaults(config=str(ROOT / "configs/gradient_flow/cifar10_diagnostics.yml"))
    official.parser.add_argument("--diagnostic-beta", type=float, default=.95)
    official.parser.add_argument("--diagnostic-eps", type=float, default=1e-12)
    official.parser.add_argument("--diagnostic-max-batches", type=int, default=3,
                                 help="0 means all batches; default is a tiny diagnostic")
    official.parser.add_argument("--diagnostic-max-epochs", type=int, default=1,
                                 help="0 means the full inherited schedule")
    official.parser.add_argument("--block-alphas", type=float, nargs="+")
    official.parser.add_argument("--disable-diagnostics", action="store_true")
    original_parse, original_train, original_validate = official._parse_args, official.train_one_epoch, official.validate
    state = {"monitor": None, "adapter": None, "handle": None, "writer": None, "epochs": 0}

    def parse():
        args, text = original_parse()
        if (int(os.environ.get("WORLD_SIZE", "1")) != 1 or args.apex_amp or args.model_ema or
                args.torchscript or args.resume or args.sage_v2 or getattr(args, "umsen", False)):
            raise ValueError("Stage 1 supports fresh/weights-initialized single-GPU eager training, native AMP or FP32, no SAGE/EMA/resume")
        if args.sps_alpha != 4 or args.diagnostic_max_batches < 0 or args.diagnostic_max_epochs < 0:
            raise ValueError("SPS must remain alpha=4; diagnostic limits must be nonnegative")
        if args.diagnostic_max_batches == 1:
            raise ValueError("Existing trainer progress logging requires at least two batches")
        if args.block_alphas is not None and len(args.block_alphas) != args.layer:
            raise ValueError("Expected one --block-alphas value per block")
        destination = output_directory(os.path.join(args.output or "results/gradient_flow", args.experiment))
        args.output, args.experiment = str(destination.parent), destination.name
        state["destination"], state["args"] = destination, args
        write_json(destination / "diagnostic_config.json", dict(args=vars(args), source=provenance(),
                   purpose="passive gradient-flow diagnostic; no adaptive controller",
                   epoch_budget=args.diagnostic_max_epochs, batch_budget=args.diagnostic_max_batches,
                   validation_is_partial=args.diagnostic_max_batches > 0))
        return args, text

    def train_epoch(epoch, model, loader, optimizer, loss_fn, args, **kwargs):
        if args.diagnostic_max_epochs and state["epochs"] >= args.diagnostic_max_epochs:
            # The existing main catches this after the previous epoch's checkpoint.
            raise KeyboardInterrupt
        if state["monitor"] is None:
            scope = BlockAlphas(model)
            scope.set_all(args.block_alphas or [args.fixed_alpha] * len(scope.nodes))
            state["monitor"] = GradientMonitor(model, args.diagnostic_beta, args.diagnostic_eps,
                                                enabled=not args.disable_diagnostics)
            if not args.disable_diagnostics:
                state["writer"] = MetricWriter(state["destination"])
                native = kwargs.get("loss_scaler")
                if native is not None and not hasattr(native, "_scaler"):
                    raise ValueError("Only native AMP's GradScaler is supported")
                adapter = DiagnosticScaler(state["monitor"], getattr(native, "_scaler", None), state["writer"].write)
                state["adapter"] = adapter
                state["handle"] = model.register_forward_pre_hook(adapter.before_forward)
        if state["adapter"] is not None:
            state["adapter"].epoch = epoch
            kwargs["loss_scaler"] = state["adapter"]
        result = original_train(epoch, model, LimitedLoader(loader, args.diagnostic_max_batches),
                                optimizer, loss_fn, args, **kwargs)
        state["epochs"] += 1
        if state["writer"] is not None:
            state["writer"].summarize()
            torch.save(state["monitor"].state_dict(), state["destination"] / "diagnostic_state.pt")
        return result

    def validate(model, loader, loss_fn, args, **kwargs):
        return original_validate(model, LimitedLoader(loader, args.diagnostic_max_batches), loss_fn, args, **kwargs)

    official._parse_args, official.train_one_epoch, official.validate = parse, train_epoch, validate
    try:
        official.main()
    finally:
        if state["handle"] is not None:
            state["handle"].remove()
        if state["monitor"] is not None:
            state["monitor"].close()
            official.functional.reset_net(state["monitor"].model)
        if state["writer"] is not None:
            state["writer"].close()
        official._parse_args, official.train_one_epoch, official.validate = original_parse, original_train, original_validate


if __name__ == "__main__":
    main()
