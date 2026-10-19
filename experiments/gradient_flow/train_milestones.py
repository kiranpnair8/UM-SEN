"""Optional epoch-boundary milestone copies around the unmodified SAGE CIFAR trainer."""

import os
from pathlib import Path
import shutil
import sys

from .common import CIFAR, ROOT


def milestone_saver_class(base, epochs=()):
    epochs = frozenset(epochs)
    if not epochs:
        return base

    class MilestoneSaver(base):
        def save_checkpoint(self, epoch, metric=None):
            self._milestone_epoch_save = True
            try:
                return super().save_checkpoint(epoch, metric)
            finally:
                self._milestone_epoch_save = False

        def _save(self, path, epoch, metric=None):
            super()._save(path, epoch, metric)
            if getattr(self, "_milestone_epoch_save", False) and epoch in epochs:
                destination = Path(self.checkpoint_dir) / f"checkpoint_epoch{epoch:03d}.pth.tar"
                temporary = Path(str(destination) + ".tmp")
                # Copy enriched full state BEFORE timm publishes last.pth.tar.
                # No RNG draws, optimization, scheduler steps or tensor transforms.
                shutil.copyfile(path, temporary)
                os.replace(temporary, destination)
    return MilestoneSaver


def main():
    sys.path.insert(0, str(CIFAR))
    import train as official
    official.parser.add_argument("--milestone-checkpoints", nargs="*", type=int, default=[])
    official.parser.add_argument("--trajectory-id", default="")
    parse, saver_factory = official._parse_args, official.experiment_state.state_saver_class

    def checked_parse():
        args, text = parse()
        if args.milestone_checkpoints:
            if (any(e < 0 for e in args.milestone_checkpoints) or not args.trajectory_id or
                    args.fixed_alpha != 4 or args.sps_alpha != 4 or args.seed != 42 or
                    args.init_checkpoint or args.initial_checkpoint or args.pretrained or args.model_ema or
                    args.sage_v2 or getattr(args, "umsen", False) or not args.deterministic or
                    args.no_resume_opt or args.start_epoch is not None or args.use_multi_epochs_loader or
                    int(os.environ.get("WORLD_SIZE", "1")) != 1):
                raise ValueError("Milestone trajectory must be deterministic seed42, Fixed alpha4/SPS4, from scratch, no SAGE/EMA")
            if args.resume:
                from experiment_state import load_trusted_checkpoint
                checkpoint = load_trusted_checkpoint(args.resume)
                saved_args = checkpoint.get("args")
                saved_id = saved_args.get("trajectory_id") if isinstance(saved_args, dict) else getattr(saved_args, "trajectory_id", None)
                extra = checkpoint.get("sage_training_state") or {}
                if (saved_id != args.trajectory_id or not extra.get("rng") or
                        extra.get("scheduler") is None or "optimizer" not in checkpoint):
                    raise ValueError("Resume must retain this trajectory ID and complete epoch-boundary RNG/scheduler state")
            namespace = ROOT / "results/gradient_flow"
            destination = (Path(args.output) / args.experiment).resolve()
            if namespace.resolve() != namespace or not destination.is_relative_to(namespace) or destination == namespace:
                raise ValueError("Milestone training must use its own results/gradient_flow namespace")
            if args.resume and Path(args.resume).resolve().parent != destination:
                raise ValueError("Resume checkpoint must belong to this new trajectory directory")
            if not args.resume and destination.exists() and any(destination.iterdir()):
                raise ValueError("Refusing to overwrite a nonempty trajectory; resume or choose a new study")
        state["epochs"] = args.milestone_checkpoints
        return args, text

    state = {"epochs": []}
    official._parse_args = checked_parse
    official.experiment_state.state_saver_class = lambda base: milestone_saver_class(saver_factory(base), state["epochs"])
    try:
        official.main()
    finally:
        official._parse_args = parse
        official.experiment_state.state_saver_class = saver_factory


if __name__ == "__main__":
    main()
