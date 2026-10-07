#!/usr/bin/env python3
"""UM-SEN wrapper for the official CIFAR-10 Spikformer trainer."""

import train as official_train
from sage_controller import BlockController, UMSENController, set_surrogate_alpha, summarize


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
    STATE["controller"] = None
    if parsed_args is not None and parsed_args.umsen:
        STATE["controller"] = UMSENController(model, parsed_args)
        model._sage_controller = STATE["controller"]
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
