"""Three synthetic-minibatch CPU checks of the real CIFAR-10 backbone."""

import argparse

import torch
from spikingjelly.clock_driven import functional

from .common import ROOT, build_model, load_config, output_directory, provenance, seed_all, write_json
from .gradient_monitor import GradientMonitor
from .metrics_io import MetricWriter
from .step_adapter import DiagnosticScaler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/gradient_flow/smoke_cpu")
    parser.add_argument("--batches", type=int, choices=(1, 2, 3), default=3)
    args = parser.parse_args()
    torch.set_num_threads(2)
    seed_all(42)
    config = load_config(ROOT / "configs/gradient_flow/cifar10_diagnostics.yml")
    model = build_model(config, "torch").train()
    monitor = GradientMonitor(model)
    destination = output_directory(args.output)
    writer = MetricWriter(destination)
    adapter = DiagnosticScaler(monitor, sink=writer.write)
    handle = model.register_forward_pre_hook(adapter.before_forward)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    rows = []
    try:
        for step in range(args.batches):
            images, targets = torch.randn(2, 3, 32, 32), torch.tensor([1, 3])
            if step == 0:
                torch.save({"images": images, "targets": targets, "synthetic": True}, destination / "batch.pt")
            optimizer.zero_grad()
            logits = model(images)
            loss = torch.nn.functional.cross_entropy(logits, targets)
            adapter(loss, optimizer)
            functional.reset_net(model)
        rows = writer.summarize()
        assert len(rows) == config["layer"] and all(row["valid_steps"] == args.batches for row in rows)
        assert all(row["alpha_l"] == 4 and row["grad_input_norm"] > 0 and
                   row["grad_output_norm"] > 0 and row["parameter_grad_norm"] > 0 and
                   0 <= row["firing_rate"] <= 1 for row in rows)
        if args.batches > 1:
            assert all(row["directional_consistency"] is not None for row in rows)
        torch.save({"state_dict": model.state_dict(), "args": config, "synthetic_smoke_only": True}, destination / "checkpoint.pt")
        write_json(destination / "config.json", dict(config=config, source=provenance(), backend="torch",
                   device="cpu", amp=False, synthetic=True, batches=args.batches,
                   warning="Synthetic correctness smoke; not CIFAR training accuracy or paper evidence"))
    finally:
        handle.remove()
        monitor.close()
        functional.reset_net(model)
        writer.close()
    print(destination)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
