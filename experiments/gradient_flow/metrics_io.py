"""Streaming scalar records and unweighted per-step epoch summaries."""

import json
import math

from .common import write_json

FIELDS = ("alpha_l", "grad_input_norm", "grad_output_norm", "G_l", "parameter_grad_norm",
          "directional_consistency", "firing_rate")


class MetricWriter:
    def __init__(self, directory):
        self.directory = directory
        self.stream = (directory / "steps.jsonl").open("x", encoding="utf-8")
        self.epochs = {}

    def write(self, records):
        for record in records:
            self.stream.write(json.dumps(record, allow_nan=False) + "\n")
            key = (record["epoch"], record["block_id"])
            item = self.epochs.setdefault(key, {"epoch": key[0], "block_id": key[1], "steps": 0,
                                                "valid_steps": 0, "sums": {}, "counts": {}})
            item["steps"] += 1
            item["valid_steps"] += int(record["valid"])
            for name in FIELDS:
                value = record[name]
                if record["valid"] and value is not None and math.isfinite(value):
                    item["sums"][name] = item["sums"].get(name, 0.) + value
                    item["counts"][name] = item["counts"].get(name, 0) + 1
        self.stream.flush()

    def summarize(self):
        rows = []
        for item in sorted(self.epochs.values(), key=lambda x: (x["epoch"], x["block_id"])):
            rows.append({**{name: item[name] for name in ("epoch", "block_id", "steps", "valid_steps")},
                         **{name: item["sums"].get(name, 0) / item["counts"][name]
                            if item["counts"].get(name) else None for name in FIELDS},
                         "counts": dict(item["counts"])})
        write_json(self.directory / "epochs.json", dict(schema_version=1, aggregation="mean of valid step observations",
                                                       epochs=rows))
        return rows

    def close(self):
        self.summarize()
        self.stream.close()
