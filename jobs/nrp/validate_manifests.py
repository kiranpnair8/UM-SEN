"""Offline consistency checks, optionally Kubernetes SERVER DRY RUN (no job creation)."""

import argparse
import json
from pathlib import Path
import re
import subprocess

import yaml

from .generate_manifests import NAMESPACE, PVC, RUNS, SOURCE_COMMIT, render


def validate(directory, production=False):
    directory = Path(directory)
    sample = yaml.safe_load((directory/"cifar100-env-check.yaml").read_text())
    pod = sample["spec"]["template"]["spec"]
    container = pod["containers"][0]
    values = {v["name"]: v["value"] for v in container["env"] if "value" in v}
    image, study_id = container["image"], Path(values["NRP_ROOT"]).name
    secret = pod.get("imagePullSecrets", [{}])[0].get("name")
    expected = render(image, study_id, secret)
    if production and not re.fullmatch(r".+@sha256:[a-f0-9]{64}", image):
        raise ValueError("Production requires a BUILT image@sha256:DIGEST, not the checked-in build tag")
    actual = {}
    for filename, specification in expected.items():
        document = yaml.safe_load((directory/filename).read_text())
        if document != specification:
            raise ValueError("Non-deterministic/mismatched manifest: "+filename)
        actual[filename] = document
    bundle = list(yaml.safe_load_all((directory/"cifar100-final-jobs.yaml").read_text()))
    finals = [d for d in actual.values() if d["metadata"]["labels"]["nrp-stage"] == "train"]
    if bundle != finals or len(finals) != 6 or len({d["metadata"]["name"] for d in actual.values()}) != 10:
        raise ValueError("Need exactly 6 unique final jobs and 4 preparation/inspection jobs")
    return actual


def server_dry_run(directory, documents):
    prefix = ["kubectl", "--context=nautilus", "--namespace="+NAMESPACE, "--request-timeout=30s"]
    pvc = json.loads(subprocess.check_output(prefix+["get", "pvc", PVC, "-o", "json"], text=True))
    if (pvc["status"]["phase"] != "Bound" or "ReadWriteMany" not in pvc["spec"]["accessModes"] or
            pvc["spec"].get("storageClassName") != "rook-cephfs"):
        raise ValueError("Review changed PVC access mode/binding; no PVC modifications are performed")
    for name in [*documents, "cifar100-final-jobs.yaml"]:
        subprocess.run(prefix+["apply", "--dry-run=server", "-f", str(Path(directory)/name)], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path(__file__).parent)
    parser.add_argument("--production", action="store_true")
    parser.add_argument("--server", action="store_true")
    args = parser.parse_args()
    documents = validate(args.directory, args.production)
    if args.server:
        server_dry_run(args.directory, documents)
    print("PASS: exact commit/namespace/RWX PVC/no node pinning/6 runs/resource limits/output IDs/deterministic bundle.")
    print("No pods/jobs launched. Image build and GPU execution require the separate environment/smoke stages.")


if __name__ == "__main__":
    main()
