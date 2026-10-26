"""Deterministic manifest generator. Never invokes kubectl or launches jobs."""

import argparse
import json
from pathlib import Path
import re

import yaml


SOURCE_COMMIT = "01ad13da81eed92b64247c5dc3fcd44608f31a1f"
NAMESPACE = "kc-ai-research-lab"
PVC = "umsen-storage-rwx"
STUDY_ID = "cifar100-01ad13d-v1"
DEFAULT_IMAGE = "ghcr.io/kiranpnair8/um-sen-nrp:torch251-cu121-py310-rwx-v2"
RUNS = [(condition, seed) for seed in (42, 43, 44) for condition in ("fixed", "deep_only")]


def job(mode, image, study_id, *, condition=None, seed=42):
    run_id = f"{condition}_seed{seed}" if mode == "train" else mode.replace("-", "_")
    suffix = f"{'fixed' if condition == 'fixed' else 'deep'}-seed{seed}" if mode == "train" else "data-prepare" if mode == "data" else mode
    name = "cifar100-"+suffix
    root = "/storage/umsen-nrp/"+study_id
    labels = {"app.kubernetes.io/part-of": "umsen-cifar100", "nrp-stage": mode,
              "nrp-run": run_id, "nrp-source": SOURCE_COMMIT[:12]}
    if mode == "train":
        labels.update({"nrp-condition": condition.replace("_", "-"), "nrp-seed": str(seed)})
    env = [{"name": name, "value": value} for name, value in {
        "NRP_ROOT": root, "NRP_RUN_ID": run_id, "NRP_SEED": str(seed),
        "NRP_CONDITION": condition or "", "NRP_SOURCE_COMMIT": SOURCE_COMMIT,
        "NRP_IMAGE": image, "NRP_PVC": PVC}.items()]
    env += [{"name": name, "valueFrom": {"fieldRef": {"fieldPath": field}}} for name, field in
        (("POD_NAME", "metadata.name"), ("POD_UID", "metadata.uid"),
         ("POD_NAMESPACE", "metadata.namespace"), ("POD_NODE", "spec.nodeName"))]
    env += [{"name": name, "valueFrom": {"configMapKeyRef": {
        "name": "umsen-cifar100-review", "key": key, "optional": True}}} for name, key in
        (("NRP_REVIEWED_BY", "reviewed-by"), ("NRP_ACK_RECIPE", "acknowledge-inherited-normalization"))]
    env.append({"name": "NRP_GIT_TOKEN", "valueFrom": {"secretKeyRef": {
        "name": "umsen-github-read", "key": "token", "optional": True}}})
    resources = {"cpu": "16" if mode in ("train", "smoke") else "1",
                 "memory": "16Gi" if mode in ("train", "smoke") else "2Gi",
                 "ephemeral-storage": "8Gi"}
    if mode not in ("data", "inspect"):
        resources["nvidia.com/gpu"] = "1"
    env.append({"name": "NRP_RESOURCES", "value": json.dumps(resources, sort_keys=True)})
    pod = dict(restartPolicy="Never", automountServiceAccountToken=False,
        terminationGracePeriodSeconds=1800 if mode == "train" else 60,
        nodeSelector={"kubernetes.io/arch": "amd64"},
        securityContext={"runAsUser": 1000, "runAsGroup": 1000, "fsGroup": 1000,
                         "runAsNonRoot": True, "seccompProfile": {"type": "RuntimeDefault"}},
        containers=[dict(name="runner", image=image, imagePullPolicy="IfNotPresent",
            command=["/opt/nrp/entrypoint.sh"], args=[mode], env=env,
            resources=dict(requests=resources.copy(), limits=resources.copy()),
            securityContext=dict(allowPrivilegeEscalation=False, capabilities={"drop": ["ALL"]}),
            volumeMounts=[dict(name="storage", mountPath="/storage"),
                dict(name="shm", mountPath="/dev/shm"), dict(name="parent-input", mountPath="/parent-input", readOnly=True)])],
        volumes=[dict(name="storage", persistentVolumeClaim={"claimName": PVC}),
            dict(name="shm", emptyDir={"medium": "Memory", "sizeLimit": "4Gi"}),
            dict(name="parent-input", configMap={"name": "umsen-cifar100-parent-selection", "optional": True})])
    return dict(apiVersion="batch/v1", kind="Job", metadata=dict(name=name, namespace=NAMESPACE, labels=labels,
        annotations={"umsen/source-commit": SOURCE_COMMIT, "umsen/pvc-access-mode": "ReadWriteMany",
                     "umsen/image-needs-build": "true" if "@sha256:" not in image else "false"}),
        spec=dict(backoffLimit=0, activeDeadlineSeconds=172800 if mode == "train" else 3600,
                  template=dict(metadata=dict(labels=labels.copy()), spec=pod)))


def render(image=DEFAULT_IMAGE, study_id=STUDY_ID, image_pull_secret=None):
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{1,62}", study_id):
        raise ValueError("Invalid PVC experiment namespace")
    if not image or any(c.isspace() for c in image) or "$"+"{" in image:
        raise ValueError("Supply a concrete image reference")
    result = {f"cifar100-{mode if mode != 'data' else 'data-prepare'}.yaml":
        job(mode, image, study_id) for mode in ("env-check", "data", "smoke", "inspect")}
    for condition, seed in RUNS:
        key = f"cifar100-{'fixed' if condition == 'fixed' else 'deep'}-seed{seed}.yaml"
        result[key] = job("train", image, study_id, condition=condition, seed=seed)
    if image_pull_secret:
        if not re.fullmatch(r"[a-z0-9][a-z0-9.-]+", image_pull_secret):
            raise ValueError("Invalid image pull Secret name")
        for doc in result.values():
            doc["spec"]["template"]["spec"]["imagePullSecrets"] = [{"name": image_pull_secret}]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default=DEFAULT_IMAGE,
        help="Build/push provided image first, then rerender with registry/image@sha256:DIGEST")
    parser.add_argument("--study-id", default=STUDY_ID)
    parser.add_argument("--image-pull-secret", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    docs = render(args.image, args.study_id, args.image_pull_secret)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, document in docs.items():
        (args.output_dir/name).write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    finals = [v for v in docs.values() if v["metadata"]["labels"]["nrp-stage"] == "train"]
    (args.output_dir/"cifar100-final-jobs.yaml").write_text(
        yaml.safe_dump_all(finals, sort_keys=False), encoding="utf-8")
    print("Rendered 4 preparation/inspection Jobs and exactly 6 independent final Jobs. Nothing submitted.")


if __name__ == "__main__":
    main()
