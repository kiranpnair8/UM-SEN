"""Generate/validate ordinary standalone Jobs. This module never submits jobs."""

import argparse
from pathlib import Path
import re

import yaml

from .runner import PIN, ROOT, RUNS


NAMESPACE = "kc-ai-research-lab"
IMAGE_TAG = "ghcr.io/kiranpnair8/um-sen-nrp:torch251-cu121-py310-simple-v1"


def render(image):
    if not re.fullmatch(r".+@sha256:[a-f0-9]{64}",image):
        raise ValueError("Supply the built simple-v1 image@sha256:DIGEST")
    documents = {}
    for mode,condition,seed in [("data","",42),("smoke","",42),*(
            ("train",c,s) for s in (42,43,44) for c in ("fixed","deep"))]:
        run_id = f"{condition}_seed{seed}" if mode=="train" else mode
        name = f"cifar100-{condition}-seed{seed}" if mode=="train" else "cifar100-data-prepare" if mode=="data" else "cifar100-smoke"
        labels = {"app.kubernetes.io/part-of":"umsen-cifar100-simple","nrp-stage":mode,"nrp-run":run_id}
        if condition:
            labels.update({"nrp-condition":condition,"nrp-seed":str(seed)})
        env = [{"name":k,"value":v} for k,v in dict(NRP_ROOT=ROOT.as_posix(),NRP_SOURCE_COMMIT=PIN,
            NRP_RUN_ID=run_id,NRP_CONDITION=condition,NRP_SEED=str(seed),NRP_IMAGE=image).items()]
        env += [{"name":k,"valueFrom":{"fieldRef":{"fieldPath":v}}} for k,v in
            (("POD_NAME","metadata.name"),("POD_UID","metadata.uid"),("POD_NODE","spec.nodeName"))]
        mounts = [dict(name="storage",mountPath="/storage"),dict(name="shm",mountPath="/dev/shm")]
        volumes = [dict(name="storage",persistentVolumeClaim={"claimName":"umsen-storage-rwx"}),
            dict(name="shm",emptyDir={"medium":"Memory","sizeLimit":"4Gi"})]
        if mode!="data":
            # GPU jobs cannot modify the shared dataset through its normal path.
            mounts += [dict(name="storage",mountPath=(ROOT/"data").as_posix(),
                subPath="umsen-nrp/cifar100-simple/data",readOnly=True),
                dict(name="parent-input",mountPath="/parent-input",readOnly=True)]
            volumes.append(dict(name="parent-input",configMap={"name":"umsen-cifar100-parent-selection","optional":True}))
            env += [{"name":name,"valueFrom":{"configMapKeyRef":{"name":"umsen-cifar100-review",
                "key":key,"optional":True}}} for name,key in
                (("NRP_REVIEWED_BY","reviewed-by"),("NRP_ACK_RECIPE","acknowledge-inherited-normalization"))]
        requests = dict(cpu="1" if mode=="data" else "16",memory="2Gi" if mode=="data" else "16Gi",
            **{"ephemeral-storage":"8Gi"})
        if mode!="data":
            requests["nvidia.com/gpu"] = "1"
        pod = dict(restartPolicy="Never",serviceAccountName="default",automountServiceAccountToken=False,
            imagePullSecrets=[{"name":"ghcr-cred"}],nodeSelector={"kubernetes.io/arch":"amd64"},
            terminationGracePeriodSeconds=1800 if mode=="train" else 60,
            securityContext=dict(runAsUser=1000,runAsGroup=1000,fsGroup=1000,runAsNonRoot=True,
                seccompProfile={"type":"RuntimeDefault"}),volumes=volumes,
            containers=[dict(name="runner",image=image,imagePullPolicy="IfNotPresent",
                command=["/opt/simple/entrypoint.sh"],args=[mode],env=env,volumeMounts=mounts,
                resources=dict(requests=requests.copy(),limits=requests.copy()),
                securityContext=dict(allowPrivilegeEscalation=False,capabilities={"drop":["ALL"]}))])
        documents[name+".yaml"] = dict(apiVersion="batch/v1",kind="Job",
            metadata=dict(name=name,namespace=NAMESPACE,labels=labels,annotations={"umsen/source-commit":PIN}),
            spec=dict(backoffLimit=0,parallelism=1,completions=1,activeDeadlineSeconds=172800 if mode=="train" else 3600,
                template=dict(metadata=dict(labels=labels.copy()),spec=pod)))
    return documents


def validate(directory):
    directory = Path(directory)
    sample = yaml.safe_load((directory/"cifar100-smoke.yaml").read_text())
    image = sample["spec"]["template"]["spec"]["containers"][0]["image"]
    expected = render(image)
    for filename,doc in expected.items():
        if yaml.safe_load((directory/filename).read_text()) != doc:
            raise ValueError("Unexpected/unsafe manifest: "+filename)
    finals = [v for v in expected.values() if v["metadata"]["labels"]["nrp-stage"]=="train"]
    if list(yaml.safe_load_all((directory/"cifar100-final-jobs.yaml").read_text())) != finals:
        raise ValueError("Combined manifest differs from the six independent jobs")
    if {p.name for p in directory.glob("*.yaml")} != {*expected,"cifar100-final-jobs.yaml"}:
        raise ValueError("Unexpected YAML resources; this folder must contain only the nine ordinary Job manifests")
    assert len(finals)==6 and {(d["metadata"]["labels"]["nrp-condition"],int(d["metadata"]["labels"]["nrp-seed"])) for d in finals}==RUNS
    return expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image",help="Built image@sha256:DIGEST; required when generating")
    parser.add_argument("--output-dir",type=Path,default=Path("results/nrp_simple/manifests"))
    parser.add_argument("--check",action="store_true",help="Validate only; never rewrite manifests")
    args = parser.parse_args()
    if not args.check:
        if not args.image:
            parser.error("--image is required")
        documents = render(args.image)
        args.output_dir.mkdir(parents=True,exist_ok=True)
        for name,doc in documents.items():
            (args.output_dir/name).write_text(yaml.safe_dump(doc,sort_keys=False),encoding="utf-8")
        (args.output_dir/"cifar100-final-jobs.yaml").write_text(yaml.safe_dump_all(
            [d for d in documents.values() if d["metadata"]["labels"]["nrp-stage"]=="train"],sort_keys=False),encoding="utf-8")
    validate(args.output_dir)
    print("PASS: one CPU data job, one GPU smoke, six independent GPU runs; pinned SHA/PVC/namespace/ghcr-cred; no jobs submitted")


if __name__ == "__main__":
    main()
