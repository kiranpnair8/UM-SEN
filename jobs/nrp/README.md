# CIFAR-100 on Nautilus NRP

This is infrastructure, not a new scientific recipe. Scientific code is fetched
at exactly 01ad13da81eed92b64247c5dc3fcd44608f31a1f into independent PVC
worktrees. The support commit/image is recorded separately. Lawrence source,
running CIFAR-10 jobs, and results are never accessed or changed by these Jobs.

The inherited reviewed recipe is a 60-epoch CIFAR-10-recipe prefix adapted to
100 classes, NOT a claim of official CIFAR-100 reproduction. It intentionally
retains the audited CIFAR-10 normalization. Final conditions are Fixed and
deep_only (native condition name final_sage), seeds 42, 43, 44. All scientific
model/controller/optimizer/augmentation/AMP/resume logic remains in pinned code.

## Shared storage and independent scheduling

Use the existing Bound 100Gi umsen-storage-rwx PVC: ReadWriteMany on
rook-cephfs. No PVC is created or modified by this workflow. All jobs mount it
at /storage. No data/results are copied from, moved on, or deleted from any
other PVC. Prepare the dataset/evidence once on THIS volume before training.

There is no hostname selector, nodeName, node/pod affinity, or storage-induced
node pinning. The only node selector is amd64, matching the container image.
All six GPU jobs are independently schedulable across different GPU nodes and
may run concurrently, subject to GPU/CPU/memory availability and namespace
quotas. No concurrency or GPU reservation is implied by submitting them.
Final/smoke memory request/limit is 16Gi, not Lawrence's 64G host allocation.
Each final Job requests/limits exactly one GPU. No automatic retries or idle
GPU sleep containers. Deadline is 48h active time, plus up to 30min graceful
epoch-boundary pause. If the scheduler cannot fit the jobs, review scheduling
without silently changing the scientific recipe. Shared Git bootstrap and
dataset staging use CephFS advisory file locks; per-run worktrees, caches,
logs, statuses, results and checkpoints remain independent.

## 1. Build an immutable image (required)

The checked-in image tag is a build target, NOT an assertion it exists.
Docker was unavailable locally; the image has NOT been built or GPU-validated.
From the support checkout, on a machine with Docker and registry push access:

    docker build --platform linux/amd64 -t ghcr.io/kiranpnair8/um-sen-nrp:torch251-cu121-py310-rwx-v2 jobs/nrp
    docker push ghcr.io/kiranpnair8/um-sen-nrp:torch251-cu121-py310-rwx-v2
    docker buildx imagetools inspect ghcr.io/kiranpnair8/um-sen-nrp:torch251-cu121-py310-rwx-v2

Use the reported registry digest, not a locally invented digest:

    export IMAGE='ghcr.io/kiranpnair8/um-sen-nrp@sha256:REPLACE_WITH_REAL_DIGEST'
    export M=results/nrp/manifests
    python -m jobs.nrp.generate_manifests --image "$IMAGE" --output-dir "$M"
    python -m jobs.nrp.validate_manifests --directory "$M" --production --server

Rebuild the image for this RWX runner; do not reuse the previous image digest.
Use a new reviewed study ID via --study-id if a pre-existing readiness marker
on this PVC belongs to different support/image/software. Never overwrite or
relabel old experimental state. The old --node generator option is removed.

Generated runtime manifests stay ignored. For a private image, create an
appropriate imagePullSecret and pass --image-pull-secret NAME to the generator.
For a private GitHub repository, create Secret umsen-github-read with key token
using your credential-management procedure; never put tokens in source/YAML/logs.
Git uses optional askpass, never embeds the token in the stored remote URL.
Do not change context globally: commands specify context and namespace explicitly.

## 2. Environment check: GPU, versions, disk, source

    kubectl --context=nautilus -n kc-ai-research-lab apply -f "$M/cifar100-env-check.yaml"
    kubectl --context=nautilus -n kc-ai-research-lab logs -f job/cifar100-env-check
    kubectl --context=nautilus -n kc-ai-research-lab wait --for=condition=complete job/cifar100-env-check --timeout=3600s
    kubectl --context=nautilus -n kc-ai-research-lab delete job cifar100-env-check

The check prints nvidia-smi, GPU/driver, Python, torch, compiled CUDA,
torchvision, timm, SpikingJelly, numpy, image pip freeze, disk/free space, PVC,
paths, HEAD, node/pod, requested resources/cgroup limits. It exercises CuPy
kernel compilation on one GPU. Persistent logs and readiness survive Job deletion.
Readiness compares pinned source, image/software and storage identity, not
node/GPU identity. Every GPU job independently exercises CuPy and records its
actual GPU, memory, driver and node. Different compatible GPU nodes are allowed;
they are NOT treated as interchangeable for runtime/overhead comparisons.

## 3. Import real parent-selection evidence and download data ONCE

The pinned CIFAR-100 pipeline requires the original reviewed deep_only
selection. Copy these four small existing files from the COMPLETED historical
method-ablation study to a local transfer directory. Do not modify them or
inspect/copy the active CIFAR-10 run. No Lawrence checkpoints are required:

    results/gradient_flow/controller_method_ablation_seed42/final_selection.json
    results/gradient_flow/controller_method_ablation_seed42/study_manifest.json
    results/gradient_flow/controller_method_ablation_seed42/summary.json
    configs/gradient_flow/controller_followup.yml

Name the copies final_selection.json, study_manifest.json, summary.json, and
controller_followup.yml under parent-selection/. Then:

    kubectl --context=nautilus -n kc-ai-research-lab create configmap umsen-cifar100-parent-selection --from-file=parent-selection/
    kubectl --context=nautilus -n kc-ai-research-lab create configmap umsen-cifar100-review --from-literal=reviewed-by=YOUR_NAME --from-literal=acknowledge-inherited-normalization=true
    kubectl --context=nautilus -n kc-ai-research-lab apply -f "$M/cifar100-data-prepare.yaml"
    kubectl --context=nautilus -n kc-ai-research-lab logs -f job/cifar100-data-prepare
    kubectl --context=nautilus -n kc-ai-research-lab wait --for=condition=complete job/cifar100-data-prepare --timeout=3600s
    kubectl --context=nautilus -n kc-ai-research-lab delete job cifar100-data-prepare

This CPU-only Job downloads CIFAR-100 once into shared PVC storage, verifies
both splits and the original selection hashes/commit, and records data hashes.
Existing valid data is reused. Final/smoke require these files and use
download=False; they never independently download. The pinned native approval
is created separately in each pristine worktree using the same original evidence
and explicit normalization acknowledgement. Missing evidence is a hard stop,
not permission to fabricate a parent selection or bypass approval.

## 4. Real CIFAR-100 smoke (not a benchmark result)

    kubectl --context=nautilus -n kc-ai-research-lab apply -f "$M/cifar100-smoke.yaml"
    kubectl --context=nautilus -n kc-ai-research-lab logs -f job/cifar100-smoke
    kubectl --context=nautilus -n kc-ai-research-lab wait --for=condition=complete job/cifar100-smoke --timeout=3600s

Two epochs, eight real training minibatches and two validation minibatches per
epoch; full production CIFAR-100 architecture and AMP/CuPy/deep_only path.
Only this smoke's horizon/loaders are shortened, within the smoke process.
Readiness requires actual optimizer updates, four-block diagnostics and fixed
block-0 alpha=4. Scientific final horizon/loaders remain untouched.
Inspect persisted smoke metrics/status with a terminating CPU-only Job:

    kubectl --context=nautilus -n kc-ai-research-lab apply -f "$M/cifar100-inspect.yaml"
    kubectl --context=nautilus -n kc-ai-research-lab logs -f job/cifar100-inspect
    kubectl --context=nautilus -n kc-ai-research-lab wait --for=condition=complete job/cifar100-inspect --timeout=3600s
    kubectl --context=nautilus -n kc-ai-research-lab delete job cifar100-inspect cifar100-smoke

## 5. Launch six long jobs ONLY after reviewing the smoke

MANUAL command; no script automatically runs this step:

    kubectl --context=nautilus -n kc-ai-research-lab apply -f "$M/cifar100-final-jobs.yaml"

This bundle contains exactly the six independent fixed/deep seed42/43/44 Jobs.
Individual YAML files can instead be applied once each; do not submit both as
distinct experiments. Applying the bundle again does not create additional Jobs.

## Monitor, inspect assignments, and preserve failures

    kubectl --context=nautilus -n kc-ai-research-lab get jobs -l nrp-stage=train -w
    kubectl --context=nautilus -n kc-ai-research-lab get pods -l app.kubernetes.io/part-of=umsen-cifar100 -o wide
    kubectl --context=nautilus -n kc-ai-research-lab logs -f job/cifar100-fixed-seed42
    kubectl --context=nautilus -n kc-ai-research-lab logs -f job/cifar100-deep-seed42
    kubectl --context=nautilus -n kc-ai-research-lab logs -l nrp-condition=deep-only --all-containers=true --prefix=true --tail=50 --max-log-requests=6 -f
    kubectl --context=nautilus -n kc-ai-research-lab describe pod POD_NAME
    kubectl --context=nautilus -n kc-ai-research-lab exec POD_NAME -- nvidia-smi

Repeat logs by seed43/44 as needed. GPU model/UUID/node are also in each
persistent attempt record; pod resource limits show one assigned GPU.
For a confirmed FAILED Job only, deletion removes Job/pod, never PVC data:

    kubectl --context=nautilus -n kc-ai-research-lab delete job cifar100-fixed-seed42

Review failure and saved status before any reapply. Native failed-run guards
remain strict. Paused valid native checkpoints resume through the original
authorization; this wrapper does not restart failed runs or overwrite results.
Never delete umsen-storage-rwx. Never use bulk deletion against running work.
Reapply the CPU inspection Job after deleting its completed prior Job for an
updated persistent summary. Logs remain available on PVC after pod deletion.

## Persistent layout and provenance

Default root inside umsen-storage-rwx: /storage/umsen-nrp/cifar100-01ad13d-v1/

    code/UM-SEN/                        shared clean pinned object checkout
    data/cifar100/cifar-100-python/      train/test/meta, one shared dataset
    evidence/parent-selection/          original four evidence files
    checks/                            environment/data/smoke readiness
    logs/<run_id>/<pod_uid>.log         stdout + stderr per attempt
    runs/<run_id>/attempts/             versions/hardware/config/native identity
    runs/<run_id>/nrp-status.json       infrastructure status
    runs/<run_id>/result-index.json     completed native result locator
    runs/<run_id>/repo/                 independent clean pinned worktree

Run IDs: fixed_seed42, deep_only_seed42, fixed_seed43, deep_only_seed43,
fixed_seed44, deep_only_seed44. Scientific outputs/checkpoints/status are under
each worktree's results/gradient_flow/cifar100_deep_only_followup/seedN/fixed/
or final_sage/. Smoke uses its own worktree and nrp_cifar100_smoke output.
Each final records source commit, seed/dataset/condition, resolved config,
native signature/approval/batch hashes, runtime versions, image digest,
support-script hash, node, GPU model/memory/UUID/driver, requested resources,
and cgroup limits. Do not compare raw runtime/overhead with Lawrence without
accounting for these hardware/software/resource differences.

## Software contract and remaining validation

The image pins Python3.10, torch2.5.1+cu121, torchvision0.20.1+cu121,
timm1.0.28, SpikingJelly0.0.0.0.14, numpy2.2.6 and PyYAML6.0.3.
Official CUDA12.1.1-devel Ubuntu22.04 amd64 base is digest-pinned; no core CUDA
fallback is required/planned. Python patch is Ubuntu-selected, not guaranteed
Lawrence3.10.20. CuPy13.3.0, SciPy1.15.3, matplotlib3.10.7 and Pillow11.3.0
are pinned; their historical Lawrence versions were not supplied. The actual
host NVIDIA driver differs and is recorded, never silently treated as equal.
Any mismatch in required core versions aborts rather than silently substituting.

Local validation covers unit tests, YAML consistency, and server-side Kubernetes
dry-run only. It does NOT prove image pull/build success or GPU smoke success.
Review current cluster capacity/quota/storage use before launch. The existing100Gi
PVC is preserved; free space is checked, no automatic artifact pruning occurs.
