# Standalone CIFAR-100 Fixed vs Deep-only SAGE on NRP

No Lease API, Role, RoleBinding, custom ServiceAccount, shared Git checkout,
filesystem mutex, or coordination process. This new path does not invoke the
older jobs/nrp/ runner. Existing older scripts/results are preserved as history;
do not launch that workflow alongside this one. No command below is executed
automatically by the generator.

All commands below are **PowerShell**, run from the repository root on branch
gradient-flow-stage1-validation. Use a build/support clone, not the active Lawrence
CIFAR-10 working directory. No Lawrence filesystem paths are used.

## Frozen scientific source and recipe

The image packages a clean checkout of exactly
01ad13da81eed92b64247c5dc3fcd44608f31a1f, not the infrastructure commit. Build and
runtime both verify HEAD. Every GPU run copies it into its own PVC directory;
no runtime Git clone/network credentials or shared Git serialization is needed.
Scientific source files remain unchanged. The NRP wrapper only suppresses native
worktree-local mutex contexts inside the independent run directory. No surrogate,
model, controller, dataset, AMP, optimizer or schedule equations are replaced.

Verified from that source/config: CIFAR-100 with Spikformer-4-384, 12 heads,
T=4, patch=4, MLP ratio=4, 100 classes. Fixed uses transformer and SPS alpha=4.
Deep-only fixes transformer block0 and SPS at4, adapts blocks1-3 using the
existing response-aware MethodPolicy, eta=.10, max step=.10, bounds[1,8].
Updates are once per epoch; reference windows are five epochs. Sixty training
epochs remain a PREFIX of the inherited 300-epoch cosine schedule, not a retuned
60-epoch schedule. Batch128, AdamW LR.0005, decay.06, warmup20, RandAugment,
Mixup.5, Random Erasing.25, label smoothing.1 and native AMP remain inherited.
Both conditions use the identical configuration except the controller method.

This is the already-reviewed CIFAR-10-recipe adaptation to CIFAR-100, including
the inherited CIFAR-10 normalization, NOT a claim of official CIFAR-100 reproduction.
Run artifacts are not present locally; no experimental outcomes are claimed here.

## 1. Build and push the new image

Lease-v3 cannot be reused as-is: its baked entrypoint starts Lease coordination
and lacks this runner/private packaged source. The new image contains no dataset,
experimental outputs or checkpoints. Docker must be running locally.

```powershell
$Tag = 'ghcr.io/kiranpnair8/um-sen-nrp:torch251-cu121-py310-simple-v1'
$env:DOCKER_BUILDKIT = '1'
docker login ghcr.io
docker build --platform linux/amd64 -t $Tag jobs/nrp_simple
docker push $Tag
docker buildx imagetools inspect $Tag
```

If GitHub source is private, supply a read-only token using BuildKit instead of
putting it in a build argument, Dockerfile, Git remote or committed file:

```powershell
# Set GITHUB_TOKEN in your local environment through your credential procedure.
docker build --platform linux/amd64 --secret id=github_token,env=GITHUB_TOKEN -t $Tag jobs/nrp_simple
docker push $Tag
docker buildx imagetools inspect $Tag
```

Core software matches the requested Lawrence pins: Python3.10.x,
torch2.5.1+cu121, torchvision0.20.1+cu121, timm1.0.28, SpikingJelly0.0.0.0.14,
numpy2.2.6. CUDA12.1.1-devel Ubuntu22.04 base is digest-pinned. Python patch,
CuPy13.3.0 and other auxiliary pins may differ from Lawrence; actual runtime and
hardware are recorded. Never compare raw elapsed time across different hardware
as controller overhead. The image has not been built/GPU-tested by local unit tests.

## 2. Generate and validate manifests

Use the actual registry digest printed by imagetools inspect:

```powershell
$Image = 'ghcr.io/kiranpnair8/um-sen-nrp@sha256:REPLACE_WITH_ACTUAL_DIGEST'
$M = 'results/nrp_simple/manifests'
python -m jobs.nrp_simple.manifests --image $Image --output-dir $M
python -m jobs.nrp_simple.manifests --check --output-dir $M
function k { & kubectl --context=nautilus -n kc-ai-research-lab @args }
```

This generates nine ignored YAML files: one data Job, one smoke Job, six final
Jobs, and a combined six-Job launch file. Every Job uses default ServiceAccount,
explicit imagePullSecrets ghcr-cred, existing umsen-storage-rwx at /storage,
backoffLimit0/restartNever/parallelism1/completions1. There are no RBAC resources
or API token volumes. Each GPU Job requests exactly one generic NVIDIA GPU.
No verified NRP A100 selector exists in this support path, so none is invented.
Only amd64 image architecture is selected. A later verified hardware selector can
be added explicitly to generated manifests, without touching science; regenerate
the original manifests to use the strict canonical local validator again.

Ensure the PVC is writable by UID/GID1000 (fsGroup1000 is requested) and ghcr-cred
already exists. This workflow does not repair permissions, create PVCs or Secrets,
or request RBAC privileges. Never reuse run output directories for different
settings or submit a second differently named Job against the same run_id.
This ordinary Job workflow intentionally provides no distributed exactly-once
guarantee during forced pod deletion/node partitions; operators must avoid
concurrent duplicate writers. Six different run_ids remain independent.

## 3. Prepare shared data once (CPU only)

```powershell
k apply -f "$M/cifar100-data-prepare.yaml"
k logs -f job/cifar100-data-prepare
k wait --for=condition=complete job/cifar100-data-prepare --timeout=3600s
```

Data: /storage/umsen-nrp/cifar100-simple/data/cifar100/cifar-100-python/.
Marker: /storage/umsen-nrp/cifar100-simple/data/CIFAR100_READY.
Preparation verifies torchvision integrity, 50000 train/10000 test examples,
100 classes, and records SHA256 of train/test/meta. Valid data plus a matching
marker returns successfully without another download. A mismatched existing
marker/data fails for review, never silently repairs/relabels it. No other Job
downloads data. GPU Jobs additionally mount this data directory read-only.

For a confirmed failed/terminated previous data Job only, replace its immutable
Job template before retry; no PVC/data deletion is needed:

```powershell
k delete job cifar100-data-prepare --ignore-not-found --wait=true
k apply -f "$M/cifar100-data-prepare.yaml"
```

## 4. Retain original scientific selection evidence

Data preparation does NOT need approval files. GPU runs retain the pinned
scientific pipeline's original selection verification, rather than inventing a
new deadband/selection record. Reuse the existing static ConfigMaps:
umsen-cifar100-parent-selection and umsen-cifar100-review. These are input data,
not coordination. No in-pod Kubernetes API access is used.

The parent-selection ConfigMap must contain the original four keys:
final_selection.json, study_manifest.json, summary.json, controller_followup.yml.
They correspond to the completed controller_method_ablation_seed42 study and its
original config. The pinned pipeline verifies their original hashes and parent
commit7cd92a8d7402957083c855823dca906d5ac681b6. Each GPU run stores its own copy.
Review ConfigMap keys remain reviewed-by and acknowledge-inherited-normalization
(literal true). If these existing inputs are unavailable, obtain the original
files; do not fabricate them or bypass scientific verification. No active
Lawrence run is accessed by this runner.

## 5. Run the GPU smoke and inspect its GPU

After data preparation completed, and only after removing a confirmed failed or
completed older smoke Job if its immutable template needs replacement:

```powershell
k apply -f "$M/cifar100-smoke.yaml"
k logs -f job/cifar100-smoke
k wait --for=condition=complete job/cifar100-smoke --timeout=3600s
k get pods -l app.kubernetes.io/part-of=umsen-cifar100-simple,nrp-stage=smoke -o wide
```

While the pod is running:

```powershell
$Pod = k get pods -l app.kubernetes.io/part-of=umsen-cifar100-simple,nrp-stage=smoke -o jsonpath='{.items[0].metadata.name}'
k exec $Pod -- nvidia-smi
```

The smoke itself prints nvidia-smi, torch/CUDA versions, CUDA availability,
device name/properties, and performs CuPy kernel compilation, real CIFAR-100
loading, full backbone construction, FP32 forward/backward, block1 alpha=5
intervention with unchanged logits/loss and untouched SPS, alpha restoration,
the inherited Deep-only finite-difference response probe and policy path, and a
persistent smoke.json write. No optimizer steps or accuracy claims. All checks
must pass for exit0/SMOKE PASSED. This is not an AMP training benchmark; final
training retains the inherited native AMP and overflow-handling implementation.

## 6. Launch the six jobs manually

Only after reviewing the successful smoke. Nothing launches this automatically:

```powershell
k apply -f "$M/cifar100-final-jobs.yaml"
```

Names: cifar100-fixed-seed42, cifar100-deep-seed42, cifar100-fixed-seed43,
cifar100-deep-seed43, cifar100-fixed-seed44, cifar100-deep-seed44. All may schedule
across different GPU nodes concurrently, subject to quotas/resources. Each has
16 CPUs/16Gi memory, 48h active deadline and native epoch-boundary pause on TERM
(up to30min grace). No shared run, Git or dataset-preparation mutex is used.

## 7. Monitor

```powershell
k get jobs -l app.kubernetes.io/part-of=umsen-cifar100-simple -w
k get pods -l app.kubernetes.io/part-of=umsen-cifar100-simple -o wide
k logs -f job/cifar100-fixed-seed42
k logs -f job/cifar100-deep-seed42
k describe job cifar100-deep-seed43
```

Native failed-run and checkpoint-integrity guards remain intact. This wrapper
does not reset a failed scientific run or erase partial checkouts. Valid paused
runs use the existing native resume authorization. Do not force-delete active
pods or use another name to start a duplicate output writer.

## 8. Collect completed summaries in PowerShell

The runner prints one machine-readable SIMPLE_RESULT_JSON line on completion.
Collect from retained completed Job logs without creating another helper pod:

```powershell
New-Item -ItemType Directory -Force results/nrp_simple/collected | Out-Null
$Names = 42,43,44 | ForEach-Object { "cifar100-fixed-seed$_"; "cifar100-deep-seed$_" }
$Summaries = @()
foreach ($Name in $Names) {
    $Lines = k logs "job/$Name"
    $Lines | Set-Content -Encoding utf8 "results/nrp_simple/collected/$Name.log"
    $Result = $Lines | Where-Object { $_.StartsWith('SIMPLE_RESULT_JSON=') } | Select-Object -Last 1
    if ($Result) { $Summaries += ($Result.Substring('SIMPLE_RESULT_JSON='.Length) | ConvertFrom-Json) }
    else { Write-Warning "$Name has no completed summary; inspect its status/log" }
}
$Summaries | ConvertTo-Json -Depth 40 | Set-Content -Encoding utf8 results/nrp_simple/collected/summaries.json
```

Permanent PVC root: /storage/umsen-nrp/cifar100-simple/. Runs are
runs/fixed_seed42, runs/deep_seed42, etc. Each contains status.json, attempts/
(actual GPU/node/runtime/image/source), logs/<pod_uid>.log, resolved_config.json,
smoke.json or result.json, and its private repo/. Native scientific outputs,
model_best.pth.tar, last.pth.tar, metrics, provenance and config remain at
repo/results/gradient_flow/cifar100_deep_only_followup/seedN/fixed or final_sage.
Deep-only is mapped to the native final_sage name without changing its method.
The final result includes native training signature/selection provenance.
Do not delete the PVC or historical NRP/Lawrence results to use this workflow.
