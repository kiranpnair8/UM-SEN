"""NRP-only Kubernetes Lease coordination; no PVC advisory locks."""

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import ssl
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen


NAMESPACE = "kc-ai-research-lab"
SERVICE_ACCOUNT = "umsen-cifar100-coordination"
SCOPES = ("git", "env_check", "data", "smoke", "inspect", *(
    f"{condition}_seed{seed}" for seed in (42, 43, 44) for condition in ("fixed", "deep_only")))
DURATION = 120


def lease_name(study, scope):
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{1,62}", study) or scope not in SCOPES:
        raise ValueError("Unknown NRP lock study/scope")
    return "umsen-c100-"+hashlib.sha256(study.encode()).hexdigest()[:10]+"-"+scope.replace("_", "-")


def event(kind, **fields):
    print(json.dumps(dict(coordination=kind, **fields), sort_keys=True), flush=True)


class LeaseBusy(RuntimeError):
    pass


class LeaseLost(RuntimeError):
    pass


class KubernetesAPI:
    def __init__(self):
        if os.environ.get("POD_NAMESPACE") != NAMESPACE:
            raise ValueError("Unexpected coordination namespace")
        host = os.environ["KUBERNETES_SERVICE_HOST"]
        if ":" in host:
            host = "["+host+"]"
        self.base = "https://"+host+":"+os.environ.get("KUBERNETES_SERVICE_PORT_HTTPS", "443")
        self.credentials = Path("/var/run/secrets/nrp")
        self.ssl = ssl.create_default_context(cafile=str(self.credentials/"ca.crt"))

    def request(self, path, body=None):
        # Re-read projected tokens so Kubernetes token rotation is honored.
        request = Request(self.base+path, data=None if body is None else json.dumps(body).encode(),
            method="GET" if body is None else "PUT", headers={"Authorization":"Bearer "+
            (self.credentials/"token").read_text().strip(), "Content-Type":"application/json"})
        with urlopen(request, context=self.ssl, timeout=10) as response:
            return json.load(response)

    def get_lease(self, name):
        return self.request(f"/apis/coordination.k8s.io/v1/namespaces/{NAMESPACE}/leases/{name}")

    def put_lease(self, name, body):
        return self.request(f"/apis/coordination.k8s.io/v1/namespaces/{NAMESPACE}/leases/{name}", body)

    def pod_finished(self, name, uid):
        try:
            pod = self.request(f"/api/v1/namespaces/{NAMESPACE}/pods/{name}")
        except HTTPError as error:
            if error.code == 404:
                return True
            raise
        return pod["metadata"]["uid"] != uid or pod.get("status", {}).get("phase") in ("Failed", "Succeeded")


def fatal_loss(error):
    # Retain uncertain ownership until the entire pod terminates; never unlock
    # while interrupted Python/subprocess work might still write shared storage.
    event("ownership-lost-exit", error=repr(error))
    os._exit(74)


class KubernetesLease:
    def __init__(self, study, scope, *, api=None, holder=None, now=None, on_lost=fatal_loss):
        self.name = lease_name(study, scope)
        self.holder = holder or os.environ["POD_NAME"]+"/"+os.environ["POD_UID"]
        parts = self.holder.split("/")
        if len(parts) != 2 or not all(re.fullmatch(r"[a-zA-Z0-9.-]+", p) for p in parts):
            raise ValueError("Lease holder must identify pod name AND UID")
        self.api = api or KubernetesAPI()
        self.now = now or (lambda: datetime.now(timezone.utc))
        self.on_lost = on_lost
        self.active = False
        self.stop = threading.Event()

    def log(self, kind, **fields):
        event(kind, lease=self.name, holder=self.holder, **fields)

    def acquire(self):
        body = self.api.get_lease(self.name)
        spec = body.setdefault("spec", {})
        previous = spec.get("holderIdentity", "")
        if previous:
            # Expiration alone is insufficient: paused/partitioned writers may live.
            stamp = spec.get("renewTime") or spec.get("acquireTime")
            if not stamp:
                raise LeaseBusy("Occupied lease has no trustworthy timestamp: "+self.name)
            expired = (self.now()-datetime.fromisoformat(stamp.replace("Z", "+00:00"))).total_seconds() > spec.get("leaseDurationSeconds", DURATION)
            owner = previous.split("/")
            if (not expired or len(owner) != 2 or
                    not all(re.fullmatch(r"[a-zA-Z0-9.-]+", p) for p in owner) or
                    not self.api.pod_finished(*owner)):
                self.log("busy", previous_holder=previous, expired=expired)
                raise LeaseBusy("Lease held by "+previous+": "+self.name)
        stamp = self.now().isoformat(timespec="microseconds").replace("+00:00", "Z")
        spec.update(holderIdentity=self.holder, leaseDurationSeconds=DURATION, acquireTime=stamp,
            renewTime=stamp, leaseTransitions=spec.get("leaseTransitions", 0)+1)
        try:
            self.api.put_lease(self.name, body)  # metadata.resourceVersion is CAS.
        except HTTPError as error:
            if error.code == 409:
                raise LeaseBusy("Concurrent Lease acquisition: "+self.name) from error
            raise
        self.active = True
        self.log("acquired", previous_holder=previous, recovered_stale=bool(previous))

    def check(self):
        body = self.api.get_lease(self.name)
        if not self.active or body.get("spec", {}).get("holderIdentity") != self.holder:
            raise LeaseLost("Lease ownership changed: "+self.name)
        return body

    def renew(self):
        body = self.check()
        body["spec"]["renewTime"] = self.now().isoformat(timespec="microseconds").replace("+00:00", "Z")
        self.api.put_lease(self.name, body)

    def release(self):
        if not self.active:
            return
        try:
            body = self.check()
            body["spec"]["holderIdentity"] = ""
            self.api.put_lease(self.name, body)
            self.log("released")
        finally:
            self.active = False

    def heartbeat(self):
        while not self.stop.wait(20):
            try:
                self.renew()
            except Exception as error:
                self.log("renew-failed", error=repr(error))
                self.on_lost(error)
                return

    @contextmanager
    def held(self, *, blocking=False):
        deadline = time.monotonic()+600
        while True:
            try:
                self.acquire()
                break
            except LeaseBusy:
                if not blocking or time.monotonic() >= deadline:
                    raise
                time.sleep(2)
        self.stop.clear()
        worker = threading.Thread(target=self.heartbeat, daemon=True)
        worker.start()
        try:
            yield self
        finally:
            self.stop.set()
            worker.join()
            try:
                self.release()
            except Exception as error:
                self.log("release-failed", error=repr(error))


@contextmanager
def delegate_native_lock(path, repo, guard):
    """The outer per-run Lease already encloses native worktree-local locks."""
    path, repo = Path(path).absolute(), Path(repo).absolute()
    if path.resolve() != path or not path.is_relative_to(repo):
        raise ValueError("Native lock escaped its independently leased worktree")
    guard.check()
    guard.log("delegated-worktree-lock", path=str(path))
    yield
