"""Deterministic fake-API concurrency/stale recovery; never touches Kubernetes."""

import copy
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
from urllib.error import HTTPError

import pytest

from jobs.nrp import lease_lock as locks
from jobs.nrp import generate_manifests as generator
from jobs.nrp import nrp_runner as runner


NOW = datetime(2026,10,29,tzinfo=timezone.utc)


class API:
    def __init__(self):
        self.body = dict(metadata={"resourceVersion":"1"},spec={"leaseDurationSeconds":120})
        self.finished = False
        self.mutex = threading.Lock()
        self.barrier = None

    def get_lease(self,name):
        with self.mutex:
            body = copy.deepcopy(self.body)
        if self.barrier:
            self.barrier.wait(timeout=5)
        return body

    def put_lease(self,name,body):
        with self.mutex:
            if body["metadata"]["resourceVersion"] != self.body["metadata"]["resourceVersion"]:
                raise HTTPError("api",409,"Conflict",{},None)
            self.body = copy.deepcopy(body)
            self.body["metadata"]["resourceVersion"] = str(int(self.body["metadata"]["resourceVersion"])+1)

    def pod_finished(self,name,uid):
        return self.finished


def guard(api,holder="pod-a/uid-a",scope="data"):
    return locks.KubernetesLease("cifar100-01ad13d-v1",scope,api=api,holder=holder,now=lambda:NOW)


def test_concurrent_cas_attempt_has_exactly_one_winner():
    api = API()
    api.barrier = threading.Barrier(2)
    outcomes = []
    def attempt(holder):
        try:
            guard(api,holder).acquire()
            outcomes.append("acquired")
        except locks.LeaseBusy:
            outcomes.append("busy")
    threads = [threading.Thread(target=attempt,args=(h,)) for h in ("pod-a/uid-a","pod-b/uid-b")]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
        assert not t.is_alive()
    assert sorted(outcomes) == ["acquired","busy"]


def test_release_and_retry_after_success_or_exception():
    api = API()
    with guard(api).held():
        with pytest.raises(locks.LeaseBusy):
            guard(api,"pod-b/uid-b").acquire()
    assert api.body["spec"]["holderIdentity"] == ""
    with pytest.raises(ValueError,match="operation failed"):
        with guard(api,"pod-b/uid-b").held():
            raise ValueError("operation failed")
    with guard(api,"pod-c/uid-c").held():
        assert api.body["spec"]["holderIdentity"] == "pod-c/uid-c"


def test_stale_lock_recovers_only_after_expiry_and_owner_termination():
    api = API()
    old = guard(api)
    old.acquire()
    api.body["spec"]["renewTime"] = (NOW-timedelta(seconds=121)).isoformat()
    with pytest.raises(locks.LeaseBusy):
        guard(api,"pod-b/uid-b").acquire()
    api.finished = True
    new = guard(api,"pod-b/uid-b")
    new.acquire()
    with pytest.raises(locks.LeaseLost):
        old.release()
    assert api.body["spec"]["holderIdentity"] == "pod-b/uid-b"
    new.release()
    guard(api,"pod-c/uid-c").acquire()


def test_terminated_but_unexpired_holder_cannot_be_replaced():
    api = API()
    guard(api).acquire()
    api.finished = True
    with pytest.raises(locks.LeaseBusy):
        guard(api,"pod-b/uid-b").acquire()


def test_unknown_owner_state_fails_closed_and_renewal_is_cas_protected():
    api = API()
    current = guard(api)
    current.acquire()
    current.renew()
    api.body["spec"]["renewTime"] = (NOW-timedelta(seconds=121)).isoformat()
    def forbidden(*a):
        raise HTTPError("api",403,"Forbidden",{},None)
    api.pod_finished = forbidden
    with pytest.raises(HTTPError):
        guard(api,"pod-b/uid-b").acquire()
    assert api.body["spec"]["holderIdentity"] == "pod-a/uid-a"


def test_delegated_native_lock_cannot_escape_independent_worktree(tmp_path):
    api = API()
    with guard(api).held() as current:
        with locks.delegate_native_lock(tmp_path/"repo/.approval.lock",tmp_path/"repo",current):
            pass
        with pytest.raises(ValueError,match="escaped"):
            with locks.delegate_native_lock(tmp_path/"other/.lock",tmp_path/"repo",current):
                pass
    assert not (tmp_path/"repo/.approval.lock").exists()


def test_coordination_rbac_precreated_leases_and_scoped_tokens():
    docs = generator.coordination()
    leases = [d for d in docs if d["kind"] == "Lease"]
    assert len(leases) == len(locks.SCOPES) == 11
    assert all(d["metadata"]["namespace"] == locks.NAMESPACE for d in docs)
    assert all(d["spec"] == {"leaseDurationSeconds":120} for d in leases)
    role = next(d for d in docs if d["kind"] == "Role")
    assert role["rules"][0]["verbs"] == ["get","update"]
    assert set(role["rules"][0]["resourceNames"]) == {d["metadata"]["name"] for d in leases}
    assert len({locks.lease_name("cifar100-01ad13d-v1",s) for s in locks.SCOPES}) == 11
    for doc in generator.render().values():
        pod = doc["spec"]["template"]["spec"]
        assert pod["serviceAccountName"] == locks.SERVICE_ACCOUNT
        assert pod["automountServiceAccountToken"] is False
        assert pod["volumes"][-1]["projected"]["sources"][0]["serviceAccountToken"]["expirationSeconds"] == 3600


def test_data_stage_has_one_outer_lock_not_nested_self_lock(tmp_path,monkeypatch):
    active, acquired = [], []
    @contextmanager
    def locked(path,**kw):
        assert path not in active, "Nested self-lock regression"
        active.append(path)
        acquired.append(path.name)
        yield SimpleNamespace(check=lambda:None,log=lambda *a,**k:None)
        active.remove(path)
    monkeypatch.setattr(runner,"locked",locked)
    monkeypatch.setattr(runner,"paths",lambda:tmp_path)
    monkeypatch.setattr(runner,"checkout",lambda *a:tmp_path/"repo")
    monkeypatch.setattr(runner,"source_module",lambda *a:SimpleNamespace())
    monkeypatch.setattr(runner,"snapshot",lambda *a,**k:{})
    monkeypatch.setattr(runner,"stage_parent",lambda *a:{})
    monkeypatch.setattr(runner,"data_hashes",lambda *a:{})
    monkeypatch.setattr(runner,"record_attempt",lambda *a:None)
    monkeypatch.setenv("NRP_RUN_ID","data")
    monkeypatch.setenv("POD_UID","uid-a")
    monkeypatch.setattr(sys,"argv",["runner","data"])
    calls=[]
    monkeypatch.setitem(sys.modules,"torchvision.datasets",SimpleNamespace(CIFAR100=lambda *a,**k:calls.append(k)))
    runner.main()
    runner.main()  # A completed/released attempt must remain retryable.
    assert acquired == ["data.lock","data.lock"]
    assert [c["train"] for c in calls] == [True,False,True,False]
    assert all(c["download"] for c in calls)


def test_heartbeat_lost_ownership_invokes_fail_closed_handler():
    api, failures = API(), []
    current = guard(api)
    current.acquire()
    api.body["spec"]["holderIdentity"] = "different-pod/other-uid"
    current.on_lost = failures.append
    current.stop = SimpleNamespace(wait=lambda seconds:False)
    current.heartbeat()
    assert len(failures) == 1 and isinstance(failures[0],locks.LeaseLost)
    assert api.body["spec"]["holderIdentity"] == "different-pod/other-uid"


@pytest.mark.parametrize("uid,phase,expected",[("uid-a","Running",False),("uid-a","Unknown",False),
    ("uid-a","Failed",True),("uid-a","Succeeded",True),("new-uid","Running",True)])
def test_api_checks_owner_uid_and_terminal_phase(uid,phase,expected):
    api = locks.KubernetesAPI.__new__(locks.KubernetesAPI)
    api.request = lambda path:dict(metadata={"uid":uid,"deletionTimestamp":"set"},status={"phase":phase})
    assert api.pod_finished("pod-a","uid-a") is expected


def test_missing_owner_pod_is_recoverable_but_permission_error_is_not():
    api = locks.KubernetesAPI.__new__(locks.KubernetesAPI)
    def fail(code):
        def request(path):
            raise HTTPError("api",code,"error",{},None)
        return request
    api.request = fail(404)
    assert api.pod_finished("pod-a","uid-a") is True
    api.request = fail(403)
    with pytest.raises(HTTPError):
        api.pod_finished("pod-a","uid-a")
