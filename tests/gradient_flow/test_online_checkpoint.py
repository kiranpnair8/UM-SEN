"""Synthetic locally generated full-state fixtures, never experimental results."""

import argparse
import copy
import inspect
import json
import pickle
import shutil
from types import SimpleNamespace

import pytest
import torch
from spikingjelly.clock_driven import functional

from experiments.gradient_flow import online_checkpoint as trust
from experiments.gradient_flow import amp_resume as lineage
from experiments.gradient_flow.common import ROOT, load_config
from experiments.gradient_flow.online_runtime import OnlineRuntime
from experiments.gradient_flow.online_train import load_checkpoint
from experiments.gradient_flow.validate_online_resume import restore_and_check
from test_online_controller import model_batch


@pytest.fixture
def full(tmp_path, monkeypatch):
    config = load_config(ROOT/'configs/gradient_flow/controller_online.yml')
    monkeypatch.setattr(trust, 'ROOT', tmp_path)
    destination = tmp_path/'results/gradient_flow/controller_online_seed42/fixed'
    destination.mkdir(parents=True)
    identity = dict(method='fixed', synthetic=True, source_commit=lineage.LEGACY_COMMIT,
                    training_signature='synthetic-unit-only', runtime={}, batch_fingerprint='synthetic')
    model, batch = model_batch()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0005, weight_decay=.06)
    from timm.scheduler.cosine_lr import CosineLRScheduler
    scheduler = CosineLRScheduler(optimizer, t_initial=300, warmup_t=20, warmup_lr_init=1e-5)
    native = torch.amp.GradScaler('cpu', init_scale=128.)
    owner = OnlineRuntime(model, optimizer, scheduler, native, batch, 'fixed', config['online'], destination, identity)
    from experiment_state import capture_rng
    loaders = [SimpleNamespace(generator=torch.Generator().manual_seed(s)) for s in (42, 43)]
    try:
        for epoch in range(7):
            owner.start_epoch(epoch)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(batch['images']), batch['targets'])
            owner.adapter(loss, optimizer)
            functional.reset_net(model)
            owner.finish_epoch(dict(loss=float(loss.detach()), top1=0., top5=0.), dict(loss=1., top1=0., top5=0.))
            scheduler.step(epoch+1)
        saved = dict(version=2, epoch=6, args=argparse.Namespace(model='SYNTHETIC-UNIT-ONLY'),
            state_dict=copy.deepcopy(model.state_dict()), optimizer=copy.deepcopy(optimizer.state_dict()),
            amp_scaler=native.state_dict(), gradient_flow_online=owner.state_dict(),
            sage_training_state=dict(scheduler=scheduler.state_dict(), rng=capture_rng(loaders),
                                     best_metric=0., best_epoch=0, controller=None))
        torch.save(saved, destination/'last.pth.tar')
        owner.persist()
        (destination/'config.json').write_text(json.dumps(config))
        (destination/'provenance.json').write_text(json.dumps(dict(identity=identity)))
    finally:
        owner.close()
    return destination, identity, config, saved, batch


def test_timm_1028_signature_and_trusted_full_restore_is_epoch7(full, tmp_path, monkeypatch):
    import timm
    from timm.models import resume_checkpoint
    from timm.utils.cuda import NativeScaler
    from timm.scheduler.cosine_lr import CosineLRScheduler
    assert 'weights_only' in inspect.signature(resume_checkpoint).parameters
    assert inspect.signature(resume_checkpoint).parameters['weights_only'].default is True
    destination, identity, config, expected, batch = full
    with pytest.raises(pickle.UnpicklingError):
        torch.load(destination/'last.pth.tar', weights_only=True)
    cap = trust.authorize(destination, identity, config, resume=True)
    saved = load_checkpoint(cap.path, cap)
    model, _ = model_batch()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0005, weight_decay=.06)
    scheduler = CosineLRScheduler(optimizer, t_initial=300, warmup_t=20, warmup_lr_init=1e-5)
    scaler = object.__new__(NativeScaler)
    scaler._scaler = torch.amp.GradScaler('cpu')
    scratch = tmp_path/'scratch'
    scratch.mkdir()
    runtime = OnlineRuntime(model, optimizer, scheduler, scaler._scaler, batch, 'fixed', config['online'], scratch, identity)
    calls = []
    def installed(model, path, optimizer=None, loss_scaler=None, log_info=True, weights_only=True):
        calls.append(weights_only)
        return resume_checkpoint(model, path, optimizer, loss_scaler, log_info, weights_only=weights_only)
    before = {p.name: p.read_bytes() for p in destination.iterdir() if p.is_file()}
    def forbidden(*args, **kwargs):
        raise AssertionError('Read-only validation must never forward or step')
    monkeypatch.setattr(model, 'forward', forbidden)
    monkeypatch.setattr(optimizer, 'step', forbidden)
    try:
        report = restore_and_check(model, optimizer, scheduler, scaler, runtime, saved, cap, installed)
        assert report['checkpoint_epoch'] == 6 and report['online_rows'] == report['next_epoch'] == 7
        assert all(report['checks'].values()) and report['optimizer_updates'] == report['forwards'] == 0
        assert calls == [False] and runtime.adapter.step == 7
        assert all((destination/name).read_bytes() == data for name, data in before.items())
    finally:
        runtime.close()


@pytest.mark.parametrize('problem', ['path', 'hash', 'identity', 'config', 'no_resume', 'unsealed'])
def test_unvalidated_inputs_cannot_call_full_deserializer(full, monkeypatch, problem):
    destination, identity, config, _, _ = full
    def forbidden(*a, **kw):
        raise AssertionError('Deserialization must not be reached')
    monkeypatch.setattr(torch, 'load', forbidden)
    if problem == 'hash':
        with (destination/'last.pth.tar').open('ab') as stream:
            stream.write(b'corrupted')
    if problem == 'identity':
        identity = {**identity, 'method': 'random_matched'}
    if problem == 'config':
        config = {**config, 'lr': .1}
    with pytest.raises(ValueError):
        if problem == 'unsealed':
            trust.OnlineCheckpoint(destination/'last.pth.tar', trust.digest(destination/'last.pth.tar'), identity, 6).load()
        else:
            trust.authorize(destination.parent.parent/'arbitrary' if problem == 'path' else destination,
                            identity, config, resume=problem != 'no_resume')


def test_default_model_loading_safe_and_no_unsafe_retry(full, tmp_path, monkeypatch):
    destination, _, _, _, _ = full
    original = torch.load
    calls = []
    def spy(*a, **kw):
        calls.append(kw.get('weights_only'))
        return original(*a, **kw)
    monkeypatch.setattr(torch, 'load', spy)
    with pytest.raises(pickle.UnpicklingError):
        load_checkpoint(destination/'last.pth.tar')
    assert calls == [True]
    plain = tmp_path/'model.pth'
    torch.save({'weight': torch.ones(2)}, plain)
    assert torch.equal(load_checkpoint(plain)['weight'], torch.ones(2))
    assert calls == [True, True]


def test_scoped_resume_rejects_other_paths_and_unsupported_api_without_fallback(full):
    from timm.models import resume_checkpoint
    original = resume_checkpoint
    destination, identity, config, _, _ = full
    cap = trust.authorize(destination, identity, config, resume=True)
    calls = []
    def supported(model, path, weights_only=True):
        calls.append(weights_only)
        return 7
    guarded = trust.scoped_timm_resume(supported, cap)
    with pytest.raises(ValueError, match='path/hash'):
        guarded(None, destination/'arbitrary.pth.tar')
    assert calls == []
    with pytest.raises(RuntimeError, match='no unsafe fallback'):
        trust.scoped_timm_resume(lambda model, path: 7, cap)
    from timm.models import resume_checkpoint as still_original
    assert still_original is original


def test_existing_amp_marker_and_protected_hashes_survive_second_patch(full, monkeypatch):
    destination, identity, config, saved, _ = full
    root = destination.parent
    (destination/'checkpoint_integrity.json').unlink()  # Exercise the genuine historical evidence path.
    backup = destination/'amp_resume_origin_epoch006.pth.tar'
    shutil.copyfile(destination/'last.pth.tar', backup)
    metrics_backup = destination/'amp_resume_origin_metrics.json'
    shutil.copyfile(destination/'metrics.json', metrics_backup)
    record = dict(legacy_commit=lineage.LEGACY_COMMIT, implementation_commit=lineage.AMP_FIX_COMMIT,
        original_fixed_identity=identity, prefix_epochs=list(range(7)),
        protected_checkpoint=str(backup), source_checkpoint_sha256=trust.digest(backup),
        prefix_metrics_sha256=trust.digest(destination/'metrics.json'),
        protected_prefix_artifacts={'metrics': dict(path=str(metrics_backup), sha256=trust.digest(metrics_backup))})
    marker = root/'amp_compatibility.json'
    marker.write_text(json.dumps(record))
    before = {p: p.read_bytes() for p in (marker, backup, metrics_backup)}
    patch = 'reviewed-resume-unit-test'
    monkeypatch.setattr(lineage, 'verify_resume_patch', lambda current: ['experiments/gradient_flow/online_checkpoint.py'])
    proposed = {**identity, 'source_commit': patch}
    current, transition = lineage.reconcile(root, proposed, config, dict(commit=patch), True,
        lambda unused: (_ for _ in ()).throw(AssertionError('No full load before authorization')))
    assert current == identity and transition['approved_patch_lineage'] == [lineage.LEGACY_COMMIT, lineage.AMP_FIX_COMMIT, patch]
    lineage.preserve_lineage(root, transition, patch)
    lineage.preserve_lineage(root, transition, patch)  # Immutable/idempotent concurrent condition enrollment.
    cap = trust.authorize(destination, identity, config, resume=True, allow_amp=True)
    assert cap.load()['epoch'] == 6
    assert all(p.read_bytes() == data for p, data in before.items())
    assert (root/('amp_resume_lineage_'+patch+'.json')).is_file()


def test_lineage_rejects_unrelated_files_and_arbitrary_later_parent(monkeypatch):
    def output(cmd, **kwargs):
        if '--format=%P' in cmd:
            return lineage.AMP_FIX_COMMIT+'\n'
        if '--format=%s' in cmd:
            return 'fix: allow validated full-state online checkpoint resume\n'
        return 'spikformer/cifar10/model.py\n'
    monkeypatch.setattr(lineage.subprocess, 'check_output', output)
    with pytest.raises(ValueError, match='arbitrary later'):
        lineage.verify_resume_patch('not-approved')
    monkeypatch.setattr(lineage.subprocess, 'check_output', lambda *a, **kw: 'later-parent\n')
    with pytest.raises(ValueError, match='arbitrary later'):
        lineage.verify_resume_patch('later-commit')


def test_parallel_launcher_has_condition_locks_and_random_dependency():
    text = (ROOT/'jobs/gradient_flow/run_controller_online_cifar10.sh').read_text()
    assert 'controller_online_${CONDITION}.lock' in text and '${CONDITIONS:-' in text
    assert 'RANDOM_MATCHED requires completed RESPONSE_AWARE' in text
    assert 'This launcher requires gpu002' not in text
