#!/usr/bin/env python3
"""Deterministic fault injection against the shipped checked-skill runtime.

This is an implementation audit with constructed contracts, not a second
agent benchmark or an estimate of a deployment failure probability.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from delegation_lab.certify import Certificate, ProofNode
from delegation_lab.evidence import EvidenceKey, EvidenceKind, EvidenceRecord, EvidenceStore, Provenance, TrustLevel
from delegation_lab.ir import Literal, Region, ToolProgram
from delegation_lab.lifecycle import CertificateRegistry, CertificateStatus
from delegation_lab.runtime import InMemoryBackend, RequestContext
from delegation_lab.skill import CertifiedSkill, SkillChecker, SkillRuntime, SkillStep, step_claim


def fixture():
    read = ToolProgram('read', {'id': '$id'}, (Literal('ready', ('$id',)),))
    write = ToolProgram('write', {'id': '$id'}, (Literal('eligible', ('$id',)),))
    posts = [(Literal('observed', ('$id',)),), (Literal('written', ('$id',)),)]
    region = Region((('intent', 'serve'),))
    records = []
    for name, program, post, writes in [('read', read, posts[0], False), ('write', write, posts[1], True)]:
        records.append(EvidenceRecord(EvidenceKey('review', name), EvidenceKind.FACT,
                                     step_claim(name, program, post, writes),
                                     Provenance('fixture://reviewed-contract', 'v1', 'constructed-input', 'review fixture'),
                                     trust=TrustLevel.ACCEPTED, scope_dependencies=frozenset({'scope'}),
                                     closure_dependencies=frozenset({'closure'})))
    store = EvidenceStore(records)
    steps = []
    for record, program, post, writes in zip(records, [read, write], posts, [False, True], strict=True):
        cert = Certificate.build(region, program, record.payload, ProofNode(record.payload, source_id=record.id), store)
        steps.append(SkillStep(cert, post, writes))
    skill = CertifiedSkill.build(region, ('id', 'principal'), (Literal('ready', ('$id',)),), steps)
    context = RequestContext('request', 'alice', 'consent', {'id': 'item', 'principal': 'alice', 'intent': 'serve'})
    return store, skill, context


class Backend(InMemoryBackend):
    def __init__(self, fault='none'):
        super().__init__(facts={Literal('ready', ('item',))}, policy_epochs={'scope': 'v1'},
                         tool_mutability={'read': False, 'write': True})
        self.fault = fault
        self.before_first_snapshot = None

    def snapshot(self):
        if self.before_first_snapshot is not None:
            callback = self.before_first_snapshot
            self.before_first_snapshot = None
            callback()
        return super().snapshot()

    def commit(self, name, arguments, **kw):
        if name == 'write' and self.fault == 'state_race':
            self.change(facts=set(self.facts) - {Literal('eligible', ('item',))})
        if name == 'write' and self.fault == 'policy_race':
            # Isolate the policy comparison: operational state stays unchanged.
            self.policy_epochs['scope'] = 'v2'
        committed, result, reason = super().commit(name, arguments, **kw)
        if committed and name == 'read':
            if self.fault != 'missing_post':
                self.facts.add(Literal('observed', ('item',)))
            if self.fault != 'missing_guard':
                self.facts.add(Literal('eligible', ('item',)))
        if committed and name == 'write':
            self.facts.add(Literal('written', ('item',)))
        return committed, result, reason

    def is_write(self, name):
        return False if name == 'write' and self.fault == 'effect_type' else super().is_write(name)


def run():
    cases = []
    for name in ['nominal', 'semantic_edit', 'version_edit', 'tool_tamper', 'argument_tamper',
                 'principal_mismatch', 'missing_guard', 'missing_post', 'state_race', 'policy_race', 'effect_type',
                 'closure_invalidation', 'selective_invalidation', 'untrusted_proposal', 'initial_snapshot_race']:
        store, skill, context = fixture()
        backend = Backend(name)
        registry = CertificateRegistry(SkillChecker())
        registry.register(skill, store)
        detail = ''
        if name == 'initial_snapshot_race':
            def update_during_first_snapshot():
                old = store.get('review:write')
                store.admit(replace(old, payload=Literal('revised_contract', ()),
                                    provenance=replace(old.provenance, source_version='v2')))
                backend.policy_epochs['scope'] = 'v2'
            backend.before_first_snapshot = update_during_first_snapshot
        if name in {'semantic_edit', 'version_edit'}:
            old = store.get('review:write')
            updated = replace(old, payload=Literal('revised_contract', ()) if name == 'semantic_edit' else old.payload,
                              provenance=replace(old.provenance, source_version='v2') if name == 'version_edit' else old.provenance)
            store.admit(updated)
        if name in {'tool_tamper', 'argument_tamper'}:
            step = skill.steps[-1]
            program = replace(step.certificate.program, tool_name='other_write') if name == 'tool_tamper' else replace(step.certificate.program, arguments={'id': 'other_item'})
            altered = replace(skill, steps=(skill.steps[0], replace(step, certificate=replace(step.certificate, program=program))))
            valid, detail = SkillChecker().check(altered, store)
            passed = not valid and detail == 'proof is not bound to the exact step program'
        elif name == 'untrusted_proposal':
            proposal = replace(store.get('review:write'), trust=TrustLevel.PROPOSED)
            try:
                store.admit(proposal)
            except ValueError as exc:
                detail = str(exc); passed = True
            else:
                passed = False
        elif name == 'selective_invalidation':
            other_records = [replace(r, key=EvidenceKey('unrelated', r.key.item_id),
                                     scope_dependencies=frozenset({'other_scope'}), closure_dependencies=frozenset({'other_closure'}))
                             for r in store.records]
            other_store = EvidenceStore(other_records)
            other_steps = []
            for step, record in zip(skill.steps, other_records, strict=True):
                # Original ids are in sorted read, write order.
                cert = Certificate.build(skill.region, step.certificate.program, record.payload,
                                         ProofNode(record.payload, source_id=record.id), other_store)
                other_steps.append(replace(step, certificate=cert))
            other = CertifiedSkill.build(skill.region, skill.required_inputs, skill.preconditions, other_steps)
            for r in other_store.records:
                store.admit(r)
            registry.register(other, store)
            affected = registry.suspend_changed(['scope'])
            passed = affected == (skill.skill_id,) and registry.status(other.skill_id) is CertificateStatus.ACTIVE
            detail = 'affected suspended; disjoint skill remains active'
        else:
            if name == 'principal_mismatch':
                context = replace(context, principal='bob')
            if name == 'closure_invalidation':
                registry.suspend_changed(['closure'])
            result = SkillRuntime(registry, store, backend).execute(context)
            writes = sum(e['tool'] == 'write' for e in backend.effects.values())
            passed = (result.reason == 'skill completed' and writes == 1) if name == 'nominal' else writes == 0
            expected_reason = {'nominal': 'skill completed', 'semantic_edit': 'no matching active skill',
                               'version_edit': 'no matching active skill', 'principal_mismatch': 'no matching active skill',
                               'missing_guard': 'step precondition failed', 'missing_post': 'step postcondition failed; stopped',
                               'state_race': 'state changed before commit', 'policy_race': 'policy changed before commit',
                               'effect_type': 'tool effect type disagrees with reviewed step',
                               'closure_invalidation': 'no matching active skill',
                               'initial_snapshot_race': 'no matching active skill'}[name]
            passed = passed and result.reason == expected_reason
            if name in {'semantic_edit', 'version_edit'}:
                passed = passed and registry.status(skill.skill_id) is CertificateStatus.SUSPENDED
            detail = result.reason
        cases.append({'case': name, 'passed': passed, 'write_effects': sum(e['tool'] == 'write' for e in backend.effects.values()), 'detail': detail})
    if not all(case['passed'] for case in cases):
        raise RuntimeError(json.dumps(cases, indent=2))
    return {'schema_version': 'runtime-fault-review-v1', 'scope': 'constructed contracts; deterministic implementation cases, not independent deployment trials',
            'cases': cases, 'case_count': len(cases), 'passed_count': sum(c['passed'] for c in cases),
            'adverse_case_count': len(cases) - 1, 'unsafe_write_effects': sum(c['write_effects'] for c in cases if c['case'] != 'nominal'),
            'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'runtime_source_sha256': {n: hashlib.sha256((ROOT / 'src/delegation_lab' / n).read_bytes()).hexdigest()
                                      for n in ['skill.py', 'certify.py', 'lifecycle.py', 'runtime.py', 'evidence.py']}}


if __name__ == '__main__':
    output = ROOT / 'results/runtime_review/results.json'
    if output.exists():
        raise FileExistsError('refusing to overwrite runtime review')
    result = run(); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(result, indent=2))
