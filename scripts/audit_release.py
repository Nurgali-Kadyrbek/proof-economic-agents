#!/usr/bin/env python3
"""Audit public selection, integrity, and credential/path leakage."""
from __future__ import annotations
import hashlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'SHA256SUMS'
ALLOWED = {'.py','.json','.yaml','.toml','.md','.cff','.tex','.txt'}
FORBIDDEN_NAMES = {'.env','AGENTS.md','AGENT.md','DECISIONS.md','NOTES.md','template.tex','mdpi.cls'}
FORBIDDEN_PARTS = {'__pycache__','.pytest_cache','.venv','data','artifacts','submission','scheduler','cache','logs'}
PATTERNS = [re.compile(b'/' + b'(?:home|mnt|scratch|storage)/'),
            re.compile(rb'ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|hf_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9]{20,}'),
            re.compile(rb'(?i)(?:api[_-]?key|password|secret[_-]?key)\s*[:=]\s*[\"\x27][^\"\x27\n]{8,}')]

def files():
    return sorted(p for p in ROOT.rglob('*') if p.is_file() and '.git' not in p.relative_to(ROOT).parts)

def audit():
    shipped={}
    for p in files():
        rel=p.relative_to(ROOT).as_posix()
        if p.name in FORBIDDEN_NAMES or FORBIDDEN_PARTS.intersection(p.relative_to(ROOT).parts): raise RuntimeError(f'forbidden release file: {rel}')
        if p.suffix not in ALLOWED and p.name not in {'LICENSE','.gitignore','SHA256SUMS'}: raise RuntimeError(f'unexpected file: {rel}')
        data=p.read_bytes()
        for pattern in PATTERNS:
            if pattern.search(data): raise RuntimeError(f'sensitive path or credential pattern in {rel}')
        if p != MANIFEST: shipped[rel]=hashlib.sha256(data).hexdigest()
    recorded={}
    for line in MANIFEST.read_text().splitlines():
        digest,rel=line.split('  ',1)
        if rel in recorded: raise RuntimeError(f'duplicate manifest entry: {rel}')
        recorded[rel]=digest
    if recorded!=shipped:
        missing=sorted(set(shipped)-set(recorded));extra=sorted(set(recorded)-set(shipped))
        altered=sorted(k for k in set(recorded)&set(shipped) if recorded[k]!=shipped[k])
        raise RuntimeError(f'checksum manifest mismatch: missing={missing}, extra={extra}, altered={altered}')
    print(f'PASS: {len(shipped)} shipped files; checksums, selection, and leakage scan')

if __name__=='__main__': audit()
