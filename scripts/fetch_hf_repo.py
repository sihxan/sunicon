#!/usr/bin/env python3
"""Fetch an ungated HF repo onto the shared volume.

Local nvme is at 100% (<1 GiB free) and CLAUDE_EXECUTION_ORDER requires >=15 GiB
headroom on the working filesystem, so new weights land on /dataset (94 TB free,
562 MB/s write measured) and are symlinked back into models/.

`huggingface-cli` in this env is a deprecation stub that exits 0 without
downloading, so this uses the library API directly.
"""
import argparse, sys
from pathlib import Path
from huggingface_hub import snapshot_download

ap = argparse.ArgumentParser()
ap.add_argument("repo")
ap.add_argument("dest")
ap.add_argument("--allow", nargs="*", default=None)
a = ap.parse_args()

p = snapshot_download(
    repo_id=a.repo, local_dir=a.dest, allow_patterns=a.allow,
    max_workers=8, tqdm_class=None,
)
files = sorted(x for x in Path(p).rglob("*") if x.is_file() and ".cache" not in x.parts)
total = sum(x.stat().st_size for x in files)
print(f"\n{a.repo} -> {p}")
print(f"{len(files)} files, {total/1e9:.2f} GB")
for f in files:
    if f.stat().st_size > 100e6:
        print(f"  {f.stat().st_size/1e9:7.2f} GB  {f.relative_to(p)}")
