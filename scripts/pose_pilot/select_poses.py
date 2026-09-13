#!/usr/bin/env python3
"""Pick pose control maps for the FLUX.2 pose-transfer pilot.

Source: the OLD project's DWPose pass over the MUNN 26FW lookbook
(data/pose/munn_26fw). Only single-person, full-body frames with all 18
OpenPose body joints are eligible, because the RefControl README asks for a
clean uncluttered skeleton and a multi-model frame renders several.

Using the SKELETON rather than the lookbook photo as the structural reference
is also what keeps a real model's appearance out of the conditioning: a
DWPose render carries joint positions and nothing else.

Diversity: greedy farthest-point over hip-normalised joint coordinates, so the
chosen poses are as unlike each other as the bank allows.
"""
from __future__ import annotations
import csv, json, shutil
from pathlib import Path
import numpy as np

OLD = Path("/home/shkim/repos/sunicon_base_modelcut_generation-main")
POSE = OLD / "data/pose/munn_26fw"
OUT = Path("data/pose_refs/munn26fw"); OUT.mkdir(parents=True, exist_ok=True)
N = 4

rows = list(csv.DictReader((POSE / "manifests/pose_results.csv").open(encoding="utf-8")))
cand = [r for r in rows
        if r["shot_type"] == "full_body"
        and int(r["primary_visible_joints"]) == 18
        and int(r["detected_person_count"]) == 1]
print(f"{len(rows)} frames -> {len(cand)} single-person full-body 18-joint candidates")

feats, keep = [], []
for r in cand:
    d = json.loads((POSE / "json" / (Path(r["filename"]).stem + ".json")).read_text())
    kp = np.array(d["people"][d["primary_person_index"]]["body_keypoints_18"], dtype=np.float32)
    xy, conf = kp[:, :2], kp[:, 2]
    if (conf <= 0).any():
        continue
    # hip-centre origin, torso-length scale -> translation/scale invariant
    hipc = (xy[8] + xy[11]) / 2
    torso = np.linalg.norm(xy[1] - hipc) + 1e-6
    feats.append(((xy - hipc) / torso).ravel()); keep.append((r, d))
F = np.stack(feats)

sel = [int(np.linalg.norm(F - F.mean(0), axis=1).argmax())]          # most extreme first
while len(sel) < N:
    d = np.min(np.linalg.norm(F[:, None] - F[sel][None], axis=2), axis=1)
    d[sel] = -1
    sel.append(int(d.argmax()))

meta = []
for i, idx in enumerate(sel):
    r, d = keep[idx]
    stem = Path(r["filename"]).stem
    src = OLD / r["pose_vis_primary_path"]
    dst = OUT / f"pose{i+1:02d}_{stem}.png"
    shutil.copy2(src, dst)
    kp = np.array(d["people"][d["primary_person_index"]]["body_keypoints_18"], dtype=np.float32)
    meta.append({"pose_id": f"pose{i+1:02d}", "source_frame": r["filename"],
                 "skeleton_png": str(dst), "source_vis": str(src),
                 "body_keypoints_18": kp.tolist(),
                 "render_wh": [int(d["render_width"]), int(d["render_height"])]})
    print(f"  {dst.name:<28} from {r['filename']}")

(OUT / "pose_refs.json").write_text(json.dumps(meta, indent=2))
print(f"\n{len(meta)} pose refs -> {OUT}")
