#!/usr/bin/env python3
"""Score the pose pilot on the two axes that are actually in tension.

  pose      how closely the generated body follows the skeleton it was given
  identity  how much of the Master's face survived that

Run in two stages because the two models live in different venvs: DWPose (ONNX)
is in envs/train, insightface/antelopev2 is in envs/infer.

  envs/train/bin/python ... --stage pose
  envs/infer/bin/python ... --stage identity
  envs/train/bin/python ... --stage report

Pose metric: joints are hip-centred and divided by torso length, so the score is
invariant to where in the frame the body lands and how big it is - only the
configuration counts. `joint_err` is the mean distance in torso units over the
joints both poses share; `pck02` is the fraction within 0.2 torso.

Identity caveat, unchanged from the earlier runs: ArcFace cosine is not an
independent judge when an ArcFace-derived embedding also drives the generator.
Nothing in THIS experiment conditions on ArcFace - the identity signal is the
Master image passed through the model's own reference channel - so here the
metric is independent. It still says nothing about whether the face is
attractive, on-brand, or the right gender.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


# OpenPose-18 left/right pairs, for the mirror check below.
_LR = [(2, 5), (3, 6), (4, 7), (8, 11), (9, 12), (10, 13), (14, 15), (16, 17)]


def mirror(p):
    """Reflect in x AND swap the left/right labels - a true mirror of the body."""
    q = np.asarray(p, np.float32).copy()
    q[:, 0] *= -1
    for a, b in _LR:
        q[[a, b]] = q[[b, a]]
    return q


def facing(p):
    """Signed left-minus-right shoulder x. Positive = the body faces the camera."""
    return float(p[5, 0] - p[2, 0])


def stage_pose(run: Path, poses: Path) -> None:
    sys.path.insert(0, str(ROOT / "scripts/pose_pilot"))
    from dwpose_onnx import DWPose, norm_pose
    det = DWPose("cpu")
    src = {p["pose_id"]: np.array(p["body_keypoints_18"], np.float32)[:, :2]
           for p in json.loads(poses.read_text())}
    man = json.loads((run / "manifest.json").read_text())
    out = {}
    for j in man["jobs"]:
        if j["status"] != "ok":
            continue
        img = run / j["name"]
        r = det(img)
        gen_n = norm_pose(r["body18"]) if r["n_people"] else None
        ref_n = norm_pose(src[j["pose_id"]])
        rec = {"n_people_detected": r["n_people"], "n_visible_joints": int(r.get("n_visible", 0))}
        if gen_n is None or ref_n is None:
            rec.update({"joint_err": None, "pck02": None, "note": "no usable skeleton in output"})
        else:
            ok = ~(np.isnan(gen_n).any(1) | np.isnan(ref_n).any(1))
            d = np.linalg.norm(gen_n[ok] - ref_n[ok], axis=1)
            # Also score the mirrored generated pose. A limb configuration can be
            # reproduced correctly while the body faces the other way; without this
            # the two failures are indistinguishable in one number.
            dm = np.linalg.norm(mirror(gen_n)[ok] - ref_n[ok], axis=1)
            faces_same = facing(gen_n) * facing(ref_n) > 0
            rec.update({"joint_err": round(float(d.mean()), 4),
                        "pck02": round(float((d < 0.2).mean()), 3),
                        "joint_err_mirrored": round(float(dm.mean()), 4),
                        "joint_err_best": round(float(min(d.mean(), dm.mean())), 4),
                        "pck02_best": round(float(max((d < 0.2).mean(), (dm < 0.2).mean())), 3),
                        "facing_matches_reference": bool(faces_same),
                        "joints_compared": int(ok.sum())})
        out[j["name"]] = rec
        print(f"  {j['name']:<22} people={rec['n_people_detected']} "
              f"err={rec['joint_err']} pck02={rec['pck02']}")
    (run / "metrics_pose.json").write_text(json.dumps(out, indent=2))


def stage_identity(run: Path, master: Path) -> None:
    import warnings, cv2
    warnings.filterwarnings("ignore")
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="antelopev2", root=str(ROOT / "models/InfiniteYou/supports/insightface"),
                       providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))

    def embed(p):
        fs = app.get(cv2.imread(str(p)))
        if not fs:
            return None, None
        f = max(fs, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
        return f.normed_embedding, f.bbox.astype(int)

    me, mb = embed(master)
    if me is None:
        raise SystemExit(f"no face found in master {master}")
    print(f"  master face bbox {mb.tolist()}  h={mb[3]-mb[1]}px")
    man = json.loads((run / "manifest.json").read_text())
    out = {}
    for j in man["jobs"]:
        if j["status"] != "ok":
            continue
        e, b = embed(run / j["name"])
        if e is None:
            out[j["name"]] = {"arcface_cos": None, "face_h_px": 0, "note": "no face detected"}
        else:
            out[j["name"]] = {"arcface_cos": round(float(np.dot(me, e)), 4),
                              "face_h_px": int(b[3] - b[1])}
        print(f"  {j['name']:<22} cos={out[j['name']]['arcface_cos']} "
              f"face_h={out[j['name']]['face_h_px']}px")
    (run / "metrics_identity.json").write_text(json.dumps(out, indent=2))


def stage_report(run: Path) -> None:
    man = json.loads((run / "manifest.json").read_text())
    P = json.loads((run / "metrics_pose.json").read_text())
    I = json.loads((run / "metrics_identity.json").read_text())
    rows = []
    for j in man["jobs"]:
        if j["status"] != "ok":
            continue
        rows.append({"name": j["name"], "pose_id": j["pose_id"], "condition": j["condition"],
                     "lora_scale": j["lora_scale"], **P.get(j["name"], {}), **I.get(j["name"], {})})
    (run / "metrics_joined.json").write_text(json.dumps(rows, indent=2))

    order = ["T0", "N1", "R08", "R10"]
    label = {"T0": "text pose only, refs=[master]", "N1": "native multi-ref, refs=[skel,master]",
             "R08": "RefControl LoRA 0.8", "R10": "RefControl LoRA 1.0"}
    lines = ["", "cond  n   joint_err   err_mirror-aware   pck@0.2   arcface_cos   face_px   facing_ok",
             "-" * 88]
    summary = {}
    for c in order:
        rs = [r for r in rows if r["condition"] == c]
        if not rs:
            continue
        je = [r["joint_err"] for r in rs if r.get("joint_err") is not None]
        jb = [r["joint_err_best"] for r in rs if r.get("joint_err_best") is not None]
        pk = [r["pck02_best"] for r in rs if r.get("pck02_best") is not None]
        fo = sum(1 for r in rs if r.get("facing_matches_reference"))
        cs = [r["arcface_cos"] for r in rs if r.get("arcface_cos") is not None]
        fp = [r["face_h_px"] for r in rs if r.get("face_h_px")]
        nf = sum(1 for r in rs if not r.get("arcface_cos"))
        summary[c] = {"n": len(rs), "label": label[c],
                      "joint_err_mean": round(float(np.mean(je)), 4) if je else None,
                      "joint_err_best_mean": round(float(np.mean(jb)), 4) if jb else None,
                      "facing_matches_reference": f"{fo}/{len(rs)}",
                      "pck02_mean": round(float(np.mean(pk)), 3) if pk else None,
                      "arcface_cos_mean": round(float(np.mean(cs)), 4) if cs else None,
                      "arcface_cos_min": round(float(np.min(cs)), 4) if cs else None,
                      "face_h_px_mean": int(np.mean(fp)) if fp else 0,
                      "outputs_with_no_face": nf}
        s = summary[c]
        lines.append(f"{c:<5} {s['n']:<3} {str(s['joint_err_mean']):<11} {str(s['joint_err_best_mean']):<18} "
                     f"{str(s['pck02_mean']):<10}{str(s['arcface_cos_mean']):<14}{s['face_h_px_mean']:<10}"
                     f"{s['facing_matches_reference']}")
    lines += ["", "per-image:",
              f"{'name':<24}{'err':>8}{'err_mir':>9}{'pck':>7}{'cos':>8}{'facepx':>8}{'facing':>8}"]
    for r in sorted(rows, key=lambda x: (x["pose_id"], order.index(x["condition"]))):
        lines.append(f"{r['name']:<24}{str(r.get('joint_err')):>8}{str(r.get('joint_err_best')):>9}"
                     f"{str(r.get('pck02_best')):>7}{str(r.get('arcface_cos')):>8}"
                     f"{str(r.get('face_h_px')):>8}{str(r.get('facing_matches_reference')):>8}")
    txt = "\n".join(lines)
    print(txt)
    (run / "summary.txt").write_text(txt)
    (run / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--stage", choices=["pose", "identity", "report"], required=True)
    ap.add_argument("--poses", type=Path, default=ROOT / "data/pose_refs/munn26fw/pose_refs.json")
    ap.add_argument("--master", type=Path, default=ROOT / "identities/MUNN26FW_A/master.png")
    a = ap.parse_args()
    {"pose": lambda: stage_pose(a.run, a.poses),
     "identity": lambda: stage_identity(a.run, a.master),
     "report": lambda: stage_report(a.run)}[a.stage]()
