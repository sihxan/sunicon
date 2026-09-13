#!/usr/bin/env python3
"""Score try-on outputs on the three things that can independently go wrong.

  garment   did the outfit from image 2 actually arrive
  identity  did the face from image 1 survive it
  pose      did the body stay in the pose it was given

garment  DINOv2-large cosine between the torso crop of the output and the
         garment reference, plus a plain CIE-Lab colour distance on the same
         crop. DINOv2 is used rather than CLIP because the question is
         instance-level ("is this that garment") not semantic ("is this a
         coat"), and colour delta-E is reported alongside because a high
         embedding score with the wrong colour is a failure a single number
         hides. The torso box comes from DWPose shoulders and hips, so the
         output and the reference are cropped by the same rule.

identity  ArcFace cosine against the PERSON INPUT, not the Master - try-on is
         asked to preserve what it was handed. Cosine against the Master is
         recorded too, to show the drift through the whole chain.

pose     DWPose joint error between output and person input, hip-centred and
         torso-normalised.

Two stages again, because DWPose+DINOv2 are in envs/train and insightface is in
envs/infer.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def torso_box(body18, wh, pad=0.10):
    """Shoulder-to-hip box in pixels, widened a little to catch sleeves and hem."""
    xy = np.asarray(body18, np.float32)[:, :2]
    idx = [2, 5, 8, 11]                                   # R/L shoulder, R/L hip
    if any(xy[i][0] < 0 for i in idx):
        return None
    W, H = wh
    xs = xy[idx, 0] * W; ys = xy[idx, 1] * H
    x0, x1 = xs.min(), xs.max(); y0, y1 = ys.min(), ys.max()
    mx, my = (x1 - x0) * (pad + 0.25), (y1 - y0) * pad
    return [int(max(0, x0 - mx)), int(max(0, y0 - my)), int(min(W, x1 + mx)), int(min(H, y1 + my))]


def stage_visual(run: Path) -> None:
    sys.path.insert(0, str(ROOT / "scripts/pose_pilot"))
    import cv2, torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel
    from dwpose_onnx import DWPose, norm_pose

    det = DWPose("cpu")
    proc = AutoImageProcessor.from_pretrained(str(ROOT / "models/dinov2-large"))
    dino = AutoModel.from_pretrained(str(ROOT / "models/dinov2-large")).eval()

    def embed(pil):
        with torch.no_grad():
            o = dino(**proc(images=pil, return_tensors="pt")).pooler_output[0]
        return (o / o.norm()).numpy()

    def lab_mean(pil):
        a = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2LAB).astype(np.float32)
        return a.reshape(-1, 3).mean(0)

    def torso_crop(path):
        """Crop by DWPose torso box; fall back to the middle band if no skeleton."""
        r = det(path)
        im = Image.open(path).convert("RGB")
        if r["n_people"]:
            b = torso_box(r["body18"], im.size)
            if b:
                return im.crop(tuple(b)), r, b
        W, H = im.size
        return im.crop((0, int(H * 0.10), W, int(H * 0.62))), r, None

    man = json.loads((run / "manifest.json").read_text())
    cache = {}
    out = {}
    for j in man["jobs"]:
        if j["status"] != "ok":
            continue
        for key in (j["garment"], j["person"]):
            if key not in cache:
                c, r, b = torso_crop(key)
                cache[key] = {"emb": embed(c), "lab": lab_mean(c), "pose": r, "box": b}
        gc, pc = cache[j["garment"]], cache[j["person"]]
        oc, orr, ob = torso_crop(str(run / j["name"]))
        oe, ol = embed(oc), lab_mean(oc)

        gp = norm_pose(pc["pose"]["body18"]) if pc["pose"]["n_people"] else None
        op = norm_pose(orr["body18"]) if orr["n_people"] else None
        if gp is not None and op is not None:
            ok = ~(np.isnan(gp).any(1) | np.isnan(op).any(1))
            pose_err = round(float(np.linalg.norm(gp[ok] - op[ok], axis=1).mean()), 4)
        else:
            pose_err = None

        rec = {
            "dino_cos_to_garment": round(float(np.dot(oe, gc["emb"])), 4),
            "dino_cos_to_person_original": round(float(np.dot(oe, pc["emb"])), 4),
            "lab_dist_to_garment": round(float(np.linalg.norm(ol - gc["lab"])), 2),
            "lab_dist_to_person_original": round(float(np.linalg.norm(ol - pc["lab"])), 2),
            "pose_err_vs_person": pose_err, "torso_box": ob,
            "people_detected_in_output": orr["n_people"],
        }
        out[j["name"]] = rec
        print(f"  {j['name']:<44} dinoG={rec['dino_cos_to_garment']:.3f} "
              f"dinoP={rec['dino_cos_to_person_original']:.3f} "
              f"labG={rec['lab_dist_to_garment']:6.2f} pose_err={pose_err}")
    (run / "metrics_visual.json").write_text(json.dumps(out, indent=2))


def stage_identity(run: Path, master: Path) -> None:
    import warnings, cv2
    warnings.filterwarnings("ignore")
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="antelopev2", root=str(ROOT / "models/InfiniteYou/supports/insightface"),
                       providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))

    def emb(p):
        fs = app.get(cv2.imread(str(p)))
        if not fs:
            return None, 0
        f = max(fs, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
        return f.normed_embedding, int(f.bbox[3] - f.bbox[1])

    me, mh = emb(master)
    man = json.loads((run / "manifest.json").read_text())
    pcache, out = {}, {}
    for j in man["jobs"]:
        if j["status"] != "ok":
            continue
        if j["person"] not in pcache:
            pcache[j["person"]] = emb(j["person"])
        pe, ph = pcache[j["person"]]
        oe, oh = emb(run / j["name"])
        rec = {"face_h_px": oh,
               "arcface_cos_to_person_input": None if (oe is None or pe is None) else round(float(np.dot(oe, pe)), 4),
               "arcface_cos_to_master": None if (oe is None or me is None) else round(float(np.dot(oe, me)), 4),
               "person_input_face_h_px": ph}
        out[j["name"]] = rec
        print(f"  {j['name']:<44} cos_person={rec['arcface_cos_to_person_input']} "
              f"cos_master={rec['arcface_cos_to_master']} face_h={oh}px")
    out["_master_face_h_px"] = mh
    (run / "metrics_identity.json").write_text(json.dumps(out, indent=2))


def stage_report(run: Path) -> None:
    man = json.loads((run / "manifest.json").read_text())
    V = json.loads((run / "metrics_visual.json").read_text())
    I = json.loads((run / "metrics_identity.json").read_text())
    rows = [{"name": j["name"], "person_tag": j["person_tag"], "garment_id": j["garment_id"],
             "strategy": j["strategy"], **V.get(j["name"], {}), **I.get(j["name"], {})}
            for j in man["jobs"] if j["status"] == "ok"]
    (run / "metrics_joined.json").write_text(json.dumps(rows, indent=2))

    def agg(rs, k):
        v = [r[k] for r in rs if r.get(k) is not None]
        return round(float(np.mean(v)), 4) if v else None

    lines = ["", "by prompt strategy", "-" * 92,
             f"{'strat':<6}{'n':>3}{'dino->garment':>16}{'dino->orig':>12}{'lab->garment':>14}"
             f"{'cos->person':>13}{'cos->master':>13}{'pose_err':>10}"]
    summary = {}
    for s in sorted({r["strategy"] for r in rows}):
        rs = [r for r in rows if r["strategy"] == s]
        summary[s] = {k: agg(rs, k) for k in
                      ("dino_cos_to_garment", "dino_cos_to_person_original", "lab_dist_to_garment",
                       "arcface_cos_to_person_input", "arcface_cos_to_master", "pose_err_vs_person")}
        summary[s]["n"] = len(rs)
        d = summary[s]
        lines.append(f"{s:<6}{len(rs):>3}{str(d['dino_cos_to_garment']):>16}"
                     f"{str(d['dino_cos_to_person_original']):>12}{str(d['lab_dist_to_garment']):>14}"
                     f"{str(d['arcface_cos_to_person_input']):>13}{str(d['arcface_cos_to_master']):>13}"
                     f"{str(d['pose_err_vs_person']):>10}")
    lines += ["", "by garment", "-" * 92]
    for g in sorted({r["garment_id"] for r in rows}):
        rs = [r for r in rows if r["garment_id"] == g]
        lines.append(f"{g:<6}{len(rs):>3}{str(agg(rs,'dino_cos_to_garment')):>16}"
                     f"{str(agg(rs,'dino_cos_to_person_original')):>12}"
                     f"{str(agg(rs,'lab_dist_to_garment')):>14}"
                     f"{str(agg(rs,'arcface_cos_to_person_input')):>13}"
                     f"{str(agg(rs,'arcface_cos_to_master')):>13}"
                     f"{str(agg(rs,'pose_err_vs_person')):>10}")
    lines += ["", "per image", "-" * 92,
              f"{'name':<46}{'dinoG':>8}{'labG':>8}{'cosP':>8}{'cosM':>8}{'pose':>8}"]
    for r in sorted(rows, key=lambda x: x["name"]):
        lines.append(f"{r['name']:<46}{str(r.get('dino_cos_to_garment')):>8}"
                     f"{str(r.get('lab_dist_to_garment')):>8}"
                     f"{str(r.get('arcface_cos_to_person_input')):>8}"
                     f"{str(r.get('arcface_cos_to_master')):>8}"
                     f"{str(r.get('pose_err_vs_person')):>8}")
    txt = "\n".join(lines)
    print(txt)
    (run / "summary.txt").write_text(txt)
    (run / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--stage", choices=["visual", "identity", "report"], required=True)
    ap.add_argument("--master", type=Path, default=ROOT / "identities/MUNN26FW_A/master.png")
    a = ap.parse_args()
    {"visual": lambda: stage_visual(a.run),
     "identity": lambda: stage_identity(a.run, a.master),
     "report": lambda: stage_report(a.run)}[a.stage]()
