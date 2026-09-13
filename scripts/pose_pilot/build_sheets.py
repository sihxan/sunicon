#!/usr/bin/env python3
"""Contact sheets for the pose pilot: one grid of full frames, one of face crops.

Rows are poses, columns are conditions, with the skeleton that was asked for and
the Master that was supposed to be preserved in the two leftmost columns - so a
row reads left to right as "this pose, this face, and what each method did".
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
ORDER = ["T0", "N1", "R08", "R10"]
TITLES = {"T0": "T0  text pose\nrefs=[master]", "N1": "N1  native multi-ref\nrefs=[skel,master]",
          "R08": "R08  RefControl 0.8", "R10": "R10  RefControl 1.0"}


def paste_fit(canvas, im, x, y, w, h):
    s = min(w / im.width, h / im.height)
    im2 = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.LANCZOS)
    canvas.paste(im2, (x + (w - im2.width) // 2, y + (h - im2.height) // 2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--master", type=Path, default=ROOT / "identities/MUNN26FW_A/master.png")
    ap.add_argument("--cell-h", type=int, default=460)
    a = ap.parse_args()

    man = json.loads((a.run / "manifest.json").read_text())
    poses = sorted({j["pose_id"] for j in man["jobs"]})
    skel = {j["pose_id"]: j["skeleton_png"] for j in man["jobs"]}
    metrics = {}
    mp = a.run / "metrics_joined.json"
    if mp.exists():
        metrics = {r["name"]: r for r in json.loads(mp.read_text())}

    CW, CH, HDR, LAB = int(a.cell_h * 0.8), a.cell_h, 40, 30
    cols = 2 + len(ORDER)
    W, H = CW * cols + 10 * (cols + 1), HDR + (CH + LAB) * len(poses) + 10
    sheet = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(sheet)
    heads = ["pose skeleton\n(reference 1)", "MASTER\n(reference 2)"] + [TITLES[c] for c in ORDER]
    for i, t in enumerate(heads):
        d.multiline_text((10 + i * (CW + 10) + 4, 4), t, fill="black", spacing=2)

    master = Image.open(a.master).convert("RGB")
    for r, pid in enumerate(poses):
        y = HDR + r * (CH + LAB)
        paste_fit(sheet, Image.open(skel[pid]).convert("RGB"), 10, y, CW, CH)
        paste_fit(sheet, master, 10 + CW + 10, y, CW, CH)
        d.text((14, y + CH + 6), pid, fill="black")
        for c, cond in enumerate(ORDER):
            x = 10 + (c + 2) * (CW + 10)
            p = a.run / f"{pid}_{cond}.png"
            if p.exists():
                paste_fit(sheet, Image.open(p).convert("RGB"), x, y, CW, CH)
                m = metrics.get(p.name)
                if m:
                    d.text((x + 4, y + CH + 6),
                           f"err {m.get('joint_err')}  pck {m.get('pck02')}  cos {m.get('arcface_cos')}",
                           fill="black")
            else:
                d.text((x + 4, y + CH // 2), "missing", fill="red")
    out = a.run / "contact_sheet.png"
    sheet.save(out)
    print("full frames ->", out, sheet.size)

    # face crops: the pose grid is too small to judge a face at 90 px
    try:
        import warnings, cv2, numpy as np
        warnings.filterwarnings("ignore")
        from insightface.app import FaceAnalysis
    except ImportError:
        print("insightface not in this venv - skipping face sheet "
              "(run with envs/infer/bin/python for it)")
        return 0
    app = FaceAnalysis(name="antelopev2", root=str(ROOT / "models/InfiniteYou/supports/insightface"),
                       providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))

    def crop_face(p, pad=0.55):
        img = cv2.imread(str(p))
        fs = app.get(img)
        if not fs:
            return None
        f = max(fs, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
        x0, y0, x1, y1 = f.bbox
        cx, cy, s = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0) * (1 + pad)
        h, w = img.shape[:2]
        a0, b0 = int(max(0, cx - s / 2)), int(max(0, cy - s / 2))
        a1, b1 = int(min(w, cx + s / 2)), int(min(h, cy + s / 2))
        return Image.fromarray(cv2.cvtColor(img[b0:b1, a0:a1], cv2.COLOR_BGR2RGB))

    FC = 230
    fcols = 1 + len(ORDER)
    fsheet = Image.new("RGB", (FC * fcols + 10 * (fcols + 1), HDR + (FC + LAB) * len(poses) + 10), "white")
    fd = ImageDraw.Draw(fsheet)
    for i, t in enumerate(["MASTER"] + [c for c in ORDER]):
        fd.text((10 + i * (FC + 10) + 4, 10), t, fill="black")
    mf = crop_face(a.master)
    for r, pid in enumerate(poses):
        y = HDR + r * (FC + LAB)
        if mf:
            paste_fit(fsheet, mf, 10, y, FC, FC)
        fd.text((14, y + FC + 6), pid, fill="black")
        for c, cond in enumerate(ORDER):
            x = 10 + (c + 1) * (FC + 10)
            p = a.run / f"{pid}_{cond}.png"
            if not p.exists():
                continue
            fc = crop_face(p)
            if fc is None:
                fd.text((x + 4, y + FC // 2), "no face", fill="red")
                continue
            paste_fit(fsheet, fc, x, y, FC, FC)
            m = metrics.get(p.name)
            if m:
                fd.text((x + 4, y + FC + 6), f"cos {m.get('arcface_cos')}  {m.get('face_h_px')}px", fill="black")
    fout = a.run / "face_sheet.png"
    fsheet.save(fout)
    print("face crops ->", fout, fsheet.size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
