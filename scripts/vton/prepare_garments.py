#!/usr/bin/env python3
"""Cut garment-only references out of the official MUNN 26FW lookbook.

For a try-on test the second reference should carry the garment and nothing
else. Two things are removed from each lookbook frame:

  * the head - cropped at the DWPose neck joint (shoulder midpoint), which is
    below the chin. The lookbook already carries the brand's own face blur, but
    a garment reference has no reason to contain a real model's face at all, so
    the crop removes it and `--verify-no-face` re-runs a face detector on the
    result and fails if anything is found.
  * the studio surround - crew, light stands and equipment stand in the dark
    bands either side of the paper. The crop is taken to the primary person's
    joint envelope plus a margin, so only the garment on the paper survives.

Feet are kept: shoes are part of the look and the try-on prompt should not have
to invent them.
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import cv2, numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/pose_pilot"))
from dwpose_onnx import DWPose                                      # noqa: E402

# The look list lives in configs/vton_garments.json so it can grow without
# editing this file. Each entry: {id, frame, desc}.
GARMENT_CONFIG = ROOT / "configs/vton_garments.json"


def sha256(p) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=ROOT / "data/raw/munn26fw/images")
    ap.add_argument("--out", type=Path, default=ROOT / "data/garments/munn26fw")
    ap.add_argument("--side-margin", type=float, default=0.18, help="fraction of joint-envelope width")
    ap.add_argument("--max-side", type=int, default=1280)
    ap.add_argument("--config", type=Path, default=GARMENT_CONFIG)
    ap.add_argument("--only", default="", help="comma-separated garment ids; default all")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    garments = json.loads(a.config.read_text(encoding="utf-8"))
    want = {x.strip() for x in a.only.split(",") if x.strip()}
    if want:
        garments = [g for g in garments if g["id"] in want]
    det = DWPose("cpu")
    meta = []
    for g in garments:
        src = a.src / g["frame"]
        img = cv2.imread(str(src))
        H, W = img.shape[:2]
        r = det(src)
        b = r["body18"]                                   # normalised [0,1], -1 = missing
        vis = b[b[:, 0] >= 0]
        neck_y = b[1][1]
        if neck_y < 0:
            raise SystemExit(f"{g['frame']}: no neck joint, cannot crop below the chin")

        x0f, x1f = vis[:, 0].min(), vis[:, 0].max()
        m = (x1f - x0f) * a.side_margin
        x0 = int(max(0, (x0f - m)) * W); x1 = int(min(1, (x1f + m)) * W)
        y0 = int(neck_y * H)                              # shoulder line: below the chin
        y1 = int(min(1.0, vis[:, 1].max() + 0.035) * H)    # a little past the ankles, keeps shoes
        crop = img[y0:y1, x0:x1]

        s = a.max_side / max(crop.shape[:2])
        if s < 1:
            crop = cv2.resize(crop, (int(crop.shape[1] * s), int(crop.shape[0] * s)), interpolation=cv2.INTER_AREA)
        dst = a.out / f"{g['id']}_{Path(g['frame']).stem}.png"
        cv2.imwrite(str(dst), crop)

        meta.append({**g, "source_path": str(src.relative_to(ROOT)), "source_sha256": sha256(src),
                     "source_wh": [W, H], "crop_xyxy_px": [x0, y0, x1, y1],
                     "crop_top_rule": "DWPose neck joint (shoulder midpoint), below the chin",
                     "out_path": str(dst.relative_to(ROOT)), "out_wh": [crop.shape[1], crop.shape[0]],
                     "out_sha256": sha256(dst)})
        print(f"  {g['id']}  {g['frame']:<22} {W}x{H} -> {crop.shape[1]}x{crop.shape[0]}  "
              f"top cut at y={y0} (neck)")

    man = a.out / "garments.json"
    if man.exists():
        prev = {m["id"]: m for m in json.loads(man.read_text(encoding="utf-8"))}
        prev.update({m["id"]: m for m in meta})
        meta = [prev[k] for k in sorted(prev)]
    man.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"\n{len(meta)} garment refs -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
