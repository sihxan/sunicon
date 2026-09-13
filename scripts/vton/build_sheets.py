#!/usr/bin/env python3
"""Try-on contact sheet: rows are person x garment, columns are prompt strategies.

The two inputs sit at the left of every row, so a row reads as
"this person, this garment, and what came back".
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]


def paste_fit(c, im, x, y, w, h):
    s = min(w / im.width, h / im.height)
    im2 = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.LANCZOS)
    c.paste(im2, (x + (w - im2.width) // 2, y + (h - im2.height) // 2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--cell-h", type=int, default=440)
    a = ap.parse_args()
    man = json.loads((a.run / "manifest.json").read_text())
    jobs = [j for j in man["jobs"] if j["status"] == "ok"]
    strats = sorted({j["strategy"] for j in jobs})
    keys = sorted({(j["person_tag"], j["garment_id"]) for j in jobs})
    metrics = {}
    mp = a.run / "metrics_joined.json"
    if mp.exists():
        metrics = {r["name"]: r for r in json.loads(mp.read_text())}

    CW, CH, HDR, LAB = int(a.cell_h * 0.8), a.cell_h, 34, 34
    cols = 2 + len(strats)
    sheet = Image.new("RGB", (CW * cols + 10 * (cols + 1), HDR + (CH + LAB) * len(keys) + 10), "white")
    d = ImageDraw.Draw(sheet)
    for i, t in enumerate(["person (image 1)", "garment (image 2)"] +
                          [f"{s}  {'picture only' if s=='V1' else 'picture + description'}" for s in strats]):
        d.text((10 + i * (CW + 10) + 4, 10), t, fill="black")
    for r, (ptag, gid) in enumerate(keys):
        y = HDR + r * (CH + LAB)
        j0 = next(j for j in jobs if j["person_tag"] == ptag and j["garment_id"] == gid)
        paste_fit(sheet, Image.open(j0["person"]).convert("RGB"), 10, y, CW, CH)
        paste_fit(sheet, Image.open(j0["garment"]).convert("RGB"), 10 + CW + 10, y, CW, CH)
        d.text((14, y + CH + 6), f"{ptag}  {gid}", fill="black")
        for c, s in enumerate(strats):
            x = 10 + (c + 2) * (CW + 10)
            hit = [j for j in jobs if j["person_tag"] == ptag and j["garment_id"] == gid and j["strategy"] == s]
            if not hit:
                d.text((x + 4, y + CH // 2), "missing", fill="red"); continue
            p = a.run / hit[0]["name"]
            paste_fit(sheet, Image.open(p).convert("RGB"), x, y, CW, CH)
            m = metrics.get(p.name)
            if m:
                d.text((x + 4, y + CH + 6),
                       f"garment {m.get('dino_cos_to_garment')}  face {m.get('arcface_cos_to_person_input')}"
                       f"  pose {m.get('pose_err_vs_person')}", fill="black")
    out = a.run / "contact_sheet.png"
    sheet.save(out); print("->", out, sheet.size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
