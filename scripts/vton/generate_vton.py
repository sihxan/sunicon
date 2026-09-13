#!/usr/bin/env python3
"""Virtual try-on with Qwen-Image-Edit-2511 multi-reference.

Qwen-Image-Edit-2511 (Apache-2.0, QwenImageEditPlusPipeline) takes a LIST of
reference images and lets the prompt address them as "image 1", "image 2". That
is the multi-reference channel this test uses: image 1 is the person to dress,
image 2 is the garment. No try-on LoRA, no mask, no warping network - the
question is what the stock model does with a garment reference.

Two prompt strategies, because the interesting variable is whether the garment
has to be NAMED or whether the picture is enough:

  V1  point at image 2 and say nothing about what is in it
  V2  same, plus a written description of the garment

Sizing gotcha, worth stating because it is silent: the pipeline derives the
output aspect from `image[-1]` (pipeline_qwenimage_edit_plus.py:646), i.e. the
LAST reference, not the first. With [person, garment] that would hand the output
the garment crop's tall narrow aspect. Width and height are therefore passed
explicitly from the person image, and the resolved values are logged.

Generation settings are the repo card's published example for this model
(true_cfg_scale 4.0, negative_prompt " ", 40 steps, guidance_scale 1.0), used as
published rather than retuned, so a weak result is attributable to the model and
not to settings invented here.
"""
from __future__ import annotations
import argparse, hashlib, json, time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "models/qwen_image_edit_2511"

BASE = ("Replace the clothing worn by the person in image 1 with the complete outfit shown in "
        "image 2. Keep the person in image 1 unchanged: same face, same facial features, same "
        "hairstyle, same skin tone, same body proportions, same pose and the same background and "
        "camera framing. Only the clothing changes. Fit the garment naturally to the body with "
        "correct drape, folds and contact shadows, and keep its colour, material, texture, "
        "proportions and details exactly as they appear in image 2.")
STRATEGY = {
    "V1": BASE,
    "V2": BASE + " The outfit in image 2 is {desc}.",
}


def sha256(p) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--persons", nargs="+", required=True, help="person image paths (image 1)")
    ap.add_argument("--garments", type=Path, default=ROOT / "data/garments/munn26fw/garments.json")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--strategies", default="V1,V2")
    ap.add_argument("--steps", type=int, default=40)          # repo card example
    ap.add_argument("--true-cfg", type=float, default=4.0)    # repo card example
    ap.add_argument("--guidance", type=float, default=1.0)    # repo card example
    ap.add_argument("--negative", default=" ")                # repo card example, verbatim
    ap.add_argument("--seed", type=int, default=4411)
    ap.add_argument("--seeds", type=int, nargs="*", default=None,
                    help="repeat every combination at each of these seeds; the filename "
                         "carries the seed only when more than one is given, so a single-seed "
                         "run keeps the names the first run produced")
    ap.add_argument("--garment-ids", default="", help="comma-separated ids; default all in the json")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    gar = json.loads(a.garments.read_text())
    want = {x.strip() for x in a.garment_ids.split(",") if x.strip()}
    if want:
        missing = want - {g["id"] for g in gar}
        if missing:
            raise SystemExit(f"unknown garment ids: {sorted(missing)}")
        gar = [g for g in gar if g["id"] in want]
    seeds = a.seeds or [a.seed]
    strats = [s.strip() for s in a.strategies.split(",") if s.strip()]
    persons = [Path(p) for p in a.persons]
    for p in persons:
        if not p.exists():
            raise SystemExit(f"person image missing: {p}")
    a.out_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for pi, p in enumerate(persons):
        for g in gar:
            for s in strats:
                for sd in seeds:
                    tag = f"_s{sd}" if len(seeds) > 1 else ""
                    jobs.append({"name": f"{p.stem}__{g['id']}_{s}{tag}.png",
                                 "person": str(p), "person_tag": p.stem,
                                 "garment_id": g["id"], "garment": g["out_path"],
                                 "strategy": s, "seed": sd + pi, "seed_base": sd,
                                 "prompt": " ".join(STRATEGY[s].format(desc=g["desc"]).split())})

    print(f"model    {MODEL}  (exists={(MODEL/'model_index.json').exists()})")
    print(f"persons  {len(persons)}   garments {[g['id'] for g in gar]}")
    print(f"strategies {strats}   seeds {seeds}")
    print(f"jobs     {len(jobs)} -> {a.out_dir}")
    if a.dry_run:
        (a.out_dir / "resolved_prompts.json").write_text(json.dumps(jobs, indent=2, ensure_ascii=False))
        for j in jobs[:2]:
            print(f"\n  {j['name']}\n    {j['prompt'][:300]}...")
        print("\nDRY RUN - nothing loaded")
        return 0

    import torch
    from PIL import Image
    from diffusers import QwenImageEditPlusPipeline

    if not (MODEL / "model_index.json").exists():
        raise SystemExit(f"Qwen-Image-Edit-2511 weights absent at {MODEL}")
    print("\nloading pipeline ...", flush=True)
    t0 = time.time()
    pipe = QwenImageEditPlusPipeline.from_pretrained(str(MODEL), torch_dtype=torch.bfloat16).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    print(f"  loaded in {time.time()-t0:.0f}s", flush=True)

    cache = {}
    def load(p):
        if p not in cache:
            cache[p] = Image.open(p).convert("RGB")
        return cache[p]

    rows = []
    for j in jobs:
        person, garment = load(j["person"]), load(j["garment"])
        # output geometry follows the PERSON, overriding the pipeline's image[-1] default
        W = person.width // 32 * 32
        H = person.height // 32 * 32
        g = torch.Generator(device="cuda").manual_seed(j["seed"])
        t = time.time()
        try:
            with torch.inference_mode():
                out = pipe(image=[person, garment], prompt=j["prompt"],
                           negative_prompt=a.negative, true_cfg_scale=a.true_cfg,
                           num_inference_steps=a.steps, guidance_scale=a.guidance,
                           width=W, height=H, num_images_per_prompt=1, generator=g).images[0]
            out.save(a.out_dir / j["name"]); status, err = "ok", ""
            osz = list(out.size)
        except Exception as exc:                                        # noqa: BLE001
            status, err, osz = "failed", f"{type(exc).__name__}: {exc}"[:400], None
        dt = time.time() - t
        print(f"  {j['name']:<44} {status:<7} {dt:5.1f}s  req={W}x{H} got={osz}", flush=True)
        rows.append({**j, "status": status, "error": err, "seconds": round(dt, 1),
                     "requested_wh": [W, H], "output_wh": osz,
                     "person_wh": list(person.size), "garment_wh": list(garment.size),
                     "person_sha256": sha256(j["person"]), "garment_sha256": sha256(j["garment"])})

    man = {"experiment": "qwen_edit_2511_vton_v1",
           "created_at": datetime.now().isoformat(timespec="seconds"),
           "model": "Qwen/Qwen-Image-Edit-2511", "model_dir": str(MODEL),
           "pipeline": "QwenImageEditPlusPipeline", "reference_order": "[person, garment]",
           "settings": {"steps": a.steps, "true_cfg_scale": a.true_cfg,
                        "guidance_scale": a.guidance, "negative_prompt": a.negative,
                        "source": "Qwen-Image-Edit-2511 model card example, used unmodified"},
           "sizing_note": "width/height forced from the person image; the pipeline would "
                          "otherwise take the aspect from image[-1] (the garment).",
           "prompt_strategies": STRATEGY, "jobs": rows}
    (a.out_dir / "manifest.json").write_text(json.dumps(man, indent=2, ensure_ascii=False))
    ok = sum(1 for r in rows if r["status"] == "ok")
    print(f"\n{ok}/{len(rows)} ok -> {a.out_dir}")
    return 0 if ok == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
