#!/usr/bin/env python3
"""FLUX.2 pose transfer with identity retention - 4 conditions x 4 poses.

Question: when the pose is forced to change, what keeps the selected virtual
face intact?

Black Forest Labs ships no ControlNet for FLUX.2. Their own guidance
(docs.bfl.ai/guides/usecases_editing_controlnets) is that structural control is
done through the multi-reference channel: send the structure as one reference
image, the identity as another, and say in the prompt which is which. So the
axis under test is not "ControlNet vs not" but how hard the structural
reference is bound.

  T0    refs=[master]             pose in words only        - current pipeline
  N1    refs=[skeleton, master]   BFL role prompt, no LoRA  - native multi-ref
  R08   refs=[skeleton, master]   + RefControl LoRA @0.8    - README lower bound
  R10   refs=[skeleton, master]   + RefControl LoRA @1.0    - README upper bound

RefControl (thedeoxen/refcontrol-FLUX.2-klein-9B-reference-pose-lora, Apache-2.0,
rank 32) is trained on this exact base and fixes the argument order: image 1 is
the pose map, image 2 is the reference. Its trigger phrase is used verbatim.

The structural reference is a DWPose SKELETON, not a lookbook photograph. A
skeleton render carries joint positions and nothing else, so no real model's
appearance can enter the conditioning through the pose channel.
"""
from __future__ import annotations
import argparse, hashlib, json, time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "models/flux2_klein_base_9b"
LORA = ROOT / "models/flux2_loras/refcontrol_pose/refcontrol_v2_poses.safetensors"
TRIGGER = "apply pose from image 1 with reference from image 2"

SCENE = ("a plain light grey seamless studio backdrop, soft even studio lighting, eye-level "
         "camera, the full figure from head to feet inside the frame")
WARDROBE = "wearing a plain fitted tank top and plain straight trousers"
TEXTURE = "Preserve natural skin texture with visible pores and uneven tone; no airbrushing."

# T0 has no skeleton to point at, so the pose has to be carried by words. These
# sentences are my reading of the four skeleton renders, written from the
# viewer's side of the camera, and they are recorded so the wording stays auditable.
POSE_TEXT = {
    "pose01": "standing straight and facing the camera, both arms hanging down close to the sides, feet together",
    "pose02": "standing and facing the camera with both hands resting on the hips and both elbows pushed out to the sides",
    "pose03": "standing and facing the camera with one hand resting on the hip, the other arm hanging down, and the legs crossed at the ankles",
    "pose04": "standing and facing the camera with both arms hanging down and slightly away from the body, weight shifted onto one leg",
}

ROLES = {
    "T0": ("Image 1 is the identity reference for the model. Photograph that same person {pose}, "
           "{wardrobe}, on {scene}. Preserve the person's facial identity. " + TEXTURE),
    # BFL's documented wording for combining a structural reference with an identity reference.
    "N1": ("Image 1 is a pose skeleton and image 2 is the identity reference for the model. "
           "Match the exact pose from image 1 - same arm position, same body angle, same head "
           "direction. Use the person from image 2. Photograph that person {wardrobe}, on {scene}. "
           "Do not draw the skeleton itself. Preserve the person's facial identity. " + TEXTURE),
    "LORA": (TRIGGER + ". Image 1 is a pose skeleton and image 2 is the identity reference. "
             "Photograph the person from image 2 in the pose from image 1, {wardrobe}, on {scene}. "
             "Do not draw the skeleton itself. " + TEXTURE),
}
COND = {  # condition -> (role key, refs in send order, lora scale or None)
    "T0":  ("T0",   ["master"],             None),
    "N1":  ("N1",   ["skeleton", "master"], None),
    "R08": ("LORA", ["skeleton", "master"], 0.8),
    "R10": ("LORA", ["skeleton", "master"], 1.0),
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--master", type=Path, default=ROOT / "identities/MUNN26FW_A/master.png")
    ap.add_argument("--pose-refs", type=Path, default=ROOT / "data/pose_refs/munn26fw/pose_refs.json")
    ap.add_argument("--conditions", default="T0,N1,R08,R10")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1280)     # 4:5, the skeletons' aspect
    ap.add_argument("--steps", type=int, default=50)        # klein-base, undistilled
    ap.add_argument("--guidance", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=7301)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    poses = json.loads(a.pose_refs.read_text())
    conds = [c.strip() for c in a.conditions.split(",") if c.strip()]
    a.out_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for pi, p in enumerate(poses):
        for c in conds:
            role, refs, scale = COND[c]
            txt = ROLES[role].format(pose=POSE_TEXT[p["pose_id"]], wardrobe=WARDROBE, scene=SCENE)
            jobs.append({
                "name": f"{p['pose_id']}_{c}.png", "pose_id": p["pose_id"], "condition": c,
                "seed": a.seed + pi, "lora_scale": scale, "reference_roles": refs,
                "skeleton_png": p["skeleton_png"], "source_frame": p["source_frame"],
                "resolved_prompt": " ".join(txt.split()),
            })

    print(f"base   {MODEL}")
    print(f"lora   {LORA}  (exists={LORA.exists()})")
    print(f"jobs   {len(jobs)} -> {a.out_dir}")
    for c in conds:
        n = sum(1 for j in jobs if j["condition"] == c)
        ex = next(j for j in jobs if j["condition"] == c)
        print(f"  {c:<4} x{n}  refs={ex['reference_roles']}  lora={ex['lora_scale']}")
    if a.dry_run:
        (a.out_dir / "resolved_prompts.json").write_text(json.dumps(jobs, indent=2))
        print("DRY RUN - nothing loaded")
        return 0

    import torch
    from PIL import Image
    from diffusers import Flux2KleinPipeline

    if not (MODEL / "model_index.json").exists():
        raise SystemExit(f"FLUX.2 weights absent at {MODEL}")
    if any(j["lora_scale"] is not None for j in jobs) and not LORA.exists():
        raise SystemExit(f"RefControl LoRA absent at {LORA}")

    print("\nloading pipeline ...", flush=True)
    t0 = time.time()
    pipe = Flux2KleinPipeline.from_pretrained(str(MODEL), torch_dtype=torch.bfloat16).to("cuda")
    print(f"  loaded in {time.time()-t0:.0f}s", flush=True)

    master = Image.open(a.master).convert("RGB")
    skels = {p["pose_id"]: Image.open(p["skeleton_png"]).convert("RGB").resize((a.width, a.height), Image.LANCZOS)
             for p in poses}
    shas = {"master": sha256(a.master), **{p["pose_id"]: sha256(p["skeleton_png"]) for p in poses}}

    lora_loaded = False
    rows = []
    # ordered so the LoRA is attached once, after every no-LoRA condition has run
    for j in sorted(jobs, key=lambda x: (x["lora_scale"] is not None, x["lora_scale"] or 0, x["pose_id"])):
        if j["lora_scale"] is not None:
            if not lora_loaded:
                print(f"\nattaching RefControl LoRA ...", flush=True)
                pipe.load_lora_weights(str(LORA), adapter_name="refcontrol")
                lora_loaded = True
            pipe.set_adapters(["refcontrol"], [j["lora_scale"]])
        imgs = [skels[j["pose_id"]] if r == "skeleton" else master for r in j["reference_roles"]]
        sent = [j["pose_id"] if r == "skeleton" else "master" for r in j["reference_roles"]]
        g = torch.Generator(device="cuda").manual_seed(j["seed"])
        t = time.time()
        try:
            out = pipe(prompt=j["resolved_prompt"], image=imgs, width=a.width, height=a.height,
                       num_inference_steps=a.steps, guidance_scale=a.guidance, generator=g).images[0]
            out.save(a.out_dir / j["name"]); status, err = "ok", ""
        except Exception as exc:                                            # noqa: BLE001
            status, err = "failed", f"{type(exc).__name__}: {exc}"[:400]
        dt = time.time() - t
        print(f"  {j['name']:<22} {status:<7} {dt:5.1f}s  refs={sent}  lora={j['lora_scale']}", flush=True)
        rows.append({**j, "status": status, "error": err, "seconds": round(dt, 1),
                     "references_sent": sent, "reference_sha256": [shas[s] for s in sent],
                     "lora_adapter_attached": lora_loaded and j["lora_scale"] is not None})

    man = {"experiment": "flux2_pose_identity_v1",
           "created_at": datetime.now().isoformat(timespec="seconds"),
           "base_model": str(MODEL), "lora": str(LORA) if lora_loaded else None,
           "lora_sha256": sha256(LORA) if lora_loaded else None,
           "trigger_phrase": TRIGGER, "pose_text_used_only_by": "T0",
           "width": a.width, "height": a.height, "steps": a.steps, "guidance": a.guidance,
           "pose_text": POSE_TEXT, "jobs": rows}
    (a.out_dir / "manifest.json").write_text(json.dumps(man, indent=2, ensure_ascii=False))
    ok = sum(1 for r in rows if r["status"] == "ok")
    print(f"\n{ok}/{len(rows)} ok -> {a.out_dir}")
    return 0 if ok == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
