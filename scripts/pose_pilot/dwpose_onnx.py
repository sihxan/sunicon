#!/usr/bin/env python3
"""DWPose keypoints via onnxruntime, for scoring pose fidelity.

Wraps the reference ONNX pre/post-processing vendored in vendor/dwpose_onnx/
(IDEA-Research/DWPose, `onnx` branch) instead of re-deriving it. controlnet_aux
0.0.10 ships an mmdet/mmpose implementation of the same detector, which would
pull in the whole OpenMMLab stack; the ONNX path needs only onnxruntime.

Returns OpenPose-18 body joints normalised to [0,1] with -1 for undetected, the
same layout the OLD project's pose bank already stores, so source and generated
poses are directly comparable.
"""
from __future__ import annotations
import sys
from pathlib import Path
import cv2, numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "vendor" / "dwpose_onnx"))
WEIGHTS = ROOT / "models/dwpose_onnx"

from onnxdet import inference_detector      # noqa: E402
from onnxpose import inference_pose         # noqa: E402

# DWPose emits COCO-133; the first 17 are COCO body. OpenPose-18 inserts "neck"
# as the shoulder midpoint at index 1. This is the same remap the reference
# annotator applies.
_COCO17_TO_OP18 = [0, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]


class DWPose:
    def __init__(self, device: str = "cpu"):
        import onnxruntime as ort
        prov = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                if device == "cuda" else ["CPUExecutionProvider"])
        so = ort.SessionOptions(); so.log_severity_level = 3
        self.det = ort.InferenceSession(str(WEIGHTS / "yolox_l.onnx"), so, providers=prov)
        self.pose = ort.InferenceSession(str(WEIGHTS / "dw-ll_ucoco_384.onnx"), so, providers=prov)

    def __call__(self, img_path_or_bgr):
        img = (cv2.imread(str(img_path_or_bgr)) if isinstance(img_path_or_bgr, (str, Path))
               else img_path_or_bgr)
        if img is None:
            raise FileNotFoundError(img_path_or_bgr)
        H, W = img.shape[:2]
        boxes = inference_detector(self.det, img)
        if boxes is None or len(boxes) == 0:
            return {"n_people": 0, "body18": None, "score18": None, "image_wh": [W, H]}
        kps, scores = inference_pose(self.pose, boxes, img)     # (P,133,2), (P,133)

        people = []
        for p in range(kps.shape[0]):
            k, s = kps[p], scores[p]
            # slot 0 = nose, slot 1 = neck (shoulder midpoint), slots 2..17 = remapped COCO
            op = np.full((18, 3), -1.0, np.float32)
            op[0] = [k[0, 0] / W, k[0, 1] / H, s[0]]
            if s[5] > 0.3 and s[6] > 0.3:
                op[1] = [(k[5, 0] + k[6, 0]) / 2 / W, (k[5, 1] + k[6, 1]) / 2 / H, min(s[5], s[6])]
            for slot, coco in zip(range(2, 18), [6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]):
                op[slot] = [k[coco, 0] / W, k[coco, 1] / H, s[coco]]
            op[op[:, 2] < 0.3] = -1.0
            people.append({"body18": op, "n_visible": int((op[:, 2] > 0).sum()),
                           "area": float(np.ptp(k[:17, 0]) * np.ptp(k[:17, 1]))})
        people.sort(key=lambda x: -x["area"])
        pri = people[0]
        return {"n_people": len(people), "body18": pri["body18"][:, :2],
                "score18": pri["body18"][:, 2], "n_visible": pri["n_visible"],
                "image_wh": [W, H], "all_people": people}


def norm_pose(body18: np.ndarray) -> np.ndarray | None:
    """Hip-centred, torso-scaled coordinates: translation- and scale-invariant."""
    xy = np.asarray(body18, np.float32)[:, :2]
    need = [1, 8, 11]
    if any(xy[i][0] < 0 for i in need):
        return None
    hip = (xy[8] + xy[11]) / 2
    torso = np.linalg.norm(xy[1] - hip)
    if torso < 1e-6:
        return None
    out = (xy - hip) / torso
    out[xy[:, 0] < 0] = np.nan
    return out
