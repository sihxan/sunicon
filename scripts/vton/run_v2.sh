#!/usr/bin/env bash
# Second try-on pass. V1 only - the first run showed that adding a written
# garment description lowers garment fidelity (0.851 -> 0.822), so the budget
# goes to breadth and repeatability instead of prompt variants.
#   breadth  8 unseen looks x 2 persons              = 16
#   seeds    the 3 original looks x 2 persons x 2 new seeds = 12
set -euo pipefail
P=(outputs/pose_pilot/v1/pose04_R08.png outputs/pose_pilot/v1/pose02_R08.png)
scripts/in_container.sh envs/train/bin/python scripts/vton/generate_vton.py \
  --persons "${P[@]}" --garment-ids G4,G5,G6,G7,G8,G9,G10,G11 --strategies V1 \
  --out-dir outputs/vton/qwen2511_v2_breadth
scripts/in_container.sh envs/train/bin/python scripts/vton/generate_vton.py \
  --persons "${P[@]}" --garment-ids G1,G2,G3 --strategies V1 --seeds 8801 9902 \
  --out-dir outputs/vton/qwen2511_v2_seeds
