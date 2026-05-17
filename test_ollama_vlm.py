"""
test_lmstudio_vlm.py
====================
Smoke test for the refactored vlm_association.py against LM Studio.

Usage:
    python test_lmstudio_vlm.py

Edit the IMAGE_PATH and ENRICHED_TOKENS_PATH below to point at one of
your existing pipeline outputs.
"""

import json
import sys
from pathlib import Path

from src.matching.vlm_association import VLMAssociator


# ── EDIT THESE TWO PATHS ─────────────────────────────────────────────

IMAGE_PATH = r"data\raw\1.png"

# Path to the enriched-token JSON saved by your pipeline.
# Adjust this to wherever your token-enricher output lives.
# Common locations:
#   outputs/<exp_name>/enriched_tokens/1.json
#   outputs/exp38/stage_outputs/1_enriched.json
ENRICHED_TOKENS_PATH = r"outputs\exp38\enriched_tokens\1.json"

# ─────────────────────────────────────────────────────────────────────


def main():
    image_p = Path(IMAGE_PATH)
    tokens_p = Path(ENRICHED_TOKENS_PATH)

    if not image_p.exists():
        print(f"[ERROR] Image not found: {image_p.resolve()}")
        sys.exit(1)
    if not tokens_p.exists():
        print(f"[ERROR] Enriched tokens not found: {tokens_p.resolve()}")
        print("        Update ENRICHED_TOKENS_PATH at the top of this file.")
        sys.exit(1)

    with open(tokens_p, "r", encoding="utf-8") as f:
        tokens = json.load(f)

    # If the JSON wraps tokens in a key, unwrap it.
    if isinstance(tokens, dict):
        for k in ("tokens", "enriched", "enriched_tokens", "data"):
            if k in tokens and isinstance(tokens[k], list):
                tokens = tokens[k]
                break

    if not isinstance(tokens, list):
        print(f"[ERROR] Expected a list of enriched tokens, got {type(tokens)}")
        sys.exit(1)

    print(f"Loaded {len(tokens)} enriched tokens from {tokens_p.name}")
    print(f"Calling LM Studio on image: {image_p.name}\n")

    vlm = VLMAssociator()  # uses LM Studio defaults

    result = vlm.extract(
        enriched_tokens=tokens,
        image_path=str(image_p),
        image_id=image_p.name,
    )

    vlm.print_tuples(result)
    vlm.print_diagnostics()


if __name__ == "__main__":
    main()