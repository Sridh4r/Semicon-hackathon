"""Script entrypoint for phase 1 workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from matching.template.template_matcher import match


def _load_gray_image(path: Path) -> np.ndarray:
	return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Run baseline template matching for phase 1")
	parser.add_argument("--search", default=str(PROJECT_ROOT / "sample_0001_search.png"))
	parser.add_argument("--reference", default=str(PROJECT_ROOT / "sample_0001_reference.png"))
	parser.add_argument("--reference-scale", type=float, default=1.0)
	parser.add_argument("--output-json", default=str(PROJECT_ROOT / "sample_0001_prediction.json"))
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	search_path = Path(args.search)
	reference_path = Path(args.reference)

	search = _load_gray_image(search_path)
	reference = _load_gray_image(reference_path)

	result = match(search, reference, reference_scale=args.reference_scale)

	payload = {
		"search": str(search_path),
		"reference": str(reference_path),
		"reference_scale": args.reference_scale,
		"max_score": result.max_score,
		"top_left_x": result.top_left_x,
		"top_left_y": result.top_left_y,
		"predicted_center_x": result.center_x,
		"predicted_center_y": result.center_y,
		"matched_reference_width": result.matched_reference_width,
		"matched_reference_height": result.matched_reference_height,
	}

	output_json = Path(args.output_json)
	output_json.parent.mkdir(parents=True, exist_ok=True)
	output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

	print(json.dumps(payload, indent=2))


if __name__ == "__main__":
	main()
