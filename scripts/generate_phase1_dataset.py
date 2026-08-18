"""Script entrypoint for phase 1 dataset generation."""

from pathlib import Path
import argparse
import json
import sys

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from generator.dram.cell_model import DRAMConfig
from generator.dram.dram_generator import DRAMGenerator
from preprocessing.preprocessing_pipeline import ExperimentParams, run_experiment


def _parse_csv_ints(text: str) -> list[int]:
	values: list[int] = []
	for token in text.split(","):
		t = token.strip()
		if not t:
			continue
		values.append(int(t))
	if not values:
		raise ValueError("Expected at least one integer value.")
	return values


def _parse_noise_seed_map(text: str) -> dict[int, int]:
	mapping: dict[int, int] = {}
	if not text.strip():
		return mapping
	for pair in text.split(","):
		item = pair.strip()
		if not item:
			continue
		if ":" not in item:
			raise ValueError(f"Invalid noise seed mapping: {item}")
		left, right = item.split(":", 1)
		sample_id = int(left.strip())
		seed = int(right.strip())
		mapping[sample_id] = seed
	return mapping


def _location_seed(base_seed: int | None, sample_id: int) -> int | None:
	if base_seed is None:
		return None
	return base_seed * 100003 + sample_id * 7907


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Generate DRAM samples with controlled noise/blur/contrast experiments"
	)
	parser.add_argument("--output-dir", default=str(PROJECT_ROOT))
	parser.add_argument("--sample-id", type=int, default=1)
	parser.add_argument("--sample-ids", default="")
	parser.add_argument("--target-size", type=int, default=65)
	parser.add_argument("--reference-downscale-factor", type=float, default=10.0)
	parser.add_argument("--seed", type=int, default=None)
	parser.add_argument("--experiments", default="A,B,C,D,E")
	parser.add_argument("--gaussian-sigma", type=float, default=10.0)
	parser.add_argument("--blur-radius", type=float, default=1.2)
	parser.add_argument("--contrast-jitter", type=float, default=0.25)
	parser.add_argument(
		"--noise-seed-map",
		default="1:101,2:284,3:732",
		help="CSV of sample_id:noise_seed mappings, e.g. 1:101,2:284,3:732",
	)
	parser.add_argument("--image-width", type=int, default=512)
	parser.add_argument("--image-height", type=int, default=512)
	parser.add_argument("--rows", type=int, default=16)
	parser.add_argument("--columns", type=int, default=16)
	parser.add_argument("--row-pitch", type=int, default=28)
	parser.add_argument("--column-pitch", type=int, default=28)
	parser.add_argument("--wordline-thickness", type=int, default=3)
	parser.add_argument("--bitline-thickness", type=int, default=3)
	parser.add_argument("--contact-size", type=int, default=7)
	return parser.parse_args()


def _crop_and_resize_reference(
	search_image,
	target_x: int,
	target_y: int,
	target_size: int,
	reference_downscale_factor: float,
):
	half = target_size // 2
	x0 = target_x - half
	y0 = target_y - half
	x1 = x0 + target_size
	y1 = y0 + target_size
	target_crop = search_image[y0:y1, x0:x1]
	reference_size = max(1, int(round(target_size / reference_downscale_factor)))
	reference = Image.fromarray(target_crop, mode="L").resize(
		(reference_size, reference_size),
		resample=Image.Resampling.BILINEAR,
	)
	return target_crop, reference, reference_size


def _save_sample(
	output_dir: Path,
	sample_id: int,
	experiment: str,
	search_image,
	reference_image: Image.Image,
	metadata: dict,
) -> tuple[Path, Path, Path]:
	sample_name = f"sample_{sample_id:04d}"
	suffix = f"_{experiment}"
	search_path = output_dir / f"{sample_name}{suffix}_search.png"
	reference_path = output_dir / f"{sample_name}{suffix}_reference.png"
	json_path = output_dir / f"{sample_name}{suffix}.json"

	Image.fromarray(search_image, mode="L").save(search_path)
	reference_image.save(reference_path)
	json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

	if experiment.upper() == "A":
		legacy_search = output_dir / f"{sample_name}_search.png"
		legacy_reference = output_dir / f"{sample_name}_reference.png"
		legacy_json = output_dir / f"{sample_name}.json"
		Image.fromarray(search_image, mode="L").save(legacy_search)
		reference_image.save(legacy_reference)
		legacy_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

	return search_path, reference_path, json_path


def main() -> None:
	args = parse_args()
	if args.sample_id <= 0:
		raise ValueError("sample-id must be >= 1")
	if args.target_size % 2 == 0:
		raise ValueError("target-size must be odd so target_x/target_y are true center pixels")

	experiments = [e.strip().upper() for e in args.experiments.split(",") if e.strip()]
	if not experiments:
		raise ValueError("At least one experiment is required")

	sample_ids = _parse_csv_ints(args.sample_ids) if args.sample_ids.strip() else [args.sample_id]
	noise_seed_map = _parse_noise_seed_map(args.noise_seed_map)

	for sample_id in sample_ids:
		if sample_id <= 0:
			raise ValueError("all sample ids must be >= 1")

	config = DRAMConfig(
		image_width=args.image_width,
		image_height=args.image_height,
		rows=args.rows,
		columns=args.columns,
		row_pitch=args.row_pitch,
		column_pitch=args.column_pitch,
		wordline_thickness=args.wordline_thickness,
		bitline_thickness=args.bitline_thickness,
		contact_size=args.contact_size,
	)
	params = ExperimentParams(
		gaussian_sigma=args.gaussian_sigma,
		blur_radius=args.blur_radius,
		contrast_jitter=args.contrast_jitter,
	)
	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	generator = DRAMGenerator(config)

	for sample_id in sample_ids:
		location_seed = _location_seed(args.seed, sample_id)
		target_x, target_y = generator.choose_target_center(
			target_size=args.target_size,
			seed=location_seed,
		)
		clean = generator.build_clean_image()
		noise_seed = noise_seed_map.get(sample_id)

		for experiment in experiments:
			search_image, exp_meta = run_experiment(
				clean_image=clean,
				experiment=experiment,
				sample_id=sample_id,
				noise_seed=noise_seed,
				base_seed=args.seed,
				params=params,
			)
			_, reference_image, reference_size = _crop_and_resize_reference(
				search_image=search_image,
				target_x=target_x,
				target_y=target_y,
				target_size=args.target_size,
				reference_downscale_factor=args.reference_downscale_factor,
			)

			metadata = {
				"sample_id": sample_id,
				"experiment": experiment,
				"target_x": target_x,
				"target_y": target_y,
				"target_size": args.target_size,
				"reference_size": reference_size,
				"reference_downscale_factor": args.reference_downscale_factor,
			}
			if args.seed is not None:
				metadata["seed"] = args.seed
			if location_seed is not None:
				metadata["location_seed"] = location_seed
			metadata.update(exp_meta)

			search_path, reference_path, target_json_path = _save_sample(
				output_dir=output_dir,
				sample_id=sample_id,
				experiment=experiment,
				search_image=search_image,
				reference_image=reference_image,
				metadata=metadata,
			)

			print(f"Generated search image: {search_path}")
			print(f"Generated reference image: {reference_path}")
			print(f"Saved metadata JSON: {target_json_path}")
			print(metadata)


if __name__ == "__main__":
	main()
