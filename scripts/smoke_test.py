"""GPU smoke test for one mode or the complete public API."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from inpaint_core import BoxPrompt, ImageProcessor, OutpaintMargins, PointPrompt, UpscaleOptions, preprocess_image, preprocess_image_only


MODES = (
    "segment", "remove", "replace", "background", "add-prompt",
    "add-reference", "prompt-edit", "generate", "outpaint", "upscale",
)


def save(array, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)
    print(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=(*MODES, "all"), default="segment")
    parser.add_argument("--image", type=Path, default=Path("assets/cat.jpg"))
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--output", type=Path, default=Path("outputs/smoke"))
    args = parser.parse_args()

    processor = ImageProcessor.from_config(args.config)
    raw = preprocess_image_only(args.image)
    selection = PointPrompt([(raw.shape[1] / 2, raw.shape[0] / 2)], [1])
    # The same smoke-test image is reused by the stricter OmniPaint modes.
    input_limit = min(
        processor.config.processing.max_image_side,
        int(processor.config.omnipaint.options["max_image_side"]),
    )
    image, selection = preprocess_image(raw, selection, max_side=input_limit)
    segmentation = processor.segment(image, selection)
    mask = segmentation.best_mask
    modes = MODES if args.mode == "all" else (args.mode,)

    for mode in modes:
        if mode == "segment":
            save(mask, args.output / "segment-mask.png")
        elif mode == "remove":
            save(processor.remove_object(image, mask=mask).image, args.output / "remove.png")
        elif mode == "replace":
            save(processor.replace_object(image, "a golden retriever", mask=mask).image, args.output / "replace.png")
        elif mode == "background":
            save(processor.replace_background(image, "a tropical beach").image, args.output / "background.png")
        elif mode == "add-prompt":
            box = BoxPrompt(32, 32, image.shape[1] // 3, image.shape[0] // 3)
            save(processor.add_object_by_prompt(image, "a red balloon", placement=box).image, args.output / "add-prompt.png")
        elif mode == "add-reference":
            if args.reference is None:
                print("skip add-reference: pass --reference PATH")
                continue
            reference = preprocess_image_only(args.reference, max_side=512)
            box = BoxPrompt(32, 32, image.shape[1] // 3, image.shape[0] // 3)
            save(processor.add_object_by_reference(image, reference, placement=box).image, args.output / "add-reference.png")
        elif mode == "prompt-edit":
            save(processor.prompt_edit(image, "Turn the scene into winter").image, args.output / "prompt-edit.png")
        elif mode == "generate":
            save(processor.generate_image("A cinematic mountain lake", width=768, height=768).image, args.output / "generate.png")
        elif mode == "outpaint":
            save(processor.outpaint(image, "Continue the scene naturally", OutpaintMargins(left=128, right=128)).image, args.output / "outpaint.png")
        elif mode == "upscale":
            save(processor.upscale(image, options=UpscaleOptions(scale=2, tile=512)).image, args.output / "upscale.png")

    processor.save_memory_stats(args.output / "memory.json")
    processor.release_models()


if __name__ == "__main__":
    main()
