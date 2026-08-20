"""Stable Diffusion local image generator for parking visualisation.

Uses Hugging Face ``diffusers`` to run Stable Diffusion locally. No cloud
API is required.  Model weights are downloaded on first use and cached
under ``~/.cache/huggingface/``.

Required packages (see requirements.txt):
    diffusers, transformers, accelerate, torch, pillow

Usage::

    gen = StableDiffusionGenerator(model_id="stable-diffusion-v1-5")
    pil_image = gen.generate(occupancy, parking_map)
"""

from __future__ import annotations

import os
from typing import Optional

from .base import BaseImageGenerator


class StableDiffusionGenerator(BaseImageGenerator):
    """Generate parking-lot visualisations with a local Stable Diffusion model."""

    name = "stable-diffusion"

    def __init__(
        self,
        model_id: str = "segmind/small-sd",
        device: Optional[str] = None,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
    ) -> None:
        self.model_id = model_id        self.device = device or ("cuda" if _cuda_available() else "cpu")
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self._pipeline = None

    def _load_pipeline(self):
        if self._pipeline is not None:
            return
        try:
            from diffusers import StableDiffusionPipeline
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "diffusers/torch not installed. "
                "Install with: pip install diffusers transformers accelerate torch pillow"
            ) from exc

        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self._pipeline = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            local_files_only=False,
        )
        self._pipeline = self._pipeline.to(self.device)

    def generate(self, occupancy, parking_map) -> "PIL.Image.Image":
        """Generate a visualisation image of the parking situation."""
        self._load_pipeline()

        available = [s for s in occupancy.slots if s.status == "available"]
        occupied = [s for s in occupancy.slots if s.status == "occupied"]

        available_ids = ", ".join(s.slot_id for s in available) if available else "none"
        occupied_ids = ", ".join(s.slot_id for s in occupied) if occupied else "none"
        recommended = available[0].slot_id if available else "none"

        prompt = (
            f"A clean top-down view of a modern parking lot with marked parking slots. "
            f"Slot {recommended} is highlighted in bright green as an available spot. "
            f"Other occupied slots have cars parked in them. "
            f"Parking lot with asphalt surface and white lane markings. "
            f"Professional parking guidance system visualization, clear and labelled. "
            f"High quality, detailed, realistic bird's eye view."
        )

        negative_prompt = (
            "blurry, low quality, distorted, text, watermark, deformed, "
            "ugly, bad anatomy, extra cars in highlighted spot"
        )

        image = self._pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
        ).images[0]

        return image


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False
