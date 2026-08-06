# helpers/image_generator_sdxl.py

from __future__ import annotations
import torch
from diffusers import StableDiffusionXLImg2ImgPipeline, StableDiffusionXLPipeline
from PIL import Image


class SDXLGenerator:
    def __init__(
        self,
        model_name: str = "stabilityai/stable-diffusion-xl-base-1.0",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16,
    ):
        print(f"[SDXL] Loading model: {model_name}")
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            use_safetensors=True,
        ).to(device)

    def generate(self, prompt: str, width: int = 512, height: int = 512):
        with torch.no_grad():
            out = self.pipe(prompt=prompt, width=width, height=height)
        return out.images[0]  # PIL.Image
