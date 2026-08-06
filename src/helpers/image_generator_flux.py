import torch
from diffusers import FluxPipeline

class FluxGenerator:
    def __init__(
        self,
        model_name: str = "black-forest-labs/FLUX.1-schnell",
        device: str = "cuda", 
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        print(f"[FLUX] Loading model: {model_name}")
        # VRAM이 타이트할 경우 .from_pretrained(..., load_in_4bit=True) 고려
        self.pipe = FluxPipeline.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            revision="main",
            load_in_4bit=True
        ).to(device)
        
        # CPU 오프로딩을 사용하면 VRAM을 극도로 아낄 수 있음 (속도는 느려짐)
        # self.pipe.enable_model_cpu_offload()

    def generate(self, prompt: str, width: int = 1024, height: int = 1024):
        # schnell 모델은 4 steps만으로 충분합니다.
        with torch.no_grad():
            out = self.pipe(
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=4, 
                guidance_scale=0.0, # schnell 모델은 0.0 권장
            )
        return out.images[0]