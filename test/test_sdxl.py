import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL

def test_sdxl():
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    vae_id = "madebyollin/sdxl-vae-fp16-fix"  # fp16 VAE fix

    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(
        vae_id,
        torch_dtype=torch.float16
    )

    print("Loading SDXL pipeline...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        vae=vae,
        use_safetensors=True,
        variant="fp16"
    ).to("cuda")

    prompt = (
        "a minimal black and white diagram of a multi-agent AI system generating "
        "LaTeX documents, simple, clean, no text"
    )

    print("Generating image...")
    image = pipe(
        prompt,
        num_inference_steps=30,
        guidance_scale=7.0,
    ).images[0]

    out_path = "test/sdxl_test.png"
    image.save(out_path)
    print(f"Saved image to {out_path}")

if __name__ == "__main__":
    print("PyTorch CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU count:", torch.cuda.device_count())
        print("Current GPU:", torch.cuda.get_device_name(0))
    test_sdxl()
