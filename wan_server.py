import modal

app = modal.App("story-video-server")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git")
    .pip_install(
        "torch",
        "diffusers==0.35.1",
        "transformers==4.51.3",
        "accelerate",
        "imageio",
        "imageio-ffmpeg",
        "fastapi",
        "uvicorn",
        "python-multipart",
        "pillow",
        "numpy",
    )
)

volume = modal.Volume.from_name("wan-weights", create_if_missing=True)

with image.imports():
    import io
    import torch
    from fastapi import FastAPI, File, Form, UploadFile
    from fastapi.responses import Response
    from PIL import Image as PILImage

app_description = "POST /render: multipart(image, prompt, negative, width, height, seconds, fps, steps, guidance, seed) -> mp4"

@app.function(
    image=image,
    gpu="T4",
    volumes={"/root/.cache/huggingface": volume},
    timeout=3600,
    scaledown_window=120,
)
@modal.web_server(8000, startup_timeout=1800)
def serve():
    fastapi_app = FastAPI()
    state = {"pipe": None}

    def get_pipe():
        if state["pipe"] is None:
            from diffusers import WanPipeline
            pipe = WanPipeline.from_pretrained(
                "Wan-AI/Wan2.2-TI2V-5B-Diffusers", torch_dtype=torch.float16
            )
            pipe.enable_model_cpu_offload()
            state["pipe"] = pipe
        return state["pipe"]

    @fastapi_app.get("/")
    def root():
        return {"ok": True, "service": "story-video-server", "model": "Wan-AI/Wan2.2-TI2V-5B-Diffusers"}

    @fastapi_app.post("/render")
    def render(
        image: UploadFile = File(...),
        prompt: str = Form(...),
        negative: str = Form("blurry, low quality, deformed face, extra limbs, text, watermark"),
        width: int = Form(480),
        height: int = Form(480),
        seconds: float = Form(5.0),
        fps: int = Form(16),
        steps: int = Form(30),
        guidance: float = Form(5.0),
        seed: int = Form(7),
    ):
        from diffusers.utils import load_image, export_to_video
        import tempfile, os

        raw = image.file.read()
        keyframe = PILImage.open(io.BytesIO(raw)).convert("RGB").resize((width, height))

        pipe = get_pipe()
        generator = torch.Generator("cuda").manual_seed(seed)
        result = pipe(
            prompt=prompt,
            negative_prompt=negative,
            image=keyframe,
            height=height,
            width=width,
            num_frames=int(seconds * fps),
            fps=fps,
            guidance_scale=guidance,
            num_inference_steps=steps,
            generator=generator,
        ).frames[0]

        out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        export_to_video(result, out.name, fps=fps)
        data = open(out.name, "rb").read()
        os.unlink(out.name)
        return Response(content=data, media_type="video/mp4")

    uvicorn.run(fastapi_app, host="0.0.0.0", port=8000)
