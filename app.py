from flask import Flask, render_template, request, jsonify, url_for
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
from werkzeug.utils import secure_filename
import torch
import os
import uuid
import re
import traceback   # ✅ added for detailed error logging

app = Flask(__name__)

# Directory to save generated images
OUTPUT_DIR = os.path.join("static", "generated_images")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Persistent cache directory for the model
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Choose device
if torch.cuda.is_available():
    DEVICE = "cuda"
    DTYPE = torch.float16
else:
    DEVICE = "cpu"
    DTYPE = torch.float32

print(f"Using device: {DEVICE}")

# Model name – SDXL works well for high quality
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"

# ✅ Load pipeline with safety checker explicitly disabled
pipe = StableDiffusionXLPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    cache_dir=CACHE_DIR,
    safety_checker=None,               # disable the problematic safety checker
    requires_safety_checker=False      # tell the pipeline not to expect one
)

# Faster scheduler
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(DEVICE)

# Memory optimizations
pipe.enable_attention_slicing()
if DEVICE == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("xFormers memory efficient attention enabled.")
    except Exception:
        print("xFormers not available. Continuing without it.")

def clean_prompt_for_filename(prompt: str) -> str:
    prompt = prompt.strip().lower()
    prompt = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff_-]+", "_", prompt)
    prompt = prompt[:40].strip("_")
    if not prompt:
        prompt = "image"
    return secure_filename(prompt)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate_image():
    description = request.form.get("description", "").strip()
    if not description:
        return jsonify({"error": "Please provide a description."}), 400

    try:
        final_prompt = (
            f"{description}, high quality, detailed, cinematic lighting, "
            f"sharp focus, digital art"
        )
        negative_prompt = (
            "low quality, blurry, distorted, deformed, bad anatomy, "
            "extra limbs, extra fingers, watermark, text, logo"
        )

        # ✅ Correct generator creation
        if DEVICE == "cuda":
            generator = torch.Generator(device=DEVICE)
        else:
            generator = torch.Generator()
        generator.manual_seed(torch.seed())   # now generator is valid, manual_seed returns None but sets it

        with torch.inference_mode():
            result = pipe(
                prompt=final_prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=25,
                guidance_scale=7.5,
                width=1024,
                height=1024,
                generator=generator
            )

        image = result.images[0]

        filename_base = clean_prompt_for_filename(description)
        filename = f"{filename_base}_{uuid.uuid4().hex[:8]}.png"
        output_path = os.path.join(OUTPUT_DIR, filename)
        image.save(output_path)

        image_url = url_for("static", filename=f"generated_images/{filename}")
        return jsonify({
            "prompt": description,
            "enhanced_prompt": final_prompt,
            "image_path": image_url
        })

    except Exception as e:
        # ✅ Print full traceback for debugging
        print("=" * 50)
        print("Error generating image:")
        traceback.print_exc()
        print("=" * 50)
        return jsonify({"error": f"Error generating image: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)