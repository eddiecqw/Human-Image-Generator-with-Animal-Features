# app.py
from flask import Flask, render_template, request, send_file
from diffusers import StableDiffusionPipeline
import torch
import os

# Initialize Flask app
app = Flask(__name__)

# Load the Stable Diffusion model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2")
pipe = pipe.to(device)

# Directory to save generated images
OUTPUT_DIR = "static/generated_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)


@app.route("/")
def index():
    """
    Render the main page with the input form.
    """
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate_image():
    """
    Handle the form submission to generate an image based on the text prompt.
    """
    # Get the user input from the form
    description = request.form.get("description", "")
    if not description:
        return "Error: Please provide a description!"

    # Generate the image with Stable Diffusion
    try:
        image = pipe(description, num_inference_steps=50).images[0]
        # Save the image to the output directory
        output_path = os.path.join(OUTPUT_DIR, f"{description}.png")
        image.save(output_path)
        return render_template("index.html", image_path=output_path, prompt=description)
    except Exception as e:
        return f"Error generating image: {e}"


if __name__ == "__main__":
    app.run(debug=True)