# Human-Image-Generator-with-Animal-Features
Human image generator using pretrained diffusion model and web framework 'flask' in Python and HTML\
## Before utilize the model there are libraries you may need to ensure you have installed:
•	__Flask__: For handling the web server and routes. \
•	__Stable Diffusion (diffusers)__: To load and run the text-to-image model. \
•	__Torch__: For running the model on GPU or CPU. \
•	__OS__ and __UUID__: For file management (saving images with unique filenames). \
•	__re__: For sanitizing user input to create safe filenames.

## Use Guidelines:
1.download the libraries that are included in the "**requirement.txt**"\
  (pip3 install -r requirement.txt)\
2.run the **app.py**\
3.open the website that are mentioned on the terminal\
4.**type the descriptions** concerns the images you want\

## Description：
```python
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
```
In this project, it used the technique of stable diffusion which is an ai model that is used for generating images in terms of text. This kind of generative model is trained to denoise an object, like an image, in order to extract a sample of interest. Until a sample is collected, the model is trained to denoise the image a little at a time. In order to produce a final image that complies with the request, it first paints the image with noise and random pixels. It then attempts to eliminate the noise by modifying each phase. 
