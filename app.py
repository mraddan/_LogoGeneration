import torch
import json
import os
import base64

from io import BytesIO
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler, PNDMScheduler
from diffusers import StableDiffusionPipeline
from flask import Flask, flash, redirect, render_template, request, Request, jsonify

app = Flask(__name__, template_folder='templates')

model1 = 'logo-wizard/logo-diffusion-checkpoint'
model2 = "stabilityai/stable-diffusion-xl-base-1.0"
model3 = "stabilityai/stable-diffusion-xl-refiner-1.0"

sch = PNDMScheduler.from_pretrained(
    model1, 
    subfolder='scheduler'
    )
sch1 = EulerAncestralDiscreteScheduler.from_pretrained(
    model2, 
    subfolder='scheduler'
    )
sch2 = EulerAncestralDiscreteScheduler.from_pretrained(
    model3, 
    subfolder='scheduler'
    )

pipe = StableDiffusionPipeline.from_pretrained(
    model1, 
    torch_dtype=torch.float16, 
    variant="fp16", 
    scheduler=sch
    )
base = DiffusionPipeline.from_pretrained(
    model2, 
    torch_dtype=torch.float16, 
    variant="fp16",
    scheduler=sch1
    )
refiner = DiffusionPipeline.from_pretrained(
    model3,
    text_encoder_2=base.text_encoder_2,
    vae=base.vae,
    torch_dtype=torch.float16,
    scheduler=sch2,
    use_safetensors=True,
    variant="fp16"
)

pipe.enable_model_cpu_offload()
base.enable_model_cpu_offload()
refiner.enable_model_cpu_offload()
pipe.enable_xformers_memory_efficient_attention() 
base.enable_xformers_memory_efficient_attention() 
refiner.enable_xformers_memory_efficient_attention() 

np = "low quality, worst quality, bad composition, extra digit, fewer digits, inscription, asymmetric, ugly, tiling, out of frame, disfigured, deformed, body out of frame, watermark, signature, cut off, low contrast, underexposed, overexposed, bad art, beginner, amateur"
#p = "minimalistic, game logo, lion, white and blue, bold"
n_steps = 40
n_steps2 = 15

height = 768
width = 768

image_counter = 0

@app.route('/', methods=['GET', 'POST'])
def generate_logo():
    global image_counter
    if request.method == 'POST':
        prompt = request.form.get("prompt")
        style = request.form.get("style")

        image = pipe(    
            prompt=prompt+style,
            negative_prompt=np,
            num_inference_steps=n_steps,
            height=height,
            width=width,
            guidance_scale=7.5,
            #denoising_end=high_noise_frac,
            output_type="latent"
        ).images
        image = base(
            prompt=prompt+style,
            negative_prompt=np,
            num_inference_steps=n_steps2,
            height=height,
            width=width,
            #denoising_end=high_noise_frac,
            output_type="latent"
        ).images
        image = refiner(
            prompt=prompt+style,
            negative_prompt=np,
            num_inference_steps=n_steps,
            #denoising_start=high_noise_frac,
            image=image
        ).images[0]

        filename = f"generated_image-{image_counter}.jpg"

        image_counter += 1

        # Save the image with the generated filename
        image.save(os.path.join("./result", filename))

        # Save the image to a BytesIO buffer
        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        buffer.seek(0)

        # Convert the image to a base64-encoded string
        image_data = base64.b64encode(buffer.getvalue()).decode()

        # Return the HTML with the base64-encoded image
        return render_template("home.html", image_data=image_data)
        # return jsonify({"image_base_64": image_data}), 200
    else:
        return render_template("home.html")
    
@app.route('/static/style.json')
def get_style_json():
    with open('style.json') as f:
        style_data = json.load(f)
    return json.dumps(style_data)

if __name__ == "__main__":
    app.run(debug=True)