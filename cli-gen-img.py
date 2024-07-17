import json

import torch
import diffusers
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

# Get Config
with open("config.json") as f:
	config = json.load(f)

pipeline = StableDiffusionPipeline.from_single_file(config["img_model"], torch_dtype=torch.float16, variant="fp16", safety_checker=None).to(config["pytorch_device"])
pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
if config["img_model_lora"] != "":
	pipeline.load_lora_weights(config["img_model_lora"])

prompt = f"<lora:cartoony:1> { input('Enter the prompt: ') }"
negative_prompt = "bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry,missing fingers,bad hands,missing arms, long neck, Humpbacked, deformed, bad anatomy, disfigured, poorly drawn face, mutation, mutated, extra limb, ugly, poorly drawn hands, missing limb, floating limbs, disconnected limbs, malformed hands, out of focus, long neck, long body, monochrome, watermark, signature, logo, name"
pipeline(prompt, negative_prompt=negative_prompt, generator=torch.manual_seed(1060506489), guidance_scale=config["img_model_guidance"], num_inference_steps=config["img_model_num_steps"]).images[0].save("out/gen-img.png")