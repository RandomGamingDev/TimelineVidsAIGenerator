import os
import glob
import json

import huggingface_hub
import google.generativeai as gai
import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from safetensors.torch import load_file

from timeline import *
from helpers import *

# Files that contain sensitive information (e.g. tokens) are separate
# The code cannot store sensitive information, but variables that query sensitive information from files are capitalized
# Logs of the last generation are stored for development purposes
# Prompts for each purpose are stored in files for ease of modification and reading
# Inputs that are meant to be commonly modified like the Input Prompt and Config are in the same directory as the code

# The needed tokens that go in the token directory are: gemini.token, and huggingface.token

# the "hc_" prefix for a function stands for "human checked"

# Get Config
with open("config.json") as f:
	config = json.load(f)

# Get token
with open("tokens/gemini.token") as f:
	GEMINI_API_KEY = f.read()
gai.configure(api_key=GEMINI_API_KEY)

# Get the model
model = gai.GenerativeModel(config["gpt_model"])

# Teach the AI how to generate a parsable timeline via conditioner prompt
with open("prompts/conditioner.prompt") as f:
	conditioner_prompt = f.read()
chat = model.start_chat(history=[{ "role": "user", "parts": [conditioner_prompt] }])

# Ask it to generate a timeline based on the prompt
with open("input.prompt") as f:
	input_prompt = f.read()
response = chat.send_message(input_prompt)
# Write AI's response log
log(response.text, "logs/response.log")

# Parse the response timeline
timeline = Timeline(response.text)
# Log the parsed response timeline so that common errors in parsing or the AI's output can be logged and dealt with
log(timeline.blob(), "logs/parsed-response.log")

# Log into Hugging Face
with open("tokens/huggingface.token") as f:
	HUGGINGFACE_API_KEY = f.read()
huggingface_hub.login(token=HUGGINGFACE_API_KEY)

# Initialize the local Stable Diffusion model
pipeline = StableDiffusionPipeline.from_single_file(config["img_model"], torch_dtype=torch.float16, variant="fp16", safety_checker=None).to(config["pytorch_device"])
pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
if config["img_model_lora"] != "":
	pipeline.load_lora_weights(config["img_model_lora"])

create_test_img = False
if create_test_img:
	prompt = "<lora:cartoony:1> flat color, 1girl, white sundress, straw hat, happy, picnic, sunny day, park, clouds,"
	negative_prompt = "bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry,missing fingers,bad hands,missing arms, long neck, Humpbacked, deformed, bad anatomy, disfigured, poorly drawn face, mutation, mutated, extra limb, ugly, poorly drawn hands, missing limb, floating limbs, disconnected limbs, malformed hands, out of focus, long neck, long body, monochrome, watermark, signature, logo, name"
	pipeline(prompt, negative_prompt=negative_prompt, generator=torch.manual_seed(1060506489), guidance_scale=config["img_model_guidance"], num_inference_steps=config["img_model_num_steps"]).images[0].save("logs/test-img.png")

os.makedirs("logs/imgs/", exist_ok=True)
for f in glob.glob("logs/imgs/*"):
	os.remove(f)
events_imgs = timeline.hc_gen_events_imgs(guidance_scale=config["img_model_guidance"], num_inference_steps=config["img_model_num_steps"], batch_size=1, pipeline=pipeline)
for i in range(len(events_imgs)):
	event_imgs = events_imgs[i]
	for j in range(len(event_imgs)):
		event_img = event_imgs[j]
		event_img.save(f"logs/imgs/Event #{ i + 1 } Image #{ j }.png")