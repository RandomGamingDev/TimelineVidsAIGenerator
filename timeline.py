import torch
from diffusers import PNDMScheduler, StableDiffusionPipeline

# Create the classes for managing and parsing the AI's response
class Event:
	def __init__(self, event_str: str):
		name_start = event_str.find('.') + 1
		time_start = event_str.find('(', name_start) + 1
		name_end = time_start - 1
		time_end = event_str.find(')', time_start)
		img_prompt_start = event_str.find('(', time_end + 1) + 1
		img_prompt_end = event_str.find(')', img_prompt_start)
		description_start = event_str.find(':', img_prompt_end) + 1

		self.name = event_str[name_start:name_end].strip()
		self.time = event_str[time_start:time_end].strip()
		self.img_prompt = event_str[img_prompt_start:img_prompt_end].strip().replace('[', '<').replace(']', '>')
		self.description = event_str[description_start:].strip()

	def gen_imgs(self, batch_size: int, pipeline: StableDiffusionPipeline):
		with torch.autocast("cuda"):
			negative_prompt = "bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry,missing fingers,bad hands,missing arms, long neck, Humpbacked, deformed, bad anatomy, disfigured, poorly drawn face, mutation, mutated, extra limb, ugly, poorly drawn hands, missing limb, floating limbs, disconnected limbs, malformed hands, out of focus, long neck, long body, monochrome, watermark, signature, logo, name"
			#negative_prompt = "easynegative ,   By bad artist -neg,   symbol,   (text),   logo,   (signature),   badhandsv5-neg"
			#pipeline([", AGGA_ST010,  1girl,  cute girl, <lora:EMS-50853-EMS:0.800000>"] * batch_size, negative_prompt=[negative_prompt] * batch_size, guidance_scale=7.5, num_inference_steps=50).images[0].save("test.png")
			return pipeline([self.img_prompt] * batch_size, negative_prompt=[negative_prompt] * batch_size, guidance_scale=7.5, num_inference_steps=50).images

	def __str__(self) -> str:
		return f"<Event>{ self.name } ({ self.time }) ({ self.img_prompt }): { self.description }</Event>"

class Timeline:
	def __init__(self, timeline_str: str):
		self.events = []
		for line in timeline_str.split('\n'):
			lstrip_line = line.lstrip()
			list_num_end = lstrip_line.find('.')

			if not lstrip_line[:list_num_end].isdigit():
				continue

			self.events.append(Event(lstrip_line))

	def gen_events_imgs(self, batch_size: int, pipeline: StableDiffusionPipeline):
		return [event.gen_imgs(batch_size=batch_size, pipeline=pipeline) for event in self.events]

	def blob(self) -> str:
		return [f"{ i + 1 }. { str(self.events[i]) }\n" for i in range(len(self.events))]

	def __str__(self) -> str:
		return str([str(event) for event in self.events])