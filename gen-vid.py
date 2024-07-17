import glob
import textwrap

import PIL
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import VideoFileClip, ImageClip, CompositeVideoClip

from timeline import *

class Color:
	WHITE = (255, 255, 255)

# Settings
vid_res = (1920, 1080)
num_vis_tabs = 3
start_delay = 10
speed = 100

# Parse the response timeline
with open("logs/response.log") as f:
	response_txt = f.read()
timeline = Timeline(response_txt)

# Width of tab
tab_res_w = int(vid_res[0] / num_vis_tabs)
# Create the image to render to
timeline_img = PIL.Image.new(mode="RGB", size=(tab_res_w * len(timeline.events), vid_res[1]))
timeline_draw = ImageDraw.Draw(timeline_img)
# Create the font used for rendering
def resize_font(size):
	global font
	font = ImageFont.truetype("fonts/Renogare.otf", size)
font = None

# Paste the images
for f in glob.glob("logs/imgs/*"):
	event_num_start = len("logs/imgs/Event #")
	event_i = int(f[event_num_start:f.find(' ', event_num_start)]) - 1
	event = timeline.events[event_i]

	# Use smth like this to validate the txt range font.getsize(text)[0]
	# Draw the title
	resize_font(tab_res_w / 9)
	title_loc = (event_i * tab_res_w + tab_res_w / 2, 25)
	timeline_draw.text(title_loc, event.name, Color.WHITE, anchor="mt", font=font)

	# Draw the time
	time_loc = (title_loc[0], title_loc[1] + font.size)
	resize_font(font.size * 0.6)
	timeline_draw.text(time_loc, event.time, Color.WHITE, anchor="mt", font=font)

	# Draw the description
	resize_font(font.size * 0.8)
	event_description_lines = textwrap.wrap(event.description, width=vid_res[0] / (2 * font.size))
	if len(event_description_lines) > 7:
		resize_font(font.size * 7.5 / len(event_description_lines))
	for i in range(len(event_description_lines)):
		event_description_line = event_description_lines[i]
		description_loc = (title_loc[0], time_loc[1] + font.size * (2 + i))
		timeline_draw.text(description_loc, event_description_line, Color.WHITE, anchor="ma", font=font)

	# Draw the illustration image
	img_loc = (event_i * tab_res_w, vid_res[1] - tab_res_w)
	Image.Image.paste(timeline_img, Image.open(f).resize((tab_res_w, tab_res_w)), img_loc)

# Save the image
timeline_img.save("logs/timeline.png")

# Start creating the video
timeline_clip = \
	ImageClip("logs/timeline.png") \
		.set_start(0) \
		.set_duration((tab_res_w * len(timeline.events) - vid_res[0]) / speed) \
		.set_position(lambda t: (-speed * max(t - start_delay, 0), "center"))
          
final = CompositeVideoClip([timeline_clip], size=vid_res) # Remember to add audio
final.write_videofile("out/vid.mp4", audio=True, fps=24) # webm videos don't have colors???