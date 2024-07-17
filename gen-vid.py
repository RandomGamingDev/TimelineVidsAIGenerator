import glob
import textwrap

import PIL
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import ImageClip, AudioFileClip, CompositeAudioClip, CompositeVideoClip

from timeline import *

class Color:
	WHITE = (255, 255, 255)

# Settings
vid_res = (1920, 1080)
num_vis_tabs = 3
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
	font = ImageFont.truetype("assets/fonts/Renogare.otf", size)
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
timeline_finish_time = (tab_res_w * len(timeline.events) - vid_res[0]) / speed
vid_duration = (tab_res_w * len(timeline.events) - vid_res[0] / 2) / speed
end_fade_duration = vid_duration - timeline_finish_time

# Get the music
background_music = \
	AudioFileClip("assets/music/In Dreamland.opus") \
		.set_start(0) \
		.set_duration(vid_duration) \
		.audio_fadeout(end_fade_duration)

# Get the timeline's image itself
timeline_clip = \
	ImageClip("logs/timeline.png") \
		.set_start(0) \
		.set_duration(vid_duration) \
		.set_position(lambda t: (vid_res[0] + -speed * min(t, timeline_finish_time), "center")) \
		.fadeout(end_fade_duration)
          
video = CompositeVideoClip([timeline_clip], size=vid_res)
video = video.set_audio(CompositeAudioClip([background_music]))
video.write_videofile("out/vid.mp4", audio=True, fps=24) # webm videos don't have colors???