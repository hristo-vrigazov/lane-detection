from moviepy.editor import VideoFileClip
from utils import lane_detection_pipeline

def detect_lanes_video(input_file_name, output_file_name):
	print("Detecting lanes in video {0} and writing to {1}".format(input_file_name, output_file_name))
	input_clip = VideoFileClip(input_file_name)
	output_clip = input_clip.fl_image(lane_detection_pipeline)
	output_clip.write_videofile(output_file_name, audio=False)