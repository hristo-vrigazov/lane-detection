from utils import lane_detection_pipeline
from cv2 import imread
from cv2 import imwrite

def detect_lanes_image(input_file_name, output_file_name):
	print("Detecting lanes in image {0} and writing to {1}".format(input_file_name, output_file_name))
	input_image = imread(input_file_name)
	output_image = lane_detection_pipeline(input_image)
	imwrite(output_file_name, output_image)