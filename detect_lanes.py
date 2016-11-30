import click
from video_lane_detection import detect_lanes_video
from image_lane_detection import detect_lanes_image

@click.command()
@click.option('-v/-i', 
	default=False,
	prompt='Video? Enter N for image',
	help='Video or image')
@click.option('--input_file_name',
	prompt='What is the file name of the input?',
	help='Input file name')
@click.option('--output_file_name',
	prompt='What is the file name of the output?',
	help='Output file name')
def detect_lanes(v, input_file_name, output_file_name):
	if v:
		detect_lanes_video(input_file_name, output_file_name)
	else:
		detect_lanes_image(input_file_name, output_file_name)


if __name__ == '__main__':
    detect_lanes()