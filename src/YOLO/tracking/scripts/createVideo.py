"""
    This script reconstructs a MOT video out of the single images.

"""

import configparser
import cv2
import os

# Path to the directory containing frames
source_directory = '/Users/martinkraus/Downloads/MOT17Det/train/MOT17-13'
frames_directory = source_directory + "/img1"
config = source_directory + "/seqinfo.ini"

# Read the config to get the frame rate
configParser = configparser.ConfigParser()
configParser.read(config)
frame_rate = int(configParser['Sequence']['frameRate'])

# Get the list of frame filenames
frame_files = sorted(os.listdir(frames_directory))

# Specify the video file name and codec
video_filename = 'output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can try other codecs like 'XVID' or 'MJPG'

# Set the frame size based on the first frame
first_frame = cv2.imread(os.path.join(frames_directory, frame_files[0]))
frame_size = (first_frame.shape[1], first_frame.shape[0])

# Create a VideoWriter object
video_writer = cv2.VideoWriter(video_filename, fourcc, frame_rate, frame_size)

# Iterate through each frame and add it to the video
for frame_file in frame_files:
    frame_path = os.path.join(frames_directory, frame_file)
    frame = cv2.imread(frame_path)
    
    # Write the frame to the video
    video_writer.write(frame)

# Release the VideoWriter object
video_writer.release()

print(f"Video created successfully: {video_filename}")