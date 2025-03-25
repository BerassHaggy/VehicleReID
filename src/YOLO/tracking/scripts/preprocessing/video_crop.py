import cv2

"""
This script crops an input video based on the passes startTime
and endTime.
"""

# Input and output video file paths
input_path = "/Users/martinkraus/Downloads/AICity22_Track1_MTMC_Tracking/train/S01/c001/vdo.avi"
output_path = "/Users/martinkraus/Downloads/out.mp4"

cap = cv2.VideoCapture(input_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))

# VideoWriter object to save the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Where to crop the video
start_time = 60
end_time = 120
start_frame = int(start_time * fps)
end_frame = int(end_time * fps)

# Loop through the frames
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Check the frames regarding the time boundaries
    if start_frame <= frame_count <= end_frame:
        out.write(frame)

    if frame_count > end_frame:
        # Out of boundary
        break

    frame_count += 1

cap.release()
out.release()
cv2.destroyAllWindows()
