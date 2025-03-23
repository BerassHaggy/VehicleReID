import cv2

# Input and output video file paths
input_path = "/Users/martinkraus/Downloads/output_video.mp4"
output_path = "moving_vehicles.mp4"

# Open the video capture
cap = cv2.VideoCapture(input_path)

# Get the video's frames per second (fps)
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create a VideoWriter object to save the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Set the start and end times (in seconds)
start_time = 10
end_time = 15

# Calculate the frame numbers corresponding to the start and end times
start_frame = int(start_time * fps)
end_frame = int(end_time * fps)

# Loop through the frames and extract only the desired portion
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # If the current frame is within the desired time range, save it to the output
    if start_frame <= frame_count <= end_frame:
        out.write(frame)

    # Stop when the end frame is reached
    if frame_count > end_frame:
        break

    frame_count += 1

# Release everything when done
cap.release()
out.release()
cv2.destroyAllWindows()