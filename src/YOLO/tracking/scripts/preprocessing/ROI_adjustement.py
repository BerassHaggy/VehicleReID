import cv2
import matplotlib.pyplot as plt


"""
Test of ROI size in order to compare tracking predictions with
ground truths.
"""

input_path = "/Users/martinkraus/Library/CloudStorage/OneDrive-ZápadočeskáuniverzitavPlzni/Dokumenty/škola/DP/YOLO/scripts/AI CITY video/AI_CITY_video.mp4"
cap = cv2.VideoCapture(input_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

ROI = (0, height, width, 320)
x_min, y_max, x_max, y_min = ROI
frame_number = 0
# Main loop
while cap.isOpened():
    frame_number += 1
    ret, frame = cap.read()
    if frame_number == 80:
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 4)
        fig = plt.figure(figsize=(frame.shape[1] / 100, frame.shape[0] / 100), dpi=100)
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.savefig("/Users/martinkraus/Downloads/vertical_tight.png", bbox_inches='tight', pad_inches=0)
        plt.show()
        break

