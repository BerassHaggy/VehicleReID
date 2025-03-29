import cv2
import matplotlib.pyplot as plt


"""
Test of ROI size in order to compare tracking predictions with
ground truths.
"""

input_path = "/Users/martinkraus/Downloads/AICity22_Track1_MTMC_Tracking/train/S01/c001/vdo.avi"
cap = cv2.VideoCapture(input_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

ROI = (0, height, width, 320)
x_min, y_max, x_max, y_min = ROI

# Main loop
while cap.isOpened():
    ret, frame = cap.read()

    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.title("ROI (Blue), Ground Truth (Green), Predictions (Red)")
    plt.axis("off")
    plt.show()
    break

