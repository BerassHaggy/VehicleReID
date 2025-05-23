import matplotlib.pyplot as plt
import pandas as pd
import cv2

"""
This script reads the ground truth labels and displays
the annotations in the provided video.
"""


def showAnnotations(gt_file, video_path, output_video_path, datasetType: str, imageVisualization: bool):
    # Load the ground-truth data
    vehicle_gt_df = None
    if datasetType.startswith("MOT"):
        gt_df = pd.read_csv(gt_file, header=None,
                            names=['frame', 'object_id', 'bbox_left', 'bbox_top', 'bbox_width', 'bbox_height',
                                   'confidence',
                                   'class_id', 'visibility'])
        vehicle_gt_df = gt_df[gt_df["class_id"] == 3]  # 3 for MOT Challenge || 1 for AI CITY Challenge

    elif datasetType.startswith("AI CITY"):
        gt_df = pd.read_csv(gt_file, header=None,
                            names=[
                                'frame', 'object_id', 'bbox_left', 'bbox_top',
                                'bbox_width', 'bbox_height', 'class_id',
                                'unused1', 'unused2', 'unused3'
                            ])
        vehicle_gt_df = gt_df[gt_df["class_id"] == 1]  # 3 for MOT Challenge || 1 for AI CITY Challenge

    # Open video file
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    frame_number = 0  # Starting at 1
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1

        # Filter bounding boxes for the current frame
        frame_detections = vehicle_gt_df[vehicle_gt_df['frame'] == frame_number]

        for _, row in frame_detections.iterrows():
            x, y, w, h = int(row['bbox_left']), int(row['bbox_top']), int(row['bbox_width']), int(row['bbox_height'])
            obj_id = int(row['object_id'])
            class_id = int(row['class_id'])

            # Draw bounding box
            color = (0, 255, 0)  # Green color
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2, lineType=cv2.LINE_AA)

            # Put object ID and class label
            label = f"ID: {obj_id} Class: {class_id}"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, lineType=cv2.LINE_AA)

        # Write the frame to output video
        out.write(frame)
        if imageVisualization:
            if frame_number == 370:
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                fig = plt.figure(figsize=(image_rgb.shape[1]/100, image_rgb.shape[0]/100), dpi=100)
                plt.imshow(image_rgb)
                plt.axis("off")
                plt.savefig("/Users/martinkraus/Downloads/my_plot.png", bbox_inches='tight', pad_inches=0)
                plt.show()

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Processed video saved at: {output_video_path}")


def frameFromTrackings(video_path):
    """
    This function visualizes only the selected frame out of the video with predictions.
    :param video_path: path to video predictions
    :return:
    """
    cap = cv2.VideoCapture(video_path)
    frame_number = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1

        if frame_number == 369:
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            fig = plt.figure(figsize=(image_rgb.shape[1]/100, image_rgb.shape[0]/100), dpi=100)
            plt.imshow(image_rgb)
            plt.axis("off")
            plt.savefig("/Users/martinkraus/Downloads/my_plot.png", bbox_inches='tight', pad_inches=0)
            plt.show()

    cap.release()
    cv2.destroyAllWindows()


def main():
    """
    MOT Challenge dataset
    """
    datasetType = "AI CITY"  # AI CITY || MOT
    if datasetType.startswith("MOT"):
        gt_file = "/Users/martinkraus/Downloads/MOT17Det/train/MOT17-13/gt/gt.txt"
        video_path = "/Users/martinkraus/Library/CloudStorage/OneDrive-ZápadočeskáuniverzitavPlzni/Dokumenty/škola/DP/YOLO/scripts/MOT video/moving_vehicles.mp4"
        output_video_path = "/Users/martinkraus/Downloads/test.mp4"

        predictions_video_path = "/Users/martinkraus/Downloads/TRACKING RESULTS/MOT_final_trackings.mp4"

    elif datasetType.startswith("AI CITY"):
        gt_file = "/Users/martinkraus/Downloads/AICity22_Track1_MTMC_Tracking/train/S01/c001/gt/gt_short.txt"
        video_path = "/Users/martinkraus/Library/CloudStorage/OneDrive-ZápadočeskáuniverzitavPlzni/Dokumenty/škola/DP/YOLO/scripts/AI CITY video/AI_CITY_video.mp4"
        output_video_path = "/Users/martinkraus/Downloads/test.mp4"

        predictions_video_path = "/Users/martinkraus/Library/CloudStorage/OneDrive-ZápadočeskáuniverzitavPlzni/Dokumenty/škola/DP/YOLO/scripts/TRACKING_results/AI_CITY_final_trackings_best.mp4"

    # showAnnotations(gt_file, video_path, output_video_path, datasetType, imageVisualization=True)
    frameFromTrackings(predictions_video_path)


if __name__ == "__main__":
    main()
