import pandas as pd
import cv2

"""
This script reads the ground truth labels and displays
the annotations in the provided video.
"""


def showAnnotations(gt_file, video_path, output_video_path, datasetType: str):
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
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Put object ID and class label
            label = f"ID: {obj_id} Class: {class_id}"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Write the frame to output video
        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Processed video saved at: {output_video_path}")


def main():
    """
    MOT Challenge dataset
    """
    datasetType = "AI CITY"  # AI CITY || MOT
    if datasetType.startswith("MOT"):
        gt_file = "/Users/martinkraus/Downloads/MOT17Det/train/MOT17-13/gt/gt.txt"
        video_path = "/Users/martinkraus/Library/CloudStorage/OneDrive-ZápadočeskáuniverzitavPlzni/Dokumenty/škola/DP/YOLO/scripts/MOT_video/output_video.mp4"
        output_video_path = "/Users/martinkraus/Downloads/test.mp4"

    elif datasetType.startswith("AI CITY"):
        gt_file = "/Users/martinkraus/Downloads/AICity22_Track1_MTMC_Tracking/train/S01/c001/gt/gt.txt"
        video_path = "/Users/martinkraus/Downloads/AICity22_Track1_MTMC_Tracking/train/S01/c001/vdo.avi"
        output_video_path = "/Users/martinkraus/Downloads/test.mp4"

    showAnnotations(gt_file, video_path, output_video_path, datasetType)


if __name__ == "__main__":
    main()
