import cv2
import pandas as pd
import random
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import datetime


"""
    This script performs a vehicles detection and tracking including statistics.

"""


class deepSort:
    def __init__(self, model, input_video, output_path,
                 datasetType: str, includeROI: bool, visualizeROI: bool) -> None:
        self.model = model
        self.input_video = input_video
        self.output_path = output_path
        self.tracking_results = list()
        self.color_map = dict()
        self.datasetType = datasetType
        self.includeROI = includeROI
        self.visualizeROI = visualizeROI

    def get_random_color(self):
        return random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)

    def track_vehicles(self, visible: bool, mot_challenge: bool, write_video: bool, datasetType: str):
        # Hyperparameters based on a wandb sweep
        if datasetType.startswith("AI CITY"):
            tracker = DeepSort(
                max_age=32,
                n_init=5,
                nn_budget=159,
                max_cosine_distance=0.39552185947348695
            )
        elif datasetType.startswith("MOT"):
            tracker = DeepSort(
                max_age=38,
                n_init=1,
                nn_budget=112,
                max_cosine_distance=0.1883443806825028
            )

        cap = cv2.VideoCapture(self.input_video)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore
        if write_video:
            out = cv2.VideoWriter(self.output_path, fourcc, fps, (frame_width, frame_height))

        frame_number = 0
        while cap.isOpened():

            startTime = datetime.datetime.now()
            success, frame = cap.read()
            if not success:
                break

            frame_number += 1

            # Perform detection with YOLOv8
            results = self.model(frame)

            # Extract detection results
            detections = results[0].boxes.data.tolist()

            # Loop through all detections
            dets = []
            for det in detections:
                x1, y1, x2, y2, conf, cls = det

                # Modify the detection format for the deepSORT compatibility
                # Convert to [x1, y1, width, height]
                width = x2 - x1
                height = y2 - y1
                detection_bbox = [x1, y1, width, height]

                # Append the detection in the required format for DeepSort: (bbox, confidence, class)
                dets.append((detection_bbox, conf, int(cls)))

            # Update tracker
            tracks = tracker.update_tracks(dets, frame=frame)

            # Get the time information within the video
            current_time = frame_number / fps
            current_minute = int(current_time // 60)

            # Store tracking results
            for track in tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                track_id = track.track_id

                bbox = track.to_ltrb()  # get the bounding box values for the visualization
                x1, y1, x2, y2 = bbox

                confidence = track.det_conf
                class_id = track.det_class
                visibility = 1

                # Draw results only for cars and trucks
                if class_id == 0 or class_id == 2:
                    # Draw the results
                    if track_id not in self.color_map:
                        self.color_map[track_id] = self.get_random_color()
                    color = self.color_map[track_id]  # Get the assigned color

                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cv2.putText(frame, f'ID: {track_id}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    cv2.putText(frame, f'Class: {self.model.names[class_id]}', (int(x1), int(y1) - 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    try:
                        cv2.putText(frame, f'Conf: {confidence:.2f}', (int(x1), int(y1) - 40), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, color, 2)
                        print("Frame number: " + frame_number + " is being processed.")
                    except:
                        pass

                # Decide whether to display the annotated frame
                if visible:
                    # Display the annotated frame
                    cv2.imshow("YOLOv8 Tracking", frame)

                    # Break the loop if 'q' is pressed
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                # Transform the important values for the MOT metrics
                w = x2 - x1
                h = y2 - y1
                if confidence is None:
                    confidence = 0
                self.tracking_results.append(
                    [frame_number, track_id, int(x1), int(y1), int(w), int(h), confidence, class_id, visibility])

            # Write the frame to the output video
            if write_video:
                out.write(frame)

        cap.release()


# Main logic
def main():
    """
        Gradio configuration
    """

    # Load the pretrained YOLOv8 model
    model = YOLO("../custom_vehicles.pt")
    # Define the input video
    input_video = "../AICITY/data/AI_CITY_video.mp4"
    # Define the path for the resulting video with tracking
    output_path_deepSort = "../AICITY/video/AI_CITY_final_trackings.mp4"
    datasetType = "AI CITY"

    deep_sort = deepSort(model, input_video=input_video, output_path=output_path_deepSort
                         , datasetType=datasetType, includeROI=False, visualizeROI=True)
    deep_sort.track_vehicles(visible=False, mot_challenge=True, write_video=True, datasetType=datasetType)


# Run the script
if __name__ == "__main__":
    print("ole")
    main()
