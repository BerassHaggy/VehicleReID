import cv2
import pandas as pd
import random
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import datetime
import mot_evaluator as mt
import wandb
from TrafficStatistics import TrafficStatistics

"""
    This script performs a vehicles detection and tracking including statistics.

"""


class deepSort:
    def __init__(self, model, input_video, output_path, result_output, tracking_ground_truth, mot_results,
                 datasetType: str, includeROI: bool, visualizeROI: bool) -> None:
        self.model = model
        self.input_video = input_video
        self.output_path = output_path
        self.result_output = result_output
        self.tracking_ground_truth = tracking_ground_truth
        self.mot_results = mot_results
        self.tracking_results = list()
        self.color_map = dict()
        self.datasetType = datasetType
        self.includeROI = includeROI
        self.visualizeROI = visualizeROI
        self.ROI = mt.MOTEvaluator.ROI

    def get_random_color(self):
        return random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)

    def track_vehicles(self, visible: bool, mot_challenge: bool, write_video: bool, datasetType: str):
        # Hyperparameters based on a wandb sweep
        if datasetType.startswith("AI CITY") or datasetType.startswith("Pilsen"):
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

        trafficStatistics = TrafficStatistics()

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

                # Process the trackID and classID in order to make traffic statistics
                trafficStatistics.countCarsPerClass(trackID=track_id, classID=class_id)

                # Save the occurrence values per track
                trafficStatistics.trackID_to_class[track_id].append(trafficStatistics.class_names[class_id])
                trafficStatistics.occuranceDuration(track_id, trafficStatistics.class_names[class_id], current_minute,
                                                    frame_number, current_time)

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

        if mot_challenge:
            # Convert tracking results to DataFrame in order to evaluate the MOT metrics
            columns = ['frame', 'object_id', 'bbox_left', 'bbox_top', 'bbox_width', 'bbox_height', 'confidence',
                       'class', 'visibility']
            tracking_df = pd.DataFrame(self.tracking_results, columns=columns)
            # Save the results to a .txt file
            tracking_df.to_csv(self.result_output, index=False, header=False)

        # Process the traffic statistics
        # trafficStatistics.displayStatistics()
        # trafficStatistics.processOccuranceDuration()

        # After processing the video, calculate the MOTA metric
        motaEvaluator = mt.MOTEvaluator(ground_truth_labels=self.tracking_ground_truth,
                                        predictions_filename=self.result_output,
                                        results_filename=self.mot_results)
        results, acc = motaEvaluator.evaluate(datasetType=self.datasetType, includeROI=self.includeROI)

        # Access the computed metrics
        mota_value = results.mota["summary"]
        motp_value = results.motp["summary"]
        idf1 = results.idf1["summary"]
        fp = results.num_false_positives["summary"]  # False positives
        fn = results.num_misses["summary"]  # False negatives
        num_gt = results.num_objects["summary"]  # Number of unique objects
        precision = results.precision["summary"]
        recall = results.recall["summary"]
        num_detections = results.num_detections["summary"]  # Number of detected objects
        num_matches = results.num_matches["summary"]  # Number of matches

        # MODA, MODP
        MODA = 1 - (fp + fn) / num_gt
        # MODP = num_matches / (num_detections - fp)

        # F1-score (by definition) for comparison with IDF1
        f1_score = 2 * (precision * recall) / (precision + recall)

        print("--------")
        print("MODA: " + str(MODA))
        # print("MODP: " + str(MODP))
        print("F1-score: " + str(f1_score))
        print("IDF1-score: " + str(idf1))
        print("--------")

        # Log the metric to wandb
        wandb.log({"MOTA": mota_value,
                   "MOTP": motp_value,
                   "MODA": MODA,
                   "IDF1": idf1,
                   "Precision": precision,
                   "Recall": recall,
                   "F1-score": f1_score,
                   "FP": fp,
                   "FN": fn})


# Main logic
def main():
    """
        Local configuration
    """
    """
    # Load the pretrained YOLOv8 model
    model = YOLO("/Users/martinkraus/Downloads/YOLO_tuned_models/custom_vehicles.pt")
    # Define the input video
    input_video = "/Users/martinkraus/Downloads/out.mp4"
    # Define the path for the resulting video with tracking
    output_path_deepSort = "/Users/martinkraus/Downloads/deepSORT_vehicles.mp4"
    # Define the tracking results path
    results_output_filename = "/Users/martinkraus/Library/CloudStorage/OneDrive-ZápadočeskáuniverzitavPlzni/Dokumenty/škola/DP/YOLO/scripts/MOT_challenge/deepSORT.txt"
    # Define tracking ground truth
    tracking_ground_truth = "/Users/martinkraus/Downloads/AICity22_Track1_MTMC_Tracking/train/S01/c001/gt/gt_short.txt"
    # Define path for MOT results
    mot_results = "mot_results.txt"
    datasetType = "AI CITY"
    """
    """
        Metacentrum configuration - MOT Challenge
    """

    """
    wandb.init(project="TRACKING_deepSORT", entity="krausm00")
    # Load the pretrained YOLOv8 model
    model = YOLO("../data/custom_vehicles.pt")
    # Define the input video
    input_video = "../data/moving_vehicles.mp4"
    # Define the path for the resulting video with tracking
    output_path_deepSort = "../video/MOT_final_trackings.mp4"
    # Define the tracking results path
    results_output_filename = "../results/MOT_tracking_results.txt"
    # Define a path for MOT results
    mot_results = "../results/mot_results.txt"
    # Tracking ground truths
    tracking_ground_truth = "../ground_truths/gt.txt"
    datasetType = "MOT"

    deep_sort = deepSort(model, input_video=input_video, output_path=output_path_deepSort,
                         result_output=results_output_filename,
                         tracking_ground_truth=tracking_ground_truth, mot_results=mot_results, datasetType=datasetType,
                         includeROI=False, visualizeROI=False)
    deep_sort.track_vehicles(visible=False, mot_challenge=True, write_video=True, datasetType=datasetType)
    """
    """
        Metacentrum configuration - AICITY Challenge
    """
    """
    wandb.init(project="TRACKING_deepSORT", entity="krausm00")
    # Load the pretrained YOLOv8 model
    model = YOLO("../data/custom_vehicles.pt")
    # Define the input video
    input_video = "../AICITY/data/AI_CITY_video.mp4"
    # Define the path for the resulting video with tracking
    output_path_deepSort = "../AICITY/video/AI_CITY_final_trackings.mp4"
    # Define the tracking results path
    results_output_filename = "../AICITY/results/AI_CITY_tracking_results.txt"
    # Define a path for MOT results
    mot_results = "../AICITY/results/aicity_results.txt"
    # Tracking ground truths
    tracking_ground_truth = "../AICITY/ground_truths/gt_short.txt"
    datasetType = "AI CITY"
    """

    """
            Metacentrum configuration - Pilsen dataset
        """

    wandb.init(project="TRACKING_deepSORT", entity="krausm00")
    # Load the pretrained YOLOv8 model
    model = YOLO("../data/custom_vehicles.pt")
    # Define the input video
    input_video = "../PILSEN/data/pilsen_short.mp4"
    # Define the path for the resulting video with tracking
    output_path_deepSort = "../PILSEN/video/pilsen_final_trackings.mp4"
    # Define the tracking results path
    results_output_filename = "../PILSEN/results/pilsen_tracking_results.txt"
    # Define a path for MOT results
    mot_results = "../PILSEN/results/pilsen_results.txt"
    # Tracking ground truths
    tracking_ground_truth = "../PILSEN/ground_truths/pilsen_gt.txt"
    datasetType = "Pilsen"
    deep_sort = deepSort(model, input_video=input_video, output_path=output_path_deepSort,
                         result_output=results_output_filename,
                         tracking_ground_truth=tracking_ground_truth, mot_results=mot_results, datasetType=datasetType,
                         includeROI=False, visualizeROI=False)
    deep_sort.track_vehicles(visible=False, mot_challenge=True, write_video=False, datasetType=datasetType)


# Run the script
if __name__ == "__main__":
    print("ole")
    main()
