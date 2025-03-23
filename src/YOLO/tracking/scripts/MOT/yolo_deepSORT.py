from collections import defaultdict
import cv2
import pandas as pd
import random
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import datetime
import matplotlib.pyplot as plt
import mot_evaluator as mt
#import mot_evaluator_aicity as mtaicity
import wandb
import inspect


"""
    This script performs a vehicles detection and tracking including statstics.

"""


class TrafficStatistics():
    def __init__(self) -> None:
       self.class_trackIDs = dict()
       self.class_counts = dict()
       self.class_names = {0: 'car', 1: 'motorbike', 2: 'truck', 3: 'bus', 4: "bicycle"}
       self.track_duration = dict()
       self.presence = defaultdict(set)
       self.trackID_to_class = defaultdict(list)


    def countCarsPerClass(self, trackID, classID):
        class_name = self.class_names[classID]
        if trackID not in self.class_trackIDs:
            self.class_trackIDs[trackID] = class_name

            # Increment class count
            if class_name in self.class_counts:
                self.class_counts[class_name] += 1
            else:
                self.class_counts[class_name] = 1

    def displayStatistics(self):
        df_counts = pd.DataFrame(list(self.class_counts.items()), columns=['Vehicle Class', 'Count'])
        print("Number of Vehicles per Class:")
        print(df_counts)

        # Plot the individual classes
        courses = list(self.class_names.values())
        values = list()
        for index, class_name in enumerate(courses):
            if class_name in self.class_counts.keys():
                values.append(self.class_counts[class_name])
            else:
                values.append(0)
        plt.figure(figsize=(10, 6))
        plt.bar(courses, values)
        plt.title('Number of Vehicles per Class')
        plt.xlabel('Vehicle Class')
        plt.ylabel('Count')
        plt.show()

        # Save the results
        df_counts.to_csv("traffic_statistics.csv", index=False)


    def occuranceDuration(self, trackID, className, currentMinute, frameNumber,
                          currentTime):
        """
            This method tracks how long each vehicle remains in video
        """
        self.presence[currentMinute].add(trackID)

        # Record duration
        if trackID not in self.track_duration:
            # Initialize duration entry
            self.track_duration[trackID] = {
                'start_frame': frameNumber,
                'start_time': currentTime,
                'end_frame': frameNumber,
                'end_time': currentTime
            }
        else:
            # Update end_frame and end_time
            self.track_duration[trackID]['end_frame'] = frameNumber
            self.track_duration[trackID]['end_time'] = currentTime
    
    def processOccuranceDuration(self):
        duration_data = list()
        for track_id, times in self.track_duration.items():
            start_time = times['start_time']
            end_time = times['end_time']
            duration = end_time - start_time  

            # Determine vehicle class (most common class label)
            classes = self.trackID_to_class[track_id]
            most_common_class = max(set(classes), key=classes.count)

            duration_data.append({
                'Track ID': track_id,
                'Vehicle Class': most_common_class,
                'Start Time (s)': start_time,
                'End Time (s)': end_time,
                'Duration (s)': duration
            })

        # Convert to DataFrame
        df_durations = pd.DataFrame(duration_data)
        df_durations.to_csv("MOT_challenge/occurance_duration.csv")
        print(df_durations.head())


"""
    Class representing a vehicles detection and tracking.
"""

class deepSort():
    def __init__(self, model, input_video, output_path, result_output, tracking_ground_truth, mot_results) -> None:
        self.model = model
        self.input_video = input_video
        self.output_path = output_path
        self.result_output = result_output
        self.tracking_ground_truth = tracking_ground_truth
        self.mot_results = mot_results
        self.tracking_results = list()
        self.color_map = dict()
    

    def get_random_color(self):
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    def track_vehicles(self, visible: bool, mot_challenge: bool, write_video: bool):
        tracker = DeepSort(
            max_age=28,
            n_init=2,
            nn_budget=153,
            max_cosine_distance=0.3195
        )

        trafficStatistics = TrafficStatistics()
    
        cap = cv2.VideoCapture(self.input_video)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # type: ignore
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
            
            # Prepare detections for DeepSort (format: [x1, y1, x2, y2, confidence, class])
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
                
                bbox = track.to_ltrb() # get the bounding box values for the visualization
                x1, y1, x2, y2 = bbox
               
                confidence = track.det_conf
                class_id = track.det_class
                visibility = 1

                # Process the trackID and classID in order to make traffic statistics
                trafficStatistics.countCarsPerClass(trackID=track_id, classID=class_id)

                # Save the occurance values per track
                trafficStatistics.trackID_to_class[track_id].append(trafficStatistics.class_names[class_id])
                trafficStatistics.occuranceDuration(track_id, trafficStatistics.class_names[class_id], current_minute,
                                                    frame_number, current_time)


                # Draw the results
                if track_id not in self.color_map:
                    self.color_map[track_id] = self.get_random_color()
                color = self.color_map[track_id] # Get the assigned color

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, f'ID: {track_id}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(frame, f'Class: {self.model.names[class_id]}', (int(x1), int(y1) - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                try:
                    cv2.putText(frame, f'Conf: {confidence:.2f}', (int(x1), int(y1) - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
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
                
                # Transfor the important values for the MOT metrics
                w = x2 - x1
                h = y2 - y1
                if confidence == None:
                    confidence = 0
                self.tracking_results.append([frame_number, track_id, int(x1), int(y1), int(w), int(h), confidence, class_id, visibility])

            # Write the frame to the output video
            if write_video:
                out.write(frame)
                
        cap.release()
        
        if mot_challenge:
            # Convert tracking results to DataFrame in order to evalaute the MOT metrics
            columns = ['frame', 'object_id', 'bbox_left', 'bbox_top', 'bbox_width', 'bbox_height', 'confidence', 'class', 'visibility']
            tracking_df = pd.DataFrame(self.tracking_results, columns=columns)
            # Save the results to a .txt file
            tracking_df.to_csv(self.result_output, index=False, header=False)

        # Process the traffic statistics
        # trafficStatistics.displayStatistics()
        # trafficStatistics.processOccuranceDuration()

        # After processing the video, calculate the MOTA metric
        motaEvaluator = mt.MOTEvaluator(ground_truth_labels=self.tracking_ground_truth, predictions_filename=self.result_output, results_filename=self.mot_results)
        results, acc = motaEvaluator.evaluate()

        mota_value = results.mota["summary"]
        motp_value = results.motp["summary"]
        idf1 = results.idf1["summary"]
        fp = results.num_false_positives["summary"]
        fn = results.num_misses["summary"]
        num_gt = results.num_objects["summary"]
        tp = results.num_matches["summary"]

        # MODA MODP
        MODA = 1 - (fp + fn) / num_gt
        matches = acc.mot_events.query('Type == "MATCH"')
        MODP = 1 - matches['D'].mean()

        # Precision, Recall, F1 score
        if tp + fp > 0:
            precision = tp / (tp + fp)
        else:
            precision = 0.0

        if tp + fn > 0:
            recall = tp / (tp + fn)
        else:
            recall = 0.0

        if precision + recall > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = 0.0

        print("--------")
        print("MODA: " + str(MODA))
        print("MODP: " + str(MODP))
        print("F1-score: " + str(f1_score))
        print("--------")

        
        # Log the metric to wandb
        wandb.log({"MOTA": mota_value,
                   "MOTP": motp_value,
                   "IDF1": idf1,
                   "MODA": MODA,
                   "MODP": MODP,
                   "F1-score": f1_score})
        
    

# Main logig
def main():

    """
        Local configuration
    """
    """
    # Load the pretrained YOLOv8 model
    model = YOLO("/Users/martinkraus/Downloads/custom_vehicles.pt")
    # Define the input video
    input_video = "moving_vehicles2.mp4"
    # Define the path for the resulting video with tracking
    output_path_deepSort = "/Users/martinkraus/Downloads/deepSORT_vehicles.mp4"
    # Define the tracking results path
    results_output_filename = "/Users/martinkraus/Library/CloudStorage/OneDrive-ZápadočeskáuniverzitavPlzni/Dokumenty/škola/DP/YOLO/scripts/MOT_challenge/deepSORT.txt"
    # Define tracking ground truth
    tracking_ground_truth = "/Users/martinkraus/Downloads/MOT17Det/train/MOT17-13/gt/gt_short_edited.txt"
    # Define path for MOT results
    mot_results = "mot_results.txt"

    """
    """
        Metacentrum configuration - MOT Challenge
    """

    
    wandb.init(project="TRACKING_deepSORT", entity="krausm00") 
    # Load the pretrained YOLOv8 model
    model = YOLO("../data/custom_vehicles.pt")
    # Define the input video
    input_video = "../data/moving_vehicles.mp4"
    # Define the path for the resulting video with tracking
    output_path_deepSort = "../video/final_trackings.mp4"
    # Define the tracking results path
    results_output_filename = "../results/tracking_results.txt"
    # Define a path for MOT results
    mot_results = "../results/mot_results.txt"
    # Tracking ground truths
    tracking_ground_truth = "../results/gt.txt"

    deep_sort = deepSort(model, input_video=input_video, output_path=output_path_deepSort, result_output=results_output_filename, 
                        tracking_ground_truth=tracking_ground_truth, mot_results=mot_results)
    deep_sort.track_vehicles(visible=False, mot_challenge=True, write_video=False)
    

    """
        Metacentrum configuration - AICITY Challenge
    """

    """
    wandb.init(project="TRACKING_deepSORT", entity="krausm00") 
    # Load the pretrained YOLOv8 model
    model = YOLO("../../data/custom_vehicles.pt")
    # Define the input video
    input_video = "../data/vdo.avi"
    # Define the path for the resulting video with tracking
    output_path_deepSort = "../video/final_trackings.mp4"
    # Define the tracking results path
    results_output_filename = "../results/tracking_results.txt"
    # Define a path for MOT results
    mot_results = "../results/aicity_results.txt"
    # Tracking ground truths
    tracking_ground_truth = "../results/gt.txt"

    deep_sort = deepSort(model, input_video=input_video, output_path=output_path_deepSort, result_output=results_output_filename, 
                        tracking_ground_truth=tracking_ground_truth, mot_results=mot_results)
    deep_sort.track_vehicles(visible=False, mot_challenge=True, write_video=True)
    """
    


# Run the script
if __name__ == "__main__":
    print("ole")
    main()