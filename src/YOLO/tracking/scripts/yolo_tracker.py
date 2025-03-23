from collections import defaultdict
import cv2
import pandas as pd
import random
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

class Tracker():
    def __init__(self, model, input_video, output_path, results_output) -> None:
        self.model = model
        self.input_video = input_video
        self.output_path = output_path
        # Allocate a list for tracking results
        self.tracking_results = list()
        self.results_output = results_output

    def track_vehicles(self, visible):
        cap = cv2.VideoCapture(self.input_video)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

    
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        out = cv2.VideoWriter(self.output_path, fourcc, fps, (frame_width, frame_height))

        # Store the track history
        track_history = defaultdict(lambda: [])

        # Loop through the video frames
        while cap.isOpened():
            
            # Read a frame from the video
            success, frame = cap.read()

            if success:
                # YOLOv8 tracking
                results = model.track(frame, persist=True, tracker="bytetrack.yaml")
                frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

                if results[0].boxes.id is not None:
                    # Save the tracking results
                    for i in range(len(results[0].boxes)):
                        object_id = int(results[0].boxes.id[i].item())
                        x_center, y_center, width, height = results[0].boxes.xywh[i].tolist()
                        bbox_left = x_center - width / 2
                        bbox_top = y_center - height / 2
                        bbox_width = width
                        bbox_height = height
                        confidence = results[0].boxes.conf[i].item()
                        class_id = int(results[0].boxes.cls[i].item())
                        visibility = 1  

                        self.tracking_results.append([frame_number, object_id, bbox_left, bbox_top, bbox_width, bbox_height, confidence, class_id, visibility])

            
                    # Get the boxes and track IDs
                    boxes = results[0].boxes.xywh.cpu()
                    track_ids = results[0].boxes.id.int().cpu().tolist()
                else:
                    boxes = []
                    track_ids = []

                # Visualize the results on the frame
                annotated_frame = results[0].plot()

                """
                # Plot the tracks
                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    track = track_history[track_id]
                    track.append((float(x), float(y)))  # x, y center point
                    if len(track) > 30:  # retain 30 tracks for 30 frames
                        track.pop(0)
                    
                    
                    # Draw the tracking lines
                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)
                """
                    
                # Write the annotated frame to the output video
                out.write(annotated_frame)

                if visible:
                    # Display the annotated frame
                    cv2.imshow("YOLOv8 Tracking", annotated_frame)

                    # Break the loop if 'q' is pressed
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
            else:
                # Break the loop if the end of the video is reached
                break
        
        columns = ['frame', 'object_id', 'bbox_left', 'bbox_top', 'bbox_width', 'bbox_height', 'confidence', 'class', 'visibility']
        tracking_df = pd.DataFrame(self.tracking_results, columns=columns)
        # Save the results to a .txt file
        tracking_df.to_csv(self.results_output, index=False, header=False)
        return tracking_df
        
    
class deepSort():
    def __init__(self, model, input_video, output_path, result_output) -> None:
        self.model = model
        self.input_video = input_video
        self.output_path = output_path
        self.result_output = result_output
        self.tracking_results = list()
        self.color_map = dict()
        self.aplha = 0.5 # smoothing factor
        self.exp_smoothed_bboxes = dict()
        self.kalman_bboxes = dict()

    def get_random_color(self):
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    def track_vehicles(self, visible: bool, exponential_smoothing: bool, kalman_filtering: bool, recalibrating: bool,
                       show_raw_detections: bool):
        tracker = DeepSort(max_age=30, nn_budget=70)
    
        cap = cv2.VideoCapture(self.input_video)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # type: ignore
        out = cv2.VideoWriter(self.output_path, fourcc, fps, (frame_width, frame_height))

        ret, prev_frame = cap.read()

        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        frame_number = 0
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame_number += 1


            """
            # Convert the current frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Calculate optical flow between previous and current frames
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

            # Generate a grid of the same size as the flow
            h, w = flow.shape[:2]
            flow_map = np.meshgrid(np.arange(w), np.arange(h))
            flow_map = np.array(flow_map).transpose(1, 2, 0).astype(np.float32)

            # Combine flow map with optical flow displacement
            flow_map[..., 0] += flow[..., 0]
            flow_map[..., 1] += flow[..., 1]

            # Ensure remapping coordinates are within image bounds
            flow_map[..., 0] = np.clip(flow_map[..., 0], 0, w - 1)
            flow_map[..., 1] = np.clip(flow_map[..., 1], 0, h - 1)

            # Apply remapping to stabilize the frame
            stabilized_frame = cv2.remap(frame, flow_map, None, cv2.INTER_LINEAR)

            prev_frame = frame
            prev_gray = gray

            """


            # Perform detection with YOLOv8
            results = model(frame)
            
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
                #dets.append([[x1, y1, x2, y2], conf, int(cls)])
                
            # Whether to show the raw detections based on the YOLO model
            if show_raw_detections:
                for det in dets:
                    x1, y1, x2, y2 = det[0]
                    conf = det[1]
                    cls = det[2]
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                    cv2.putText(frame, f'Conf: {conf:.2f}', (int(x1), int(y1) - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.putText(frame, f'Class: {self.model.names[cls]}', (int(x1), int(y1) - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.imshow("YOLOv8 Tracking", frame)

                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break


            # Update tracker
            tracks = tracker.update_tracks(dets, frame=frame)

            # Convert current frame to grayscale for optical flow
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Store tracking results
            for track in tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                track_id = track.track_id
                bbox = track.to_tlwh()
                ltrb = track.to_ltrb()
                bbox = ltrb
                x1, y1, x2, y2 = bbox
                """
                x1, y1, x2, y2 = bbox
                w = x2 - x1
                h = y2 - y1
                """
                confidence = track.det_conf
                class_id = track.det_class
                visibility = 1

        
                # Whether to use recalibrating in between two images
                if recalibrating:
                    # Prepare for keypoint recalibration
                    # Define a bounding box region where keypoints will be detected
                    mask = np.zeros_like(prev_gray)
                    mask[int(y1):int(y1 + h), int(x1):int(x1 + w)] = 255

                    # Detect keypoints inside the bounding box (using Shi-Tomasi corners)
                    kp_prev = cv2.goodFeaturesToTrack(prev_gray, mask=mask, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

                    if kp_prev is not None:
                        # Track keypoints from the previous to the current frame using optical flow
                        kp_current, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, kp_prev, None)

                        # Keep only successfully tracked keypoints (status == 1)
                        good_new = kp_current[status == 1]
                        good_old = kp_prev[status == 1]

                        # If enough good keypoints are tracked, recalibrate the bounding box
                        if len(good_new) > 0:
                            shift_x, shift_y = np.mean(good_new - good_old, axis=0)

                            # Adjust bounding box with the shift values
                            x1 += shift_x
                            y1 += shift_y
                            x2 += shift_x
                            y2 += shift_y

                    # Calculate new width and height of the bounding box
                    w = x2 - x1
                    h = y2 - y1

                # Decide whether to use the exponential moving average for smoothing the bboxes between each frame
                if exponential_smoothing:
                    if track_id not in self.exp_smoothed_bboxes:
                        self.exp_smoothed_bboxes[track_id] = [x1, y1, x2, y2]
                    # Make the smoothing based on the history    
                    else:
                        self.exp_smoothed_bboxes[track_id][0] = self.aplha * x1 + (1 - self.aplha) * self.exp_smoothed_bboxes[track_id][0]
                        self.exp_smoothed_bboxes[track_id][1] = self.aplha * y1 + (1 - self.aplha) * self.exp_smoothed_bboxes[track_id][1]
                        self.exp_smoothed_bboxes[track_id][2] = self.aplha * x2 + (1 - self.aplha) * self.exp_smoothed_bboxes[track_id][2]
                        self.exp_smoothed_bboxes[track_id][3] = self.aplha * y2 + (1 - self.aplha) * self.exp_smoothed_bboxes[track_id][3]
        
                    # Get the values based on the exponential smoothing
                    x1, y1, x2, y2 = self.exp_smoothed_bboxes[track_id]
                    w = x2 - x1
                    h = y2 - y1
                
                # Decide whether to use Kalman filtering
                if kalman_filtering:
                    if track_id not in self.kalman_bboxes:
                        self.kalman_bboxes[track_id] = KalmanFilter([x1, y1, x2, y2])
                    else:
                        kalman_tracker = self.kalman_bboxes[track_id]
                        kalman_tracker.predict()
                        kalman_tracker.update([x1, y1, x2, y2])
                        x1, y1, x2, y2 = kalman_tracker.get_state()
                        w = x2 - x1
                        h = y2 - y1

                    self.tracking_results.append([frame_number, track_id, x1, y1, w, h, confidence, class_id, visibility])


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
                    print("")
                    
                # Decide whether to display the annotated frame
                if visible:
                    # Display the annotated frame
                    cv2.imshow("YOLOv8 Tracking", frame)

                    # Break the loop if 'q' is pressed
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break


                # Write the frame to the output video
                out.write(frame)
                
        cap.release()

        # Convert tracking results to DataFrame
        columns = ['frame', 'object_id', 'bbox_left', 'bbox_top', 'bbox_width', 'bbox_height', 'confidence', 'class', 'visibility']
        tracking_df = pd.DataFrame(self.tracking_results, columns=columns)
        # Save the results to a .txt file
        tracking_df.to_csv(self.result_output, index=False, header=False)
        return tracking_df
    

class KalmanFilter():
    def __init__(self, bbox):
        self.dt = 1.0  
        self.u = 0  
        self.state = np.matrix([
            [bbox[0]],  # x1
            [bbox[1]],  # y1
            [bbox[2] - bbox[0]],  # width
            [bbox[3] - bbox[1]],  # height
            [0],  # velocity in x
            [0]  # velocity in y
        ])
        self.P = np.eye(6) * 1000  # covariance 
        self.F = np.matrix([
            [1, 0, 0, 0, self.dt, 0],
            [0, 1, 0, 0, 0, self.dt],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        self.H = np.matrix([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0]
        ])
        self.R = np.eye(4) * 10  # measurement noise covariance
        self.Q = np.eye(6)  # process noise covariance

    def predict(self):
        self.state = self.F * self.state + self.u
        self.P = self.F * self.P * self.F.T + self.Q
        return self.state[0:4]
    
    def update(self, bbox):
        Z = np.matrix([
            [bbox[0]],
            [bbox[1]],
            [bbox[2] - bbox[0]],
            [bbox[3] - bbox[1]]
        ])
        y = Z - self.H * self.state
        S = self.H * self.P * self.H.T + self.R
        K = self.P * self.H.T * np.linalg.inv(S)
        self.state = self.state + K * y
        self.P = (np.eye(self.H.shape[1]) - K * self.H) * self.P

    def get_state(self):
        x1 = self.state[0, 0]
        y1 = self.state[1, 0]
        x2 = self.state[0, 0] + self.state[2, 0]
        y2 = self.state[1, 0] + self.state[3, 0]
        return [x1, y1, x2, y2]


# Load the pretrained YOLOv8 model
model = YOLO("/Users/martinkraus/Downloads/custom_vehicles.pt")
# Define the input video
input_video = "moving_vehicles.mp4"
# Define the output path
output_path = "/Users/martinkraus/Downloads/vehicles_tracking.mp4"
# Define the results output file
results_output_filename = "vehicles_tracking.txt"


"""
    YOLOv8 botSort or ByteTrack
"""
tracker = Tracker(model, input_video=input_video, output_path=output_path, results_output=results_output_filename)
#results = tracker.track_vehicles(visible=False)

"""
    DeepSORT
"""
output_path_deepSort = "/Users/martinkraus/Downloads/detection_test.mp4"
results_output_filename = "CD_tracking_deepSort_exp_smoothing.txt"
deep_sort = deepSort(model, input_video=input_video, output_path=output_path_deepSort, result_output=results_output_filename)
results = deep_sort.track_vehicles(visible=True, exponential_smoothing=False, kalman_filtering=False, recalibrating=False, 
                                   show_raw_detections=False)

"""
    Kalman + DeepSORT
"""
output_path_deepSort = "/Users/martinkraus/Downloads/CD_tracking_deepSort_kalman.mp4"
results_output_filename = "CD_tracking_deepSort_kalman.txt"
deep_sort = deepSort(model, input_video=input_video, output_path=output_path_deepSort, result_output=results_output_filename)
#results = deep_sort.track_vehicles(visible=False, exponential_smoothing=False, kalman_filtering=True)