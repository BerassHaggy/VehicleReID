import motmetrics as mm
import pandas as pd

"""
This script represents a computation of MOT metrics from a tracking task.
"""


class MOTEvaluator:
    ROI = (0, 1080, 1920, 320)  # Region of Interest

    def __init__(self, ground_truth_labels, predictions_filename, results_filename):
        self.ground_truth_labels = ground_truth_labels
        self.predictions = predictions_filename
        self.results_filename = results_filename
        self.cars_id = 0  # Check whether the carID is correct (MOT Challenge - classID for vehicles == 3)
        self.ROI = MOTEvaluator.ROI  # Region of Interest

    # Check whether the current bbox is within the defined ROI
    def isInsideROI(self, bbox):
        x1, y1, x2, y2 = bbox
        x_min, y_max, x_max, y_min = self.ROI
        return x1 >= x_min and x2 <= x_max and y1 >= y_min and y2 <= y_max

    def evaluate(self, datasetType: str, includeROI: bool):
        # Read the ground truth and predictions files - AI City Challenge uses 10 labels vs 9 in MOT Challenge
        ground_truth = None
        confidence_threshold = 0.0
        if datasetType.startswith("AI CITY"):
            # Ground truth
            ground_truth = pd.read_csv(
                self.ground_truth_labels,
                header=None,
                names=[
                    'frame', 'object_id', 'bbox_left', 'bbox_top',
                    'bbox_width', 'bbox_height', 'class_id',
                    'unused1', 'unused2', 'unused3'
                ]
            )
            # Set the confidence threshold
            confidence_threshold = 0.6
            self.cars_id = 1

        elif datasetType.startswith("MOT"):
            # Ground truth
            ground_truth = pd.read_csv(
                self.ground_truth_labels,
                header=None,
                names=[
                    'frame', 'object_id', 'bbox_left', 'bbox_top',
                    'bbox_width', 'bbox_height', 'confidence', 'class_id',
                    'visibility'
                ]
            )
            # Set the confidence threshold
            confidence_threshold = 0.8
            self.cars_id = 3

        # Predictions
        predictions = pd.read_csv(
            self.predictions,
            header=None,
            names=[
                'frame', 'object_id', 'bbox_left', 'bbox_top',
                'bbox_width', 'bbox_height', 'confidence', 'class_id',
                'visibility'
            ]
        )
        # Filter the car + truck category if necessary
        ground_truth_cars = ground_truth[ground_truth['class_id'] == self.cars_id]
        predictions_cars = predictions[(predictions['class_id'] == 0) | (predictions['class_id'] == 2)]
        predictions_cars = predictions_cars[predictions_cars['confidence'] >= confidence_threshold]

        # Convert the annotations to motmetrics format
        gt_motmetrics = self.to_motmetrics_format(ground_truth_cars)
        pred_motmetrics = self.to_motmetrics_format(predictions_cars)

        # Get all unique frame numbers
        frame_ids = sorted(set(gt_motmetrics['FrameId'].unique()).union(pred_motmetrics['FrameId'].unique()))

        # Initialize the MOT accumulator
        acc = mm.MOTAccumulator(auto_id=True)

        # Evaluate each frame
        for frame_id in frame_ids:
            gt_frame = gt_motmetrics[gt_motmetrics['FrameId'] == frame_id]
            pred_frame = pred_motmetrics[pred_motmetrics['FrameId'] == frame_id]

            gt_ids = gt_frame['Id'].tolist()
            pred_ids = pred_frame['Id'].tolist()

            gt_bboxes = gt_frame['BBox'].tolist()
            pred_bboxes = pred_frame['BBox'].tolist()

            # Convert bounding boxes to [x1, y1, x2, y2] format
            gt_bboxes_xyxy = [self.bbox_to_xyxy(bbox) for bbox in gt_bboxes]
            pred_bboxes_xyxy = [self.bbox_to_xyxy(bbox) for bbox in pred_bboxes]
            distances = None

            # Whether to take the ROI in account or not
            if includeROI:
                gt_filtered = [(id_, bbox) for id_, bbox in zip(gt_ids, gt_bboxes_xyxy) if self.isInsideROI(bbox)]
                pred_filtered = [(id_, bbox) for id_, bbox in zip(pred_ids, pred_bboxes_xyxy) if
                                 self.isInsideROI(bbox)]

                # Unpack filtered results
                gt_ids_filtered, gt_bboxes_filtered = zip(*gt_filtered) if gt_filtered else ([], [])
                pred_ids_filtered, pred_bboxes_filtered = zip(*pred_filtered) if pred_filtered else ([], [])

                # Check for empty lists
                if not gt_ids_filtered and not pred_ids_filtered:
                    continue

                distances = mm.distances.iou_matrix(gt_bboxes_filtered, pred_bboxes_filtered, max_iou=0.5)
                # Update the accumulator with the frame results
                acc.update(
                    gt_ids_filtered,  # Ground truth IDs
                    pred_ids_filtered,  # Prediction IDs
                    distances  # Distance matrix
                )

            else:
                distances = mm.distances.iou_matrix(gt_bboxes_xyxy, pred_bboxes_xyxy, max_iou=0.5)
                # Update the accumulator with the frame results
                acc.update(
                    gt_ids,               # Ground truth IDs
                    pred_ids,             # Prediction IDs
                    distances             # Distance matrix
                )

        # Compute metrics
        mh = mm.metrics.create()
        metrics = [
            'idf1', 'mota', 'motp', 'precision', 'recall',
            'num_false_positives', 'num_misses', 'num_detections',
            'num_objects', 'num_matches'
        ]
        summary = mh.compute(acc, metrics=metrics, name='summary')

        return summary, acc

    def to_motmetrics_format(self, df):
        # Create a BBox column in [x1, y1, width, height] format
        df['BBox'] = df.apply(
            lambda row: [row['bbox_left'], row['bbox_top'], row['bbox_width'], row['bbox_height']],
            axis=1
        )
        df = df[['frame', 'object_id', 'BBox']]
        df.columns = ['FrameId', 'Id', 'BBox']
        return df

    @staticmethod
    def bbox_to_xyxy(bbox):
        # Convert [x, y, w, h] to [x1, y1, x2, y2]
        x1, y1, w, h = bbox
        x2 = x1 + w
        y2 = y1 + h
        return [x1, y1, x2, y2]
