import motmetrics as mm
import pandas as pd

class MOTEvaluator():
    def __init__(self, ground_truth_labels, predictions_filename) -> None:
        self.ground_truth_labels = ground_truth_labels
        self.predictions = predictions_filename
        self.cars_id = 3 # class id 3 represents cars in the MOT dataset

    def evaluate(self, results_output_filename):
        ground_truth = pd.read_csv(self.ground_truth_labels, header=None, 
                           names=['frame', 'object_id', 'bbox_left', 'bbox_top', 'bbox_width', 'bbox_height', 'confidence', 'class', 'visibility'])
        predictions = pd.read_csv(self.predictions, header=None, 
                           names=['frame', 'object_id', 'bbox_left', 'bbox_top', 'bbox_width', 'bbox_height', 'confidence', 'class', 'visibility'])
        # Filter the car category
        ground_truth_cars = ground_truth[ground_truth['class'] == self.cars_id]
        predictions_cars = predictions[predictions['class'] == 0]

        # Convert the annotations to motMetrics
        gt_motmetrics = self.to_motmetrics_format(ground_truth_cars, 'groundtruth')
        pred_motmetrics = self.to_motmetrics_format(predictions_cars, 'hypothesis') 

        # Create a list of unique frame IDs
        frame_ids = sorted(ground_truth_cars['frame'].unique())

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
            
            # Calculate distance matrix (IoU) between ground truth and predictions
            dists = mm.distances.iou_matrix(gt_bboxes, pred_bboxes, max_iou=0.5)
            
            acc.update(gt_ids, pred_ids, dists)

        # Compute metrics
        mh = mm.metrics.create()
        summary = mh.compute(acc, metrics=['idf1', 'mota', 'motp'], name='summary')

        # Write the summary to a .txt file
        with open(results_output_filename, "w") as fw:
            fw.write(summary.to_string()) # type: ignore
        print(summary)


    def to_motmetrics_format(self, df, label):
        df['bbox'] = df.apply(lambda row: [row['bbox_left'], row['bbox_top'], row['bbox_width'], row['bbox_height']], axis=1)
        df = df[['frame', 'object_id', 'bbox']]
        df.columns = ['FrameId', 'Id', 'BBox']
        df['Label'] = label
        return df 
    
# Main logic
def main():
    # Define the directory path
    dir_path = "/Users/martinkraus/Library/CloudStorage/OneDrive-ZápadočeskáuniverzitavPlzni/Dokumenty/škola/DP/YOLO/scripts/MOT_challenge/"
    # Define the file where the tracking results are stored
    results_output_filename = dir_path + "deepSORT.txt"
    # Ground truth labels
    ground_truth = "/Users/martinkraus/Downloads/MOT17Det/train/MOT17-13/gt/gt_short_edited.txt"
    # Define a path where the MOT metrics results should be stored
    results_filename = dir_path + "deepSORT_results.txt"

    # Call the MotEvaluator
    motEvaluator = MOTEvaluator(ground_truth_labels=ground_truth, predictions_filename=results_output_filename)
    motEvaluator.evaluate(results_filename)

if __name__ == "main":
    main()
