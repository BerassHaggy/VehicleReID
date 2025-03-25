import pandas as pd

"""
This script filters a ground truth frames of a given video based on passed
parameters.
"""
# Load the ground-truth file
# gt_file = "/Users/martinkraus/Downloads/MOT17Det/train/MOT17-13/gt/gt_short.txt"
gt_file = "/Users/martinkraus/Downloads/AICity22_Track1_MTMC_Tracking/train/S01/c001/gt/gt.txt"
# Specify the frame range of the cropped video
start_frame = 600
end_frame = 1200
datasetType = "AI CITY"  # AI CITY || MOT

# Load the ground-truth data
# MOT
if datasetType.startswith("MOT"):
    gt_df = pd.read_csv(gt_file, header=None,
                        names=['frame', 'object_id', 'bbox_left', 'bbox_top', 'bbox_width', 'bbox_height', 'confidence',
                               'class', 'visibility'])
    cropped_gt_df = gt_df[(gt_df['frame'] >= start_frame) & (gt_df['frame'] <= end_frame)].copy()
# AI CITY
elif datasetType.startswith("AI CITY"):
    gt_df = pd.read_csv(gt_file, header=None,
                        names=[
                            'frame', 'object_id', 'bbox_left', 'bbox_top',
                            'bbox_width', 'bbox_height', 'class_id',
                            'unused1', 'unused2', 'unused3'
                        ])
    cropped_gt_df = gt_df[(gt_df['frame'] >= start_frame) & (gt_df['frame'] <= end_frame)].copy()
    cropped_gt_df.loc[:, 'frame'] = cropped_gt_df['frame'] - start_frame + 1

# Save the filtered ground truth to a new file
cropped_gt_file = "/Users/martinkraus/Downloads/AICity22_Track1_MTMC_Tracking/train/S01/c001/gt/gt_short.txt"
cropped_gt_df.to_csv(cropped_gt_file, header=False, index=False)
