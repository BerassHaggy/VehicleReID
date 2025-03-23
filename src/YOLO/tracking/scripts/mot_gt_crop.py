import pandas as pd

# Load the ground-truth file
gt_file = "/Users/martinkraus/Downloads/MOT17Det/train/MOT17-13/gt/gt_short.txt"

# Specify the frame range of the cropped video
start_frame = 250
end_frame = 375

# Load the ground-truth data
gt_df = pd.read_csv(gt_file, header=None, 
                    names=['frame', 'object_id', 'bbox_left', 'bbox_top', 'bbox_width', 'bbox_height', 'confidence', 'class', 'visibility'])

# Filter by the frame range
cropped_gt_df = gt_df[(gt_df['frame'] >= start_frame) & (gt_df['frame'] <= end_frame)]

# Save the filtered ground truth to a new file
cropped_gt_file = "/Users/martinkraus/Downloads/MOT17Det/train/MOT17-13/gt/gt_short_edited.txt"
cropped_gt_df.to_csv(cropped_gt_file, header=False, index=False)