import json
import os

"""
This script takes a coco annotations from a tracking annotation task (CVAT)
and converts it into a YOLO annotations including the trackIDss

"""

json_path = "/Users/martinkraus/Downloads/coco/annotations/coco_annotations.json"
output_dir = "/Users/martinkraus/Downloads/yolo_trackings"
os.makedirs(output_dir, exist_ok=True)

with open(json_path, "r") as f:
    data = json.load(f)

image_id_to_info = {
    img["id"]: {
        "file_name": img["file_name"],
        "width": img["width"],
        "height": img["height"]
    }
    for img in data["images"]
}

# Process the annotations
annotations_by_image = {}
for ann in data["annotations"]:
    image_id = ann["image_id"]
    bbox = ann["bbox"]  # [x, y, width, height]
    category_id = ann["category_id"] - 1
    track_id = ann.get("attributes", {}).get("track_id", -1)

    image_info = image_id_to_info[image_id]
    width, height = image_info["width"], image_info["height"]

    # Convert bbox to YOLO format
    x_center = (bbox[0] + bbox[2] / 2) / width
    y_center = (bbox[1] + bbox[3] / 2) / height
    norm_width = bbox[2] / width
    norm_height = bbox[3] / height

    yolo_line = f"{category_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f} {track_id}"

    file_name = image_info["file_name"]
    if file_name not in annotations_by_image:
        annotations_by_image[file_name] = []
    annotations_by_image[file_name].append(yolo_line)

# Write to a .txt file for each frameID
for file_name, lines in annotations_by_image.items():
    base_name = os.path.splitext(file_name)[0]
    txt_path = os.path.join(output_dir, f"{base_name}.txt")
    with open(txt_path, "w") as f:
        f.write("\n".join(lines) + "\n")
