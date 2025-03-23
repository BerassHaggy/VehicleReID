import json
import os
from PIL import Image

# Set the paths for the input and output directories
input_dir = '/Users/martinkraus/Downloads/dataset-vehicles/train/images'
output_dir = '/Users/martinkraus/Downloads/custom_vehicles/train'

# Define the categories for the COCO dataset
categories = [{
                "id": 0,
                "name": "Car"   
    },
    {
                "id": 1,
                "name": "Motorcycle"
    },
    {
                "id": 2,
                "name": "Truck"
    },
    {
                "id": 3,
                "name": "Bus"
    },
    {
                "id": 4,
                "name": "Bicycle"
    }]


# Define the COCO dataset dictionary
coco_dataset = {
    "info": {},
    "licenses": [],
    "categories": categories,
    "images": [],
    "annotations": []
}
image_index = 0
# Loop through the images in the input directory
for image_file in os.listdir(input_dir):
    
    # Load the image and get its dimensions
    image_path = os.path.join(input_dir, image_file)
    image = Image.open(image_path)
    width, height = image.size
    
    # Add the image to the COCO dataset
    image_dict = {
        "id": image_index,
        "licence": 0,
        "file_name": image_file,
        "height": height,
        "width": width
    }
    coco_dataset["images"].append(image_dict)
    
    # Load the bounding box annotations for the image
    try:
        with open(os.path.join(input_dir, f'{"../labels/" + image_file.split(".")[0]}.txt')) as f:
            annotations = f.readlines()
    except:
        annotations = []
    
    if len(annotations) != 0:
        # Loop through the annotations and add them to the COCO dataset
        for ann in annotations:
            x, y, w, h = map(float, ann.strip().split()[1:])
            x_min, y_min = int((x - w / 2) * width), int((y - h / 2) * height)
            x_max, y_max = int((x + w / 2) * width), int((y + h / 2) * height)
            ann_dict = {
                "id": len(coco_dataset["annotations"]),
                "image_id": image_index,
                "category_id": int(ann[0]),
                "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                "area": (x_max - x_min) * (y_max - y_min),
                "segmentation": [],
                "iscrowd": 0
            }
            coco_dataset["annotations"].append(ann_dict)
    
    # Process new image
    image_index += 1

# Save the COCO dataset to a JSON file
with open(os.path.join(output_dir, 'annotations.json'), 'w') as f:
    json.dump(coco_dataset, f)