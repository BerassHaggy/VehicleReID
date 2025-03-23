import pandas as pd
import json


annotations = []
annotations_path = "/Users/martinkraus/Downloads/custom_vehicles"

train_path_images = annotations_path + "/train"
train_path_annotations = train_path_images + "/annotations.json"

test_path_images = annotations_path + "/test"
test_path_annotations = test_path_images + "/annotations.json"

val_path_images = annotations_path + "/valid"
val_path_annotations = val_path_images + "/annotations.json"

annotations.append(train_path_annotations)
annotations.append(test_path_annotations)
annotations.append(val_path_annotations)

categories_list = []
annotation_instances = {}
n_of_images_per_class = {}
n_of_images_per_class["train"] = 0
n_of_images_per_class["validation"] = 0
n_of_images_per_class["test"] = 0

def get_categories(data):
    categories = []
    for i in range(len(data)):
        categories.append(data[i]["name"])
        
    return categories


for annotation in annotations:
    with open(annotation) as fr:
        data = json.load(fr)

        # get the individual categories if not done yet
        if len(categories_list) == 0:
            categories = data["categories"]
            categories_list = get_categories(categories)
        
        # get the number of images per set
        images = data["images"]
        if "train" in annotation:
            n_of_images_per_class["train"] = len(images)
        elif "valid" in annotation:
            n_of_images_per_class["validation"] = len(images)
        elif "test" in annotation:
            n_of_images_per_class["test"] = len(images)

        # convert json to pandas
        df = pd.json_normalize(data["annotations"])

        # get the annotation instances for each category
        for i in range(len(categories)):
            annotation_instances_count = df[df['category_id'] == i].shape[0]
            try:
                annotation_instances[categories_list[i]] += annotation_instances_count
            except:
                annotation_instances[categories_list[i]] = annotation_instances_count

# print the results - number of images
total_images = 0
for value in n_of_images_per_class.values():
    total_images += value

for key,value in n_of_images_per_class.items():
    percentage = (value / total_images) * 100
    print("Number of images in " + str(key) + " set: " + str(value) + " " + str(percentage) + "%.")

# number of annotation instances
for key, value in annotation_instances.items():
    print("Category: " + str(key) + " has: " + str(value) + " annotation instances.")



