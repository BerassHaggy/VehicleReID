from ultralytics import YOLO
import cv2

"""
This script represents a YOLO prediction on a provided image.
"""
datasetType = "CD"  # CD or HV
model = YOLO("best.pt")
if datasetType.startswith("CD"):
    image_path = "CD_val_img.png"
    image = cv2.imread(image_path)
    label_map = {
        0: 'Car',
        1: 'Motorcycle',
        2: 'Truck',
        3: 'Bus',
        4: 'Bicycle'
    }
elif datasetType.startswith("HV"):
    image_path = "HV_val_img.jpg"
    image = cv2.imread(image_path)
    label_map = {
        0: 'big bus',
        1: 'big truck',
        2: 'bus-l-',
        3: 'bus-s-',
        4: 'car',
        5: 'mid truck',
        6: 'small bus',
        7: 'small truck',
        8: 'truck-l-',
        9: 'truck-m-',
        10: 'truck-s-',
        11: 'truck-xl-'
    }

classes = list(label_map.keys())

results = model(image)

for result in results:
    boxes = result.boxes
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf[0]
        cls = int(box.cls[0])

        class_name = classes[cls]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)

        # label = f"{class_name}: {conf:.2f}"
        # cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

cv2.imwrite("YOLO_prediction.jpg", image)
