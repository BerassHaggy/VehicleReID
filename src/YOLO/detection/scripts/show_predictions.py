from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt



def show_yolo_boxes(image_path):
    image = cv2.imread(image_path)
    boxes = load_ground_truth_boxes(image_path)
    for box in boxes:
        converted_box = convert_yolo_box(box)
        x, y, w, h = converted_box[0], converted_box[1], converted_box[2], converted_box[3]
        cv2.rectangle(image, (int(x), int(y)), (int(x) + int(w), int(y) + int(h)), (0, 255, 0), 2)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure()
    plt.imshow(image_rgb)
    # plt.title("Ground truth boxes")
    plt.axis("off")
    plt.show()

    return image


def load_ground_truth_boxes(image_path):
    boxes = []
    labels = image_path[0:-4] + ".txt"
    with open(labels, "r") as fr:
        lines = fr.readlines()
        for line in lines:
            line = line.split()
            boxes.append([float(line[1]), float(line[2]), float(line[3]), float(line[4])])
    return boxes


def convert_yolo_box(bounding_box):
    img_width = 1280  # 640, 720, 1280
    img_height = 720  # 480, 1280, 720
    x_center, y_center, width, height = bounding_box
    x = (x_center - width / 2) * img_width
    y = (y_center - height / 2) * img_height
    w = width * img_width
    h = height * img_height
    return [x, y, w, h]


def show_yolo_predictions(model, image):
    # Perform detection with YOLOv8
    image = cv2.imread(image)
    results = model(image)
    detections = results[0].boxes.data.tolist()
    dets = []
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    
    plt.figure()
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.show()

image_path = "data/custom_dataset/640.png"
model = YOLO("/Users/martinkraus/Downloads/custom_vehicles.pt")
show_yolo_boxes(image_path=image_path)
show_yolo_predictions(model=model, image=image_path)