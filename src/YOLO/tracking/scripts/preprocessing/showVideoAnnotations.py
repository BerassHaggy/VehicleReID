import cv2
import os
import random
import matplotlib.pyplot as plt

"""
This script creates a video with annotated objects and uses consistent colors for track IDs.
"""

# Assign a unique color to each track_id
track_id_colors = {}


def getColorForTrackID(track_id):
    """
    This function get a random color for each trackID and keeps the color assignment
    :param track_id: object trackID
    :return: color
    """
    if track_id not in track_id_colors:
        random.seed(int(track_id))
        track_id_colors[track_id] = tuple(random.randint(0, 255) for _ in range(3))
    return track_id_colors[track_id]


def drawYoloBoxes(img, label_path, class_names):
    """
    Draws YOLO-format bounding boxes on the image with class labels and track ID colors.
    :param img: input image
    :param label_path: annotations
    :param class_names: class names
    :return: image with annotations
    """
    h, w = img.shape[:2]
    if not os.path.exists(label_path):
        return img

    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 6:
            continue
        cls_id, x_center, y_center, width, height, track_id = parts
        cls_id = int(cls_id)
        track_id = int(track_id)

        x_center, y_center, width, height = map(float, (x_center, y_center, width, height))
        x1 = int((x_center - width / 2) * w)
        y1 = int((y_center - height / 2) * h)
        x2 = int((x_center + width / 2) * w)
        y2 = int((y_center + height / 2) * h)

        class_name = class_names[cls_id] if cls_id < len(class_names) else f"ID {cls_id}"
        color = getColorForTrackID(track_id)

        label_text = f"{class_name} ID:{track_id}"
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
        cv2.putText(img, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return img


def processImage(imagePath: str, labelPath: str, classNames: list):
    """
    This function processes single image with annotations
    :param imagePath: input image
    :param labelPath: corresponding annotations
    :param classNames: class labels
    :return: image with annotations
    """
    image = cv2.imread(imagePath)
    processedImage = drawYoloBoxes(image, labelPath, classNames)
    cv2.imwrite("/Users/martinkraus/Downloads/my_plot.PNG", processedImage)
    return processedImage


def processVideo(imagesFolder: str, labelsFolder: str, classNames: list,
                 outputVideoPath: str, videoFPS: int):
    """
    This function processes every image in a folder and creates a video with annotations
    :param imagesFolder: folder with images
    :param labelsFolder: folder with annotations
    :param classNames: class labels
    :param outputVideoPath: output video path
    :param videoFPS: video frames per second
    :return: none
    """
    # Sorted images
    image_files = sorted([f for f in os.listdir(imagesFolder) if f.endswith(('.jpg', '.png', '.PNG'))])
    if not image_files:
        raise ValueError("No image files found.")

    first_image_path = os.path.join(imagesFolder, image_files[0])
    first_img = cv2.imread(first_image_path)
    height, width = first_img.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(outputVideoPath, fourcc, videoFPS, (width, height))

    for index, img_file in enumerate(image_files):
        if index == 601:
            break
        img_path = os.path.join(imagesFolder, img_file)
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(labelsFolder, label_file)

        img = cv2.imread(img_path)
        if img is None:
            continue

        annotated_img = drawYoloBoxes(img, label_path, classNames)
        video_writer.write(annotated_img)

    video_writer.release()
    print("Video has been created.")


def main():
    """
    Main function
    :return:
    """
    # Config for video
    image_folder = '/Users/martinkraus/Downloads/coco/images/'
    label_folder = '/Users/martinkraus/Downloads/yolo_trackings/'
    output_video_path = 'pilsen_video_annotated.mp4'
    class_names = ['Car', 'Bus', 'Tram', 'Motorcycle', 'Cycle', 'Truck']
    output_fps = 10

    # Config for single images
    imagePath = "/Users/martinkraus/Downloads/coco/images/frame_000570.PNG"
    labelsPath = "/Users/martinkraus/Downloads/yolo_trackings/frame_000570.txt"
    processImage(imagePath, labelsPath, class_names)


if __name__ == "__main__":
    main()
