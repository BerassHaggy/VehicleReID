import cv2
import os


def draw_yolo_boxes(img, label_path, class_names):
    """

    :param img: input image
    :param label_path: annotations
    :param class_names: class names
    :return: writes annotations in the provided image
    """
    h, w = img.shape[:2]
    if not os.path.exists(label_path):
        return img  # skip if label doesn't exist
    with open(label_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue  # skip malformed lines
        cls_id, x_center, y_center, width, height = map(float, parts)
        x1 = int((x_center - width / 2) * w)
        y1 = int((y_center - height / 2) * h)
        x2 = int((x_center + width / 2) * w)
        y2 = int((y_center + height / 2) * h)
        cls_name = class_names[int(cls_id)] if int(cls_id) < len(class_names) else str(cls_id)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, cls_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return img


def main():
    # Config
    image_folder = 'images'
    label_folder = 'labels'
    output_video_path = 'output_video.mp4'
    class_names = ['car', 'truck', 'bus']  # Modify as per your dataset
    output_fps = 10

    # Sorted images
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))])
    if not image_files:
        raise ValueError("No image files found.")

    first_image_path = os.path.join(image_folder, image_files[0])
    first_img = cv2.imread(first_image_path)
    height, width = first_img.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, output_fps, (width, height))

    for img_file in image_files:
        img_path = os.path.join(image_folder, img_file)
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(label_folder, label_file)

        img = cv2.imread(img_path)
        if img is None:
            continue
        annotated_img = draw_yolo_boxes(img, label_path, class_names)
        video_writer.write(annotated_img)

    video_writer.release()
    print("Video has been created.")


if __name__ == "__main__":
    main()
