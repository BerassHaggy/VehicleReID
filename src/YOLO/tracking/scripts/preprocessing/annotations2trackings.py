import os


def processLines(annotationsFolderPath: str, numOfImages: int):
    all_lines = []
    sorted_files = sorted([f for f in os.listdir(annotationsFolderPath) if f.endswith('.txt')])

    for idx, file_name in enumerate(sorted_files):
        if idx <= numOfImages:
            file_path = os.path.join(annotationsFolderPath, file_name)
            with open(file_path, 'r') as f:
                lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line:
                    all_lines.append((idx, line))
    return all_lines


def createTrackings(annotationsFolderPath: str, outputFilePath: str, numOfImages: int):
    lines = processLines(annotationsFolderPath, numOfImages)

    with open(outputFilePath, 'w') as out_file:
        for frame_idx, line in lines:
            out_file.write(f"{frame_idx} {line}\n")

    print("Ground truth .txt file created")


def main():
    annotationsFolderPath = "/Users/martinkraus/Downloads/yolo_trackings"
    outputFilePath = "/Users/martinkraus/Downloads/pilsen_gt.txt"
    numOfImages = 600
    createTrackings(annotationsFolderPath, outputFilePath, numOfImages)


if __name__ == "__main__":
    main()
