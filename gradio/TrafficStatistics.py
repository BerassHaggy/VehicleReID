from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd

"""
This class computes and stores from a video the following metrics:
    - number of instances per class
    - duration of each VehicleID (including startTime and endTime)
"""


class TrafficStatistics:
    def __init__(self) -> None:
        self.class_trackIDs = dict()
        self.class_counts = dict()
        self.class_names = {0: 'car', 1: 'motorbike', 2: 'truck', 3: 'bus', 4: "bicycle"}
        self.track_duration = dict()
        self.presence = defaultdict(set)
        self.trackID_to_class = defaultdict(list)

    def countCarsPerClass(self, trackID, classID):
        """
        This method tracks a number of cars within each class.
        :param trackID: ID of tracked vehicle
        :param classID: ID of a corresponding class
        :return:
        """
        class_name = self.class_names[classID]
        if trackID not in self.class_trackIDs:
            self.class_trackIDs[trackID] = class_name

            # Increment class count
            if class_name in self.class_counts:
                self.class_counts[class_name] += 1
            else:
                self.class_counts[class_name] = 1

    def displayStatistics(self):
        """
        This method displays the calculated statistics-
        :return:
        """
        df_counts = pd.DataFrame(list(self.class_counts.items()), columns=['Vehicle Class', 'Count'])
        print("Number of Vehicles per Class:")
        print(df_counts)

        # Plot the individual classes
        courses = list(self.class_names.values())
        values = list()
        for index, class_name in enumerate(courses):
            if class_name in self.class_counts.keys():
                values.append(self.class_counts[class_name])
            else:
                values.append(0)
        plt.figure(figsize=(10, 6))
        plt.bar(courses, values)
        plt.title('Number of Vehicles per Class')
        plt.xlabel('Vehicle Class')
        plt.ylabel('Count')
        plt.show()

        # Save the results
        df_counts.to_csv("traffic_statistics.csv", index=False)

    def occuranceDuration(self, trackID, className, currentMinute, frameNumber,
                          currentTime):
        """
            This method tracks how long each vehicle (trackID) remains in video
        """
        self.presence[currentMinute].add(trackID)

        # Record duration
        if trackID not in self.track_duration:
            # Initialize duration entry
            self.track_duration[trackID] = {
                'start_frame': frameNumber,
                'start_time': currentTime,
                'end_frame': frameNumber,
                'end_time': currentTime
            }
        else:
            # Update end_frame and end_time
            self.track_duration[trackID]['end_frame'] = frameNumber
            self.track_duration[trackID]['end_time'] = currentTime

    def processOccuranceDuration(self):
        """
        This method processes individual trackIDs and their occurrence time.
        :return:
        """
        duration_data = list()
        for track_id, times in self.track_duration.items():
            start_time = times['start_time']
            end_time = times['end_time']
            duration = end_time - start_time

            # Determine vehicle class (most common class label)
            classes = self.trackID_to_class[track_id]
            most_common_class = max(set(classes), key=classes.count)

            duration_data.append({
                'Track ID': track_id,
                'Vehicle Class': most_common_class,
                'Start Time (s)': start_time,
                'End Time (s)': end_time,
                'Duration (s)': duration
            })

        # Convert to DataFrame
        df_durations = pd.DataFrame(duration_data)
        df_durations.to_csv("MOT_challenge/occurance_duration.csv")
        print(df_durations.head())
