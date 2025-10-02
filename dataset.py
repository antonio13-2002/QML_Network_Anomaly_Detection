import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

'''I create 2 functions to generate and visualize the dataset.
Each data has 4 features, plus a label to distinguish them between anomalies and normal data "packets".
The function generate_dataset4() create a dataset of N samples, each sample is a 4-dimensional vector.
The function plot_data() visualizes all the samples in a 2D plot, where each point is colored according to its label.
'''
feature_names = ['Duration','BytesTransferred','DestPortFrequency','UnusualFlags']
feature_names10 = ['Duration','BytesTransferred','DestPortFrequency','UnusualFlags','PacketSizeMean','PacketSizeStdDev',
                   'Protocol','InterPacketArrivalTimeMean','SessionStartTimeOfDay','NumberOfConnectionsToIP']

num_features_low = len(feature_names) # 4 features as an example
num_features_high = len(feature_names10) # 10 features as an example

def generate_dataset4(num_packets):
    # Generate random data for 4 features, we the 80% of the data as normal and 20% as anomalies
    # Generate normal data
    normal_mean_data = [10, 10000, 2, 0.05] # mean values for all the 4 features
    std_dev_normal = [5, 5000, 1, 0.02] # standard deviation for all the 4 features
    normal_packets = np.random.normal(
        loc = normal_mean_data, scale = std_dev_normal, size = (int(num_packets * 0.8),num_features_low)
    )
    label_normals = np.zeros(int(num_packets * 0.8))  # label 0 for normal packets

    # Generate anomalous data, here we assume 2 types of anomalies
    # Anomaly type 1: scanning attack: low duration, low bytes transferred, high destination port frequency, no unsual flags
    anomaly1_mean_data = [0.1, 100, 50, 0.05]
    std_dev_anomaly1 = [0.05, 50, 10, 0.01]
    # We generate a random quantity of the 20% of the total anomalies
    num_anomaly1 = np.random.randint(1, int(num_packets * 0.2))
    anomaly1_packets = np.random.normal(
        loc = anomaly1_mean_data, scale = std_dev_anomaly1, size = (num_anomaly1, num_features_low)
    )
    # Anomaly type 2: Data exfiltration: high duration, high bytes transferred, low destination port frequency, 1 unusual flag
    anomaly2_mean_data = [1000, 5000000, 1, 0.8]
    std_dev_anomaly2 = [500, 1000000, 0.5, 0.1]
    num_anomaly2 = int(num_packets * 0.2) - num_anomaly1
    anomaly2_packets = np.random.normal(
        loc = anomaly2_mean_data, scale = std_dev_anomaly2, size = (num_anomaly2, num_features_low)
    )

    label_anomalies = np.ones(int(num_packets * 0.2)) # all the anomalies are labeled as 1
    labels = np.concatenate((label_normals, label_anomalies), axis = None)
    # Combine data and labels
    packets = np.concatenate((normal_packets, anomaly1_packets, anomaly2_packets), axis = 0)
    data = np.concatenate((packets, labels.reshape(labels.shape[0],1)), axis = 1)

    # Shuffle the dataset
    np.random.shuffle(data)
    normalize(data)

    return data


def plot_data(data_with_labels):
    # Separate features (X) from labels (y)
    X_scaled = data_with_labels[:, :-1]  # All columns except the last one are features
    y = data_with_labels[:, -1]          # The last column is the label

    plot_combinations = [
        (0, 1, 2, "Duration, BytesTransferred, DestPortFrequency"),
        (0, 1, 3, "Duration, BytesTransferred, UnusualFlags"),
        (2, 3, 1, "DestPortFrequency, UnusualFlags, BytesTransferred")
    ]

    for idx_x, idx_y, idx_z, title_suffix in plot_combinations:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # plot all points of normal class
        ax.scatter(X_scaled[y == 0, idx_x],
                   X_scaled[y == 0, idx_y],
                   X_scaled[y == 0, idx_z],
                   label='Normal (0)', alpha=0.6, edgecolors='w', s=50)

        # plot all points of anomaly class
        ax.scatter(X_scaled[y == 1, idx_x],
                   X_scaled[y == 1, idx_y],
                   X_scaled[y == 1, idx_z],
                   label='Anomaly (1)', alpha=0.8, edgecolors='w', color='red', s=50)

        # Set axis labels
        ax.set_xlabel(f'{feature_names[idx_x]} (Scaled)')
        ax.set_ylabel(f'{feature_names[idx_y]} (Scaled)')
        ax.set_zlabel(f'{feature_names[idx_z]} (Scaled)')
        ax.set_title(f'Dataset in 3D: {title_suffix}')
        ax.legend()
        plt.tight_layout()
        plt.show()


def gen_dataset_high(num_packets):
    normal_mean_data = [10, 10000, 2, 0.05, 1000, 200, 0.5, 0.01, 12, 5]
    std_dev_normal = [5, 5000, 1, 0.02, 400, 150, 0.2, 0.005, 6, 3]
    normal_packets = np.random.normal(
        loc = normal_mean_data, scale = std_dev_normal, size = (int(num_packets * 0.7), num_features_high)
    )

    label_normals = np.zeros(int(num_packets * 0.7))
    # Generate anomalous data, here we assume 3 types of anomalies
    # Anomaly type 1: scanning attack: low duration, low bytes transferred, high destination port frequency, low mean packet size, low inter packet arrival
    anomaly1_mean_data = [0.1, 100, 50, 0.05, 150, 20, 1.0, 0.0001, 2, 100]
    std_dev_anomaly1 = [0.05, 50, 10, 0.01, 50, 10, 0, 0.00005, 1, 50]
    num_anomaly1 = int(num_packets * 0.1)
    anomaly1_packets = np.random.normal(
        loc = anomaly1_mean_data, scale = std_dev_anomaly1, size = (num_anomaly1, num_features_high)
    )
    # Anomaly type 2: Data exfiltration: high duration, high bytes transferred, low destination port frequency, high mean packet size, low inter packet arrival
    anomaly2_mean_data = [1000, 5000000, 1, 0.8, 1400, 50, 0.0, 0.1, 10, 1]
    std_dev_anomaly2 = [500, 1000000, 0.5, 0.1, 100, 30, 0, 0.05, 2, 0.5]
    num_anomaly2 = int(num_packets * 0.1)
    anomaly2_packets = np.random.normal(
        loc = anomaly2_mean_data, scale = std_dev_anomaly2, size = (num_anomaly2, num_features_high)
    )
    # Anomaly type 3: DDoS attack: low duration, low bytes transferred, high destination port frequency, variable mean packet size, very high number of connections to IP
    anomaly3_mean_data = [10, 10000, 50, 0.05,1000, 100, 1.0, 0.001, 18, 1000]
    std_dev_anomaly3 = [5, 5000, 10, 0.01, 200, 50, 0, 0.0001, 4, 500]
    num_anomaly3 = int(num_packets * 0.1)
    anomaly3_packets = np.random.normal(
        loc = anomaly3_mean_data, scale = std_dev_anomaly3, size = (num_anomaly3, num_features_high)
    )

    label_anomalies = np.ones(int(num_packets * 0.3)) # all the anomalies are labeled as 1
    labels = np.concatenate((label_normals, label_anomalies), axis = None)

    # Combine data and labels
    packets = np.concatenate((normal_packets, anomaly1_packets, anomaly2_packets,anomaly3_packets), axis = 0)
    data = np.concatenate((packets, labels.reshape(labels.shape[0],1)), axis = 1)

    # Shuffle the dataset
    np.random.shuffle(data)
    normalize(data)

    return data

def normalize(data):
    # here we first normalize the data, then we split it into training and test sets
    scaler = MinMaxScaler(feature_range = (0, 1),copy = True)
    data[:, :-1] = scaler.fit_transform(data[:, :-1])  # normalize the features


if __name__ == "__main__":
    num_packets = 200
    data = gen_dataset_high(num_packets)
    print(data)
    plot_data(data)