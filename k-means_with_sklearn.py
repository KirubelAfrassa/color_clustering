import matplotlib
import numpy as np
from sklearn.cluster import KMeans
import os
import cv2
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


def load_histograms(image_folder):
    histograms = []

    for image_name in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_name)
        image = cv2.imread(image_path)

        if image is not None:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            total_pixels = image.shape[0] * image.shape[1]  # Total number of pixels in the image

            hist_r = cv2.calcHist([image_rgb], [0], None, [256], [0, 256]).flatten() / total_pixels
            hist_g = cv2.calcHist([image_rgb], [1], None, [256], [0, 256]).flatten() / total_pixels
            hist_b = cv2.calcHist([image_rgb], [2], None, [256], [0, 256]).flatten() / total_pixels

            combined_hist = np.concatenate((hist_r, hist_g, hist_b))
            histograms.append(combined_hist)
        else:
            print(f"Failed to load image: {image_name}")

    return np.array(histograms)


def cluster_histograms(histograms, k=5):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(histograms)
    return kmeans.labels_


def main():
    matplotlib.use('Agg')
    # Path to the folder containing images
    image_folder_path = 'data/clothes'

    # Load the histograms
    image_histograms = load_histograms(image_folder_path)

    # Cluster the histograms
    cluster_labels = cluster_histograms(image_histograms)

    # Output the cluster labels
    print("Cluster labels for each image:")
    print(cluster_labels)

    # Load the CSV file
    df = pd.read_csv('data/filtered_class_labels.csv')

    # Extract ground truth labels
    # Assuming the class with '1' is the true class for each image
    true_labels = df[['blue', 'gray', 'green', 'red', 'white']].idxmax(axis=1)

    # Map color names to integers (consistent with your k-means output)
    label_mapping = {'blue': 0, 'gray': 1, 'green': 2, 'red': 3, 'white': 4}
    true_labels = true_labels.map(label_mapping)

    # Calculate metrics
    conf_matrix = confusion_matrix(true_labels, cluster_labels)
    accuracy = accuracy_score(true_labels, cluster_labels)
    precision = precision_score(true_labels, cluster_labels, average='macro')
    recall = recall_score(true_labels, cluster_labels, average='macro')
    f1 = f1_score(true_labels, cluster_labels, average='macro')

    ax = plt.subplot()
    sns.heatmap(conf_matrix, annot=True, fmt='g', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Expected')
    ax.set_title('Confusion Matrix')
    plt.savefig(f"confusion_matrix/sklean-k-means-confusion-matrix.png")
    plt.close()

    # Output the results
    print("Confusion Matrix:")
    print(conf_matrix)
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")


if __name__ == "__main__":
    main()

