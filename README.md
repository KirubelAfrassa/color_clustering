# K-Means Clustering for Image Color Classification

## Abstract
This study focused on the application of k-means clustering to categorize images based on their color histograms. The objective was to automatically classify images into five predefined color classes (blue, gray, green, red, and white) using a machine learning approach. The k-means algorithm was implemented from scratch and applied to a dataset of images, each labeled with one of the five color classes. The clustering process involved extracting and normalizing the color histograms of each image, followed by the application of the k-means algorithm to group these images into clusters.

The effectiveness of the clustering was evaluated using a confusion matrix and standard classification metrics. The results indicated an overall average accuracy of 25%, with precision and recall both also aligning at 25%. The confusion matrix revealed the distribution of predictions across the clusters, showing varying degrees of overlap between the different color classes. These results suggest that while the k-means algorithm was able to categorize images into distinct groups based on color similarities, the level of accuracy achieved indicates room for improvement.

## Keywords
K-means, Clustering, Color Detection

## Introduction
This project delves into the realm of machine learning, specifically focusing on the use of the k-means clustering algorithm for image classification based on color histograms. The primary task involves designing and implementing a system that can effectively cluster images by analyzing and categorizing them into predefined color groups: blue, gray, green, red, and white. This approach is rooted in the field of unsupervised learning, where the algorithm discerns inherent patterns and similarities in the data without explicit labeling. The practical applications of this study are vast and varied, extending to areas such as digital image processing, automated photo organization and retrieval, color-based content filtering in digital libraries, and even in the development of tools for visual art analysis.

## System Design
The system designed for clustering images based on color histograms using a self-implemented k-means algorithm involves several key processing steps:

1. **Data Collection and Preprocessing:**
   - Collecting a dataset of images, each classified into one of five color classes: blue, gray, green, red, and white.
   - 20 images were selected for each class, totaling 100 images used.
   - Preprocessing includes reading the images from the dataset and converting them into a suitable format for analysis.

2. **Histogram Extraction:**
   - For each image, the system extracts color histograms by processing the RGB (Red, Green, Blue) channels.
   - A histogram for each color channel is created by counting how many pixels in the image have each possible intensity value (0-255).
   - Each histogram (for R, G, and B) will be an array of 256 elements.
   - These histograms provide a way to analyze and compare the color distributions of different images.

3. **Normalization:**
   - Each histogram is normalized, scaling the pixel intensities to a range of [0-1].

4. **Feature Vector Creation:**
   - The normalized histograms of the three color channels (R, G, B) are concatenated to form a single feature vector for each image.

5. **K-Means Clustering:**
   - The k-means algorithm is implemented from scratch, starting with the random initialization of five centroids.
   - It iteratively assigns each image to the nearest centroid based on Euclidean distance.

6. **Evaluation:**
   - The clustering result is evaluated using accuracy, precision, recall, and F1 score.

## Conclusion
The clustering process demonstrated a moderate level of success in categorizing images into predefined color classes using a self-implemented k-means algorithm. The accuracy achieved was around 20-27%, suggesting room for improvement. Factors such as the k-means algorithm's sensitivity to initial centroid selection and the use of color histograms as the sole feature for clustering might have influenced the results. Future work could explore more sophisticated feature extraction methods, alternative clustering algorithms, or a larger and more diverse dataset.

### Suggested Improvements
- **Feature Enhancement:** Incorporating additional features beyond color histograms.
- **Advanced Clustering Algorithms:** Exploring more sophisticated clustering algorithms.
- **Centroid Initialization:** Implementing more robust methods for centroid initialization.
- **Post-Processing:** Introducing post-processing steps for cluster refinement.

## Appendix
The Python implementation can be found in the GitHub repository. The project structure is as follows:

- **Data:** Contains the image dataset.
- **Histograms:** Stores the generated histograms.
- **Confusion Matrix:** Contains the heat map generated after each run of k-means.
- **Code:** Includes Python scripts for preprocessing, histogram generation, and k-means implementation.

Refer to the repository for detailed code and documentation.
