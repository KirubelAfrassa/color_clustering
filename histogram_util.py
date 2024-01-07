import cv2
import os

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

def process_images(image_folder):
    # List to store histograms
    histograms = []

    # Process each image in the folder
    for image_name in os.listdir(image_folder):
        # Construct the full path to the image
        image_path = os.path.join(image_folder, image_name)

        # Read the image
        image = cv2.imread(image_path)

        # Check if the image has been successfully loaded
        if image is not None:
            # Convert image to RGB (OpenCV uses BGR by default)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Calculate the histograms for each channel
            hist_r = cv2.calcHist([image_rgb], [0], None, [256], [0, 256])
            hist_g = cv2.calcHist([image_rgb], [1], None, [256], [0, 256])
            hist_b = cv2.calcHist([image_rgb], [2], None, [256], [0, 256])

            # Normalize the histograms
            hist_r = hist_r / hist_r.sum()
            hist_g = hist_g / hist_g.sum()
            hist_b = hist_b / hist_b.sum()

            # Append histograms to the list
            histograms.append((hist_r, hist_g, hist_b))

            # Save histograms with the original image name as part of the file name
            for color, hist, color_name in zip(['r', 'g', 'b'], [hist_r, hist_g, hist_b], ['Red', 'Green', 'Blue']):
                plt.figure(figsize=(10, 3))
                plt.title(f"{color_name} Histogram for {image_name}")
                plt.plot(hist, color=color)
                plt.xlim([0, 256])
                histogram_filename = f"histograms/{os.path.splitext(image_name)[0]}_{color_name}_histogram.png"
                plt.savefig(histogram_filename)
                plt.close()

        else:
            print(f"Failed to load image: {image_name}")

    return histograms


def main():
    # Path to the folder containing images
    image_folder_path = 'data/clothes'

    # Process the images and get their histograms
    image_histograms = process_images(image_folder_path)

# Now, image_histograms contains the histograms of all processed images


if __name__ == "__main__":
    main()