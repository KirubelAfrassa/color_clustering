import os
import csv

# Path to the folder containing images
images_folder_path = 'data/clothes'

# Path to the original and new CSV files
original_csv_path = 'data/_classes.csv'
filtered_csv_path = 'data/filtered_class_labels.csv'

# Read the list of image filenames from the folder
image_filenames = set(os.listdir(images_folder_path))

# Columns to remove
columns_to_remove = ['beige', 'black', 'pattren']


def filter_csv_row(row, header):
    """ Filter out unwanted columns and return a new row """
    return [row[header.index(column)] for column in header if column not in columns_to_remove]


def main():
    # Process the CSV file
    with open(original_csv_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        header = next(reader)

        # Filter out the unwanted columns from the header
        filtered_header = [column for column in header if column not in columns_to_remove]

        with open(filtered_csv_path, mode='w', newline='') as new_file:
            writer = csv.writer(new_file)
            writer.writerow(filtered_header)

            for row in reader:
                # Check if the filename in the row matches any image in the folder
                if row[0] in image_filenames:
                    # Write the filtered row to the new CSV file
                    writer.writerow(filter_csv_row(row, header))

    print("Filtered CSV file has been created.")


if __name__ == "__main__":
    main()
