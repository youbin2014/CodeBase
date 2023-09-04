import os
import shutil


def reorganize_dataset(original_folder, destination_folder):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Loop through each subfolder in the original folder
    for subfolder_name in os.listdir(original_folder):
        # Extract the class name (first word before the hyphen)
        class_name = subfolder_name.split('-')[0].strip()

        # Create the class folder in the destination folder if it doesn't exist
        dest_class_folder = os.path.join(destination_folder, class_name)
        if not os.path.exists(dest_class_folder):
            os.makedirs(dest_class_folder)

        # Copy each file from the subfolder to the new class folder
        subfolder_path = os.path.join(original_folder, subfolder_name)
        for filename in os.listdir(subfolder_path):
            file_path = os.path.join(subfolder_path, filename)
            dest_file_path = os.path.join(dest_class_folder, filename)

            # Handle filename collisions by renaming
            counter = 1
            while os.path.exists(dest_file_path):
                dest_file_path = os.path.join(dest_class_folder, f"{counter}_{filename}")
                counter += 1

            shutil.copy(file_path, dest_file_path)


if __name__ == '__main__':
    original_folder = '/mnt/data/semantic_confusion_dataset/labeled_images_pairs_CIFAR10'
    destination_folder = '/mnt/data/semantic_confusion_dataset/additional_CIFAR10_dataset'  # replace with your desired path
    reorganize_dataset(original_folder, destination_folder)
