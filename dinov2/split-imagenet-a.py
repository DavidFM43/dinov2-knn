import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split
from torchvision.datasets import ImageFolder
import argparse

def main(dataset_root, train_size, seed):
    # Use ImageFolder to easily extract the images paths from the folders
    help_dataset = ImageFolder(root=dataset_root)
    samples = help_dataset.samples
    images, labels = zip(*samples)
    
    # We now make a stratified split of the dataset so we get
    # roughly the same proportions of classes in each split
    indices = np.arange(len(labels))
    train_indices, val_indices, train_labels, val_labels = train_test_split(
        indices,
        labels,
        test_size=1 - train_size,
        stratify=labels,
        random_state=seed
    )
    
    # Get the images of each split.
    train_images = [images[idx] for idx in train_indices]
    val_images = [images[idx] for idx in val_indices]

    # We now separate the images splits into their respective folders.
    base_dir = f"{dataset_root}-split-{train_size}-seed-{seed}"
    train_dir = os.path.join(base_dir, "train")
    os.makedirs(train_dir, exist_ok=True)
    for file_path in train_images:
        # Get both the image file as the class folder.
        class_folder = os.path.basename(os.path.dirname(file_path))
        file_name = os.path.basename(file_path)
        combined_name = os.path.join(class_folder, file_name)
        # Construct the destination path.
        destination_path = os.path.join(train_dir, combined_name)
        # Ensure the destination directory exists
        destination_dir_path = os.path.dirname(destination_path)
        os.makedirs(destination_dir_path, exist_ok=True)
        # Move the file
        shutil.copy(file_path, destination_path)

    val_dir = os.path.join(base_dir, "val")
    os.makedirs(val_dir, exist_ok=True)
    for file_path in val_images:
        # Get both the image file as the class folder.
        class_folder = os.path.basename(os.path.dirname(file_path))
        file_name = os.path.basename(file_path)
        combined_name = os.path.join(class_folder, file_name)
        # Construct the destination path.
        destination_path = os.path.join(val_dir, combined_name)
        # Ensure the destination directory exists
        destination_dir_path = os.path.dirname(destination_path)
        os.makedirs(destination_dir_path, exist_ok=True)
        # Move the file
        shutil.copy(file_path, destination_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into train and validation sets.")
    parser.add_argument('--dataset_root', type=str, required=True, help='Root directory of the dataset.')
    parser.add_argument('--train_size', type=float, required=True, help='Proportion of the dataset to include in the train split.')
    parser.add_argument('--seed', type=int, required=True, help='Random seed for reproducibility.')
    args = parser.parse_args()

    main(args.dataset_root, args.train_size, args.seed)
