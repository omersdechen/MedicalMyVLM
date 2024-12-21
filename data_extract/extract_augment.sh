#!/bin/bash

# creating photos from nifti files
python /home/user_7734/data_extract/nifti2jpg/nifti_to_jpg_cap.py

# moving output photos
# Source and destination folders
SOURCE_FOLDER="/home/user_7734/data_extract/nifti2jpg/photos"
DESTINATION_FOLDER="/home/user_7734/data_extract/augmentation/photos_to_augment/"

# Empty the destination folder before moving new photos (optional: keep the folder, just clear contents)
if [ -d "$DESTINATION_FOLDER" ]; then
    rm -rf "$DESTINATION_FOLDER"/*
    echo "Cleared the contents of the destination folder: $DESTINATION_FOLDER"
else
    echo "Destination folder does not exist: $DESTINATION_FOLDER"
fi

# Create the destination folder if it doesn't exist
if [ ! -d "$DESTINATION_FOLDER" ]; then
    mkdir -p "$DESTINATION_FOLDER"
    echo "Created destination folder: $DESTINATION_FOLDER"
fi

# Move photos from source to destination
if [ -d "$SOURCE_FOLDER" ]; then
    mv "$SOURCE_FOLDER"/* "$DESTINATION_FOLDER"/
    echo "Moved photos from $SOURCE_FOLDER to $DESTINATION_FOLDER"
else
    echo "Source folder does not exist: $SOURCE_FOLDER"
fi

# augmenting the photos
python /home/user_7734/data_extract/augmentation/augmentation.py
