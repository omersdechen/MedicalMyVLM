import cv2
import os
import sys
import albumentations as A
from albumentations.pytorch import ToTensorV2
import CyclicShift as cs

# List of available augmentations
augmentation_dict = {
    "flip": A.HorizontalFlip(p=1),
    "rotate": A.Rotate(limit=45, p=1),
    "zoom": A.Crop(x_min=200, y_min=400, x_max=400, y_max=576, p=1),
    "brightness": A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
    "blur": A.GaussianBlur(blur_limit=(3, 7), p=1),
    "shift": A.Affine(translate_percent=(0.1, 0.2), scale=1.0, rotate=0, shear=0, p=1),
    "cyclicshift": cs.CyclicShift(shift_x=0, shift_y=100),
    "rotate90ccw": "rotate90ccw",  # Placeholder for 90° rotation logic
    "cropleft": A.Crop(x_min=400, y_min=0, x_max=768, y_max=576, p=1),
    "cropright": A.Crop(x_min=0, y_min=0, x_max=200, y_max=576, p=1),
    "croptop": A.Crop(x_min=0, y_min=0, x_max=768, y_max=400, p=1)
}

def apply_augmentation(image_path, augmentation_name, output_folder):
    """
    Apply a selected augmentation on the given image and save the output.

    Parameters:
    - image_path: Path to the input image.
    - augmentation_name: The augmentation to apply (flip, rotate, etc.).
    - output_folder: The folder where the augmented image will be saved.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: The image {image_path} could not be loaded.")
        return

    # Check if the augmentation name exists
    if augmentation_name not in augmentation_dict:
        print(f"Error: '{augmentation_name}' is not a valid augmentation.")
        print("Available augmentations:", list(augmentation_dict.keys()))
        return

    if augmentation_name == "rotate90ccw":
        augmented_image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        # Get the selected augmentation
        augmentation = augmentation_dict[augmentation_name]

        # Apply the augmentation
        augmented_image = augmentation(image=image)['image']

    # Ensure the output folder exists, create it if not
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save the augmented image with a new name
    base_name, ext = os.path.splitext(os.path.basename(image_path))
    output_image_name = f"{base_name}_{augmentation_name}{ext}"
    output_image_path = os.path.join(output_folder, output_image_name)

    # Save the image
    cv2.imwrite(output_image_path, augmented_image)
    print(f"Augmented image saved as: {output_image_path}")

if __name__ == "__main__":
    # Choosing between negative or positive input photos
    choice = input("neg or pos input photos: ")

    # Get the folder path of the current script
    folder_path = os.path.dirname(os.path.abspath(__file__))
    input_folder = os.path.join(folder_path, "photos_to_augment/pos")
    output_base_folder = os.path.join(folder_path, "augmented_photos")  # Base directory to save augmented images

    # Define valid image extensions
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

    while True:
        # List available augmentations with numbering
        augmentation_names = list(augmentation_dict.keys())
        print("Available augmentations:")
        for i, name in enumerate(augmentation_names, start=1):
            print(f"{i}: {name}")

        # Get user input for augmentation choice
        user_input = input("Enter the number corresponding to the augmentation you want to apply (or type 'stop' to finish): ").strip().lower()
        if user_input == "stop":
            print("Stopping the augmentation process.")
            break

        try:
            choice = int(user_input)
            if not (1 <= choice <= len(augmentation_names)):
                raise ValueError("Invalid choice number.")
        except ValueError as e:
            print(f"Error: {e}. Please enter a valid number from the list or 'stop' to exit.")
            continue

        # Get the selected augmentation name
        augmentation_name = augmentation_names[choice - 1]

        # Create the output folder for the chosen augmentation
        output_folder = os.path.join(output_base_folder, f"output_{augmentation_name}")
        os.makedirs(output_folder, exist_ok=True)  # Ensure the folder exists

        # Apply the selected augmentation to every image in input_folder and its subfolders
        for root, _, files in os.walk(input_folder):
            for filename in files:
                file_path = os.path.join(root, filename)  # Get full path
                if os.path.isfile(file_path) and os.path.splitext(filename.lower())[1] in valid_extensions:
                    print(f"Processing file: {file_path}")
                    apply_augmentation(file_path, augmentation_name, output_folder)
                else:
                    print(f"Skipping non-image file: {file_path}")
