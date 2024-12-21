import os
import json
import nibabel as nib
import matplotlib.pyplot as plt
import sys
import cv2


# Get user input for slice choice
try:
    slice = int(input("Enter the number corresponding to the slice you want to get: "))
    if not (-100 <= slice <= 100):
        raise ValueError("Invalid slice number.")
except ValueError as e:
    print(f"Error: {e}. Please enter a valid number (-100,100).")
    sys.exit(1)


# Determine the slice suffix
slice_suffix = f"n{abs(slice)}" if slice < 0 else str(slice)
if slice_suffix == 0: slice_suffix = ""

# Define paths
folder_path = os.path.dirname(os.path.abspath(__file__))
input_folder = os.path.join(folder_path, 'input_nifti_files')
output_folder = os.path.join(folder_path, f"photos/{slice_suffix}")
cases_hyoid_json_path = os.path.join(input_folder, 'cases_hyoid.json')
captions_file_path = os.path.join(output_folder, f"captions_{slice_suffix}.json")  # one next slice

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Load cases_hyoid JSON data
with open(cases_hyoid_json_path, 'r') as f:
    cases_hyoid = json.load(f)

# List of descriptions corresponding to outputs
outputs = [
    "The hyoid bone is positioned at the same level as the mandibular plane",
    "The hyoid bone is positioned slightly inferior to the mandibular plane",
    "The hyoid bone is positioned inferior to the mandibular plane",
    "The hyoid bone is positioned significantly inferior to the mandibular plane"
]

# Initialize captions dictionary
captions = {}

# Process each case
for case_id, info in cases_hyoid.items():
    try:
        # Handle key name differences (sagittal/saggital)
        sagittal_key = 'sagittal' if 'sagittal' in info else 'saggital'
        sagittal_index = info[sagittal_key] + slice # one next slice
        
        # Construct file path for the NIfTI file
        nifti_file_path = os.path.join(input_folder, f"{case_id}.nii.gz")
        
        # Load the NIfTI file
        nifti_img = nib.load(nifti_file_path)
        img_data = nifti_img.get_fdata()
        
        # Extract the sagittal slice
        sagittal_slice = img_data[sagittal_index, :, :]
        
        # Save the sagittal slice as an image
        output_image_path = os.path.join(output_folder, f"{case_id}_{slice_suffix}.jpg")# one next slice
        sagittal_slice = cv2.rotate(sagittal_slice, cv2.ROTATE_90_COUNTERCLOCKWISE) # comes out rotated
        plt.imsave(output_image_path, sagittal_slice, cmap='gray')

        # Add description to captions
        description = outputs[info['output']]
        captions[f"{case_id}_{slice_suffix}.jpg"] = [description]# one next slice
        
        print(f"Processed case {case_id} and saved image.")
    
    except Exception as e:
        print(f"Error processing case {case_id}: {e}")

# Save captions to JSON file
with open(captions_file_path, 'w') as f:
    json.dump(captions, f, indent=4)

print(f"All images processed. Captions saved to {captions_file_path}.")
