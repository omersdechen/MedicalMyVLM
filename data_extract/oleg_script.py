import nibabel as nib
import matplotlib.pyplot as plt
import os

# For Hyoid cases, use only the sagittal slice
cases_hyoid = { 3742:{'sagittal':384, 'output':0},
                1723:{'sagittal':384, 'output':3},
                3111:{'sagittal':384, 'output':3},
                2975:{'sagittal':384, 'output':0},
                2710:{'sagittal':384, 'output':2},
                2189:{'sagittal':384, 'output':0},
                1603:{'sagittal':384, 'output':2},
                3777:{'sagittal':384, 'output':0},
                3759:{'sagittal':384, 'output':3},
                3689:{'sagittal':384, 'output':1},
                6565:{'sagittal':384, 'output':1},
                6567:{'sagittal':384, 'output':2},
                6568:{'sagittal':384, 'output':3},
                6578:{'sagittal':384, 'output':0},
                6516:{'sagittal':384, 'output':3},
                6515:{'sagittal':384, 'output':0},
                6511:{'sagittal':384, 'output':1},
                6522:{'sagittal':384, 'output':1},
                6523:{'sagittal':384, 'output':2},
                6528:{'sagittal':384, 'output':3},
                6531:{'sagittal':384, 'output':3},
                6532:{'sagittal':384, 'output':2},
                6535:{'sagittal':384, 'output':3},
                6537:{'sagittal':384, 'output':0},
                6551:{'sagittal':384, 'output':2},
                3681:{'sagittal':384, 'output':2},
                3444:{'sagittal':384, 'output':3},
                6525:{'sagittal':384, 'output':1},
                6529:{'sagittal':384, 'output':2},
                6559:{'sagittal':384, 'output':3},
                6583:{'sagittal':384, 'output':2},
                6004: {'saggital': 382, 'output': 0},
                6006: {'saggital': 382, 'output': 3},
                6007: {'saggital': 382, 'output': 2},
                6010: {'saggital': 382, 'output': 3},
                6011: {'saggital': 382, 'output': 2},
                6013: {'saggital': 382, 'output': 1},
                6016: {'saggital': 382, 'output': 3},
                6020: {'saggital': 382, 'output': 0},
                6021: {'saggital': 382, 'output': 3},
                6023: {'saggital': 382, 'output': 3},
                6024: {'saggital': 382, 'output': 1},
                6025: {'saggital': 382, 'output': 0},
                6031: {'saggital': 382, 'output': 1},
                6033: {'saggital': 382, 'output': 2},
                6036: {'saggital': 382, 'output': 2},
                6037: {'saggital': 382, 'output': 0},
                6039: {'saggital': 382, 'output': 3},
                6040: {'saggital': 382, 'output': 3},
                6047: {'saggital': 382, 'output': 3},
                6051: {'saggital': 382, 'output': 1},
                6052: {'saggital': 382, 'output': 1}}

# list of the desired outputs, can also be treated as class labels
outputs = ["The hyoid bone is positioned at the same level as the mandibular plane",
           "The hyoid bone is positioned slightly inferior to the mandibular plane",
           "The hyoid bone is positioned inferior to the mandibular plane",
           "The hyoid bone is positioned significantly inferior to the mandibular plane"]

# Load the NIfTI file
nifti_path = '/home/user_7734/omer/data_extract/olegs_drive'  # replace with your file path

case_idx = 0 # case to read and show the slices
key = list(cases_hyoid.keys())[case_idx]

nifti_file_path = os.path.join(nifti_path, str(key) + '.nii.gz')

nifti_img = nib.load(nifti_file_path)

# Access the image data
img_data = nifti_img.get_fdata()

# Display basic information about the image
print(f"Image shape: {img_data.shape}")
print(f"Image affine:\n{nifti_img.affine}")
print(f"Desired output:\n{outputs[cases_hyoid[key]['output']]}")











# plt.figure()
# plt.imshow(img_data[:, img_data.shape[1] // 2, :], cmap="gray")
# plt.title("Coronal Slice of the NIfTI Image")

# plt.figure()
# plt.imshow(img_data[:, :, img_data.shape[2] // 2], cmap="gray")
# plt.title("Axial Slice of the NIfTI Image")

# # Display the middle slice of the first axis
# plt.figure()
# plt.imshow(img_data[cases_hyoid[key]['sagittal'], :, :], cmap="gray")
# plt.title(f"sagittal Slice {cases_hyoid[key]['sagittal']} of the NIfTI Image")

# plt.show()

######## JUNK #########
# for key in cases_hyoid:
#     nifti_file_path = os.path.join(nifti_path, str(key) + '.nii.gz')
#     nifti_img = nib.load(nifti_file_path)
#     print(f"case {key}, slice {nifti_img.shape[0]//2}")
