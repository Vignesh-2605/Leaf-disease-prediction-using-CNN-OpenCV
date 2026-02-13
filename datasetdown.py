# import kagglehub

# # Download latest version
# path = kagglehub.dataset_download("emmarex/plantdisease")

# print("Path to dataset files:", r"C:\Users\Vignesh S\OneDrive\Documents\Vignesh_S\College SIMATS\ComputerVision\Capstone\Dataset")

import kagglehub

# Download dataset
path = kagglehub.dataset_download("emmarex/plantdisease")

# Print correct downloaded path
print("Dataset downloaded successfully!")
print("Path to dataset files:", path)