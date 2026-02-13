import kagglehub

# Download latest version of the dataset 
path = kagglehub.dataset_download("johnchfr/isic-2017")

print("Path to dataset files:", path)