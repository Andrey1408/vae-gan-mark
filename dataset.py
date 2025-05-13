import kagglehub

# Download latest version
path = kagglehub.dataset_download("andrey101/marketing-data-new")

print("Path to dataset files:", path)