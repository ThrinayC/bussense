import gdown

folder_id = "1sx4oej7Uy3NNxY9AE-azsd9TkfuhAYkA"
try:
    print(f"Listing calibration folder {folder_id}...")
    files = gdown.download_folder(id=folder_id, skip_download=True)
    if files:
        for f in files:
            print(" ", f)
except Exception as e:
    print("Error:", e)
