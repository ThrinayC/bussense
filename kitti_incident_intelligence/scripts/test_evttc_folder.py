import gdown

# Test inspecting the Google Drive folder for download-full-sequence
folder_id = "1J5urgKdVasGXghyO2m7Mt0fpzEiAEXRy"
try:
    print(f"Listing folder {folder_id}...")
    files = gdown.download_folder(id=folder_id, skip_download=True)
    print("Files in folder:", len(files) if files else 0)
    if files:
        for f in files[:10]:
            print(" ", f)
except Exception as e:
    print("Listing folder error:", e)
