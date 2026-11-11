from huggingface_hub import hf_hub_download

try:
    path = hf_hub_download(
        repo_id="Stary108/Extreme_Driving_Conditions_Dataset",
        filename="train/Critical_Driving/Emergency_Braking/episode_6/calibration.json",
        repo_type="dataset"
    )
    print("Successfully downloaded via huggingface_hub:", path)
    with open(path, "r") as f:
        print(f.read()[:300])
except Exception as e:
    print("huggingface_hub download error:", type(e), e)
