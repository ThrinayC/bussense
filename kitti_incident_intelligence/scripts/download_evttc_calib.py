import gdown
import os

os.makedirs("data/evttc", exist_ok=True)
gdown.download(id="1fzcRk2-t51whV39nPWZ4V4-2by9Xh0r9", output="data/evttc/calib_camchain.yaml", quiet=False)

if os.path.exists("data/evttc/calib_camchain.yaml"):
    with open("data/evttc/calib_camchain.yaml", "r", encoding="utf-8") as f:
        print(f.read())
