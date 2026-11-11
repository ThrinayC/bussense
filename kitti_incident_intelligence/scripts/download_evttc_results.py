import gdown
import os

gdown.download(id="1ygH5Zw5LchktoS7m8OTIqnjN54zMtwEU", output="data/evttc/calib_results.txt", quiet=False)
if os.path.exists("data/evttc/calib_results.txt"):
    with open("data/evttc/calib_results.txt", "r", encoding="utf-8") as f:
        print(f.read()[:2000])
